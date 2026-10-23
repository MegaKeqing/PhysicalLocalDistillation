"""确定性设置、配置校验与运行证据记录。"""
import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
import numpy as np
import torch
import torchvision
from PIL import __version__ as pillow_version


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    """将相对路径统一解析到项目根目录，同时支持绝对路径和用户主目录。"""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def git_output(root, *arguments):
    """Git 不可用或目录不属于仓库时返回 None，允许源码压缩包直接运行。"""
    if not (Path(root) / ".git").exists():
        return None
    try:
        result = subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def seed_everything(seed, threads=4):
    """启用确定性算法并固定各随机流；不静默容忍非确定性 CUDA 算子。"""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def write_json(path, payload):
    """用原子替换写 JSON，避免读到尚未写完整的状态文件。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def file_sha256(path):
    """分块计算文件摘要，用于标识源码、配置与权重。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(state):
    """按键与张量内容计算摘要，排除 torch.save 容器格式对重复性比较的影响。"""
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((tensor.dtype, tuple(tensor.shape))).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def provenance(root, device):
    """记录环境、Git 版本及本次源码摘要；源码压缩包无需安装 Git。"""
    root = Path(root)
    paths = [root / "main.py", *sorted((root / "pld").glob("*.py"))]
    return {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "torchvision": str(torchvision.__version__), "numpy": np.__version__, "pillow": pillow_version,
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": git_output(root, "rev-parse", "HEAD"),
        "git_status": git_output(root, "status", "--porcelain"), "deterministic": True,
        "source_sha256": {str(p.relative_to(root)): file_sha256(p) for p in paths}}


def validate_config(config):
    """拒绝未知字段，避免拼写错误或不支持的配置项静默失效。"""
    schemas = {
        "root": {"task", "seed", "device", "threads", "output_dir", "data", "model", "training", "pld"},
        "data": {"dataset", "root", "size", "batch_size", "valid_size", "train_size", "download",
                 "num_workers", "pin_memory", "augmentation", "shuffle_seed"},
        "training": {"epochs", "learning_rate", "weight_decay", "early_stopping"},
        "pld": {"teacher_checkpoint", "mapping", "losses", "projection_seed"}}
    for section, allowed in schemas.items():
        payload = config if section == "root" else config.get(section, {})
        if set(payload) - allowed:
            raise ValueError(f"Unknown {section} fields: {set(payload) - allowed}")
    if config["task"] not in {"teacher", "bp", "pld", "pld-seq", "ti-pld"}:
        raise ValueError("task must be teacher, bp, pld, pld-seq or ti-pld")
    if config["data"]["dataset"] not in {"mnist", "fmnist", "cifar10"}:
        raise ValueError("Unknown dataset")
    if config["data"]["size"] != config["model"].get("size", config["data"]["size"]):
        raise ValueError("Model and data image sizes differ")
    if config["training"]["epochs"] <= 0:
        raise ValueError("epochs must be positive")
    early = config["training"].get("early_stopping")
    if early and (set(early) != {"min_epochs", "patience"} or min(early.values()) <= 0):
        raise ValueError("early_stopping requires positive min_epochs and patience")
    if config["task"] == "pld" and config["model"]["type"] not in {"dpnn", "mlp"}:
        raise ValueError("PLD students must be DPNN or MLP")
    if config["task"] in {"pld-seq", "ti-pld"}:
        if config["model"]["type"] != "dpnn":
            raise ValueError("Staged distillation requires a DPNN student")
        if config["data"]["augmentation"]:
            raise ValueError("Fixed speckle caches require augmentation=[]")
        expected = "pmse" if config["task"] == "pld-seq" else "mse"
        losses = config["pld"]["losses"]
        if len(losses) != config["model"].get("num_blocks", 2) + 1:
            raise ValueError("One loss is required per hidden block and classifier")
        if any(loss["type"] != expected for loss in losses[:-1]) or losses[-1]["type"] != "kl_ce":
            raise ValueError(f"Use {expected} for hidden blocks and kl_ce for the classifier")
    for seed in (config["seed"], config["model"]["init_seed"], config["data"]["shuffle_seed"],
                 config.get("pld", {}).get("projection_seed", 0),
                 config["model"].get("tm_seed", 0)):
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seeds must be integers in [0, 2**32)")
