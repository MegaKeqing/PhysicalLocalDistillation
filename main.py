"""统一入口：python main.py --config configs/cifar10_teacher.json。"""
import argparse
import json
import os
from pathlib import Path

# 必须在第一次 CUDA 运算之前设置；严格确定性失败时直接报告。
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from pld.data import build_loaders
from pld.models import build_model
from pld.trainers import BPTrainer, PLDTrainer, fit
from pld.utils import file_sha256, project_path, provenance, seed_everything, validate_config, write_json


def main():
    """读取 JSON，拒绝覆盖已有运行，再构建教师或学生并启动所选训练器。"""
    parser = argparse.ArgumentParser(description="Physical Local Distillation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="仅覆盖输出目录，供独立重复运行使用")
    parser.add_argument("--check-epochs", type=int, help="仅运行前 N 轮复现检查，不评估测试集")
    args = parser.parse_args()
    args.config = project_path(args.config)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.check_epochs is not None and not 1 <= args.check_epochs <= config["training"]["epochs"]:
        parser.error("--check-epochs must be between 1 and the configured epoch budget")
    seed_everything(config["seed"], config.get("threads", 4))
    device = torch.device(config["device"])
    output = project_path(args.output or config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    manifest = provenance(Path(__file__).resolve().parent, device)
    manifest["config_sha256"] = file_sha256(args.config)
    manifest["command_config"] = str(args.config)
    manifest["output_dir"] = str(output)
    manifest["check_epochs"] = args.check_epochs
    write_json(output / "manifest.json", manifest)
    write_json(output / "status.json", {"status": "STARTING"})
    try:
        data_config = dict(config["data"], root=str(project_path(config["data"]["root"])))
        loaders = build_loaders(data_config)
        model = build_model(config["model"])
        if config["task"] == "pld":
            path = project_path(config["pld"]["teacher_checkpoint"])
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("format_version") != 1:
                raise ValueError("Teacher checkpoint must come from this project")
            teacher_config = checkpoint["config"]
            for key in ("dataset", "size", "valid_size"):
                if teacher_config["data"][key] != config["data"][key]:
                    raise ValueError(f"Teacher/student data mismatch: {key}")
            teacher = build_model(teacher_config["model"])
            teacher.load_state_dict(checkpoint["model"])
            manifest["teacher_checkpoint_sha256"] = file_sha256(path)
            manifest["teacher_config"] = teacher_config
            write_json(output / "manifest.json", manifest)
            trainer = PLDTrainer(model, teacher, config["training"], config["pld"],
                                 config["data"]["size"], device)
        else:
            trainer = BPTrainer(model, config["training"], device)
        fit(trainer, loaders, config, output, check_epochs=args.check_epochs)
    except BaseException as error:
        write_json(output / "status.json", {"status": "FAILED", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
