"""在空闲 GPU 上运行六项完整实验，并以相同 JSON 的前五轮 loss 验收重复性。"""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pld.utils import file_sha256, project_path, write_json

CHECK_EPOCHS = 5


def gpu_available(index):
    """查询物理 GPU UUID，拒绝在已有计算进程的 GPU 上启动新任务。"""
    uuid = subprocess.check_output(
        ["nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader"],
        text=True).strip()
    running = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True)
    if any(line.split(",")[0].strip() == uuid for line in running.splitlines()):
        raise RuntimeError(f"GPU {index} has active compute processes; no new task started")
    return uuid


def compare_losses(first, second, epochs=CHECK_EPOCHS):
    """逐字段比较前五轮全部 loss；缺轮、缺列、非有限值或任何数值差异均失败。"""
    with Path(first).open() as handle:
        original = list(csv.DictReader(handle))
    with Path(second).open() as handle:
        repeated = list(csv.DictReader(handle))
    if len(original) < epochs or len(repeated) != epochs:
        raise ValueError("Insufficient original epochs or incorrect check length")
    fields = [key for key in original[0] if key.endswith("_loss")]
    if not fields or set(fields) != {key for key in repeated[0] if key.endswith("_loss")}:
        raise ValueError("Missing or inconsistent loss fields")
    checks = {}
    for key in fields:
        pairs = [(float(a[key]), float(b[key])) for a, b in zip(original[:epochs], repeated)]
        checks[key] = all(math.isfinite(a) and math.isfinite(b) and a == b for a, b in pairs)
    epoch_order = all(int(row["epoch"]) == i for i, row in enumerate(repeated, 1))
    return dict(epochs=epochs, fields=checks, epoch_order=epoch_order,
                passed=epoch_order and all(checks.values()))


def summarize(results, repeats):
    """分别报告完整运行的精度门槛和前五轮 loss 一致性，不扩大结论范围。"""
    def accuracy(dataset, task):
        """读取保存的最终测试准确率，不重新访问测试集。"""
        return results[f"{dataset}_{task}"]["test"]["accuracy"]
    cifar_bp, cifar_pld = accuracy("cifar10", "bp"), accuracy("cifar10", "pld")
    gates = {
        "cifar10_teacher_gt_79": accuracy("cifar10", "teacher") > .79,
        "fmnist_teacher_gt_90": accuracy("fmnist", "teacher") > .90,
        "cifar10_bp_between_43_47": .43 < cifar_bp < .47,
        "cifar10_pld_gt_47_and_bp": cifar_pld > .47 and cifar_pld > cifar_bp,
        "fmnist_bp_gt_88": accuracy("fmnist", "bp") > .88,
        "fmnist_pld_gt_88": accuracy("fmnist", "pld") > .88}
    return dict(gates=gates, thresholds_passed=all(gates.values()), repeats=repeats,
                repeatability_passed=len(repeats) == 6 and all(v["passed"] for v in repeats.values()),
                repeatability_scope="identical losses during the first five epochs",
                results=results)


def main():
    """冻结源码和配置；可读取已有完整结果，只执行六项前五轮检查。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--checks-only", action="store_true", help="复用已有完整结果，只重复前五轮")
    parser.add_argument("--output", type=Path, default=Path("result/reproduction"))
    args = parser.parse_args()
    gpu_available(args.gpu)
    folder = project_path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    jobs = [f"{dataset}_{task}" for dataset in ("cifar10", "fmnist")
            for task in ("teacher", "bp", "pld")]
    paths = [ROOT / "main.py", *sorted((ROOT / "pld").glob("*.py")),
             Path(__file__).resolve(), *[ROOT / "configs" / (name + ".json") for name in jobs]]
    frozen = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
    write_json(folder / "freeze.json", {
        "files": frozen, "jobs": jobs, "check_epochs": CHECK_EPOCHS,
        "gpu": args.gpu, "pid": os.getpid(), "checks_only": args.checks_only,
        "selection": "max validation accuracy; strict improvement; no test selection",
        "seed_policy": "one preselected configuration per condition; no seed search"})
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
                       CUBLAS_WORKSPACE_CONFIG=":4096:8")
    results, repeats, completed = {}, {}, []
    try:
        for checking in (False, True):
            for name in jobs:
                for path, digest in frozen.items():
                    if file_sha256(ROOT / path) != digest:
                        raise RuntimeError(f"Frozen input changed: {path}")
                cfg_path = ROOT / "configs" / (name + ".json")
                cfg = json.loads(cfg_path.read_text())
                original = project_path(cfg["output_dir"])
                if not checking and args.checks_only:
                    if json.loads((original / "config.json").read_text()) != cfg:
                        raise RuntimeError(f"Saved configuration differs: {name}")
                    results[name] = json.loads((original / "result.json").read_text())
                    if results[name]["status"] != "COMPLETED":
                        raise RuntimeError(f"Original run is incomplete: {name}")
                    completed.append(name)
                    continue
                gpu_available(args.gpu)
                environment["PYTHONHASHSEED"] = str(cfg["seed"])
                key = name + ("_repeat" if checking else "")
                output = folder / key if checking else original
                command = [sys.executable, "main.py", "--config", str(cfg_path.relative_to(ROOT)),
                           "--output", str(output)]
                if checking:
                    command += ["--check-epochs", str(CHECK_EPOCHS)]
                with (folder / (key + ".log")).open("w") as log:
                    child = subprocess.Popen(command, cwd=ROOT, env=environment,
                                             stdout=log, stderr=subprocess.STDOUT)
                    write_json(folder / "status.json", dict(
                        status="RUNNING", job=key, pid=os.getpid(), child_pid=child.pid,
                        command=command, completed=completed))
                    code = child.wait()
                if code:
                    raise RuntimeError(f"{key} exited with code {code}")
                if checking:
                    check = json.loads((output / "check_result.json").read_text())
                    if check["epochs_ran"] != CHECK_EPOCHS or check["test_evaluations"] != 0:
                        raise RuntimeError(f"Incorrect check budget or test access: {key}")
                    repeats[name] = compare_losses(original / "history.csv", output / "history.csv")
                    if not repeats[name]["passed"]:
                        raise RuntimeError(f"First-five-epoch loss mismatch: {name}")
                else:
                    results[name] = json.loads((output / "result.json").read_text())
                completed.append(key)
                write_json(folder / "partial_results.json", dict(results=results, repeats=repeats))
                print(key, repeats[name] if checking else results[name]["test"], flush=True)
        summary = summarize(results, repeats)
        write_json(folder / "summary.json", summary)
        write_json(folder / "status.json", dict(
            status="COMPLETED", thresholds_passed=summary["thresholds_passed"],
            repeatability_passed=summary["repeatability_passed"], completed=completed))
    except BaseException as error:
        write_json(folder / "status.json", dict(status="FAILED", error=repr(error), completed=completed))
        raise


if __name__ == "__main__":
    main()
