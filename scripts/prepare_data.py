"""下载或检查数据集：python scripts/prepare_data.py cifar10 fmnist mnist。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from torchvision import datasets
from pld.utils import project_path


def prepare_datasets(names, root, download=True):
    """准备官方训练集和测试集；只检查完整性和样本数，不训练或评估模型。"""
    classes = {"cifar10": datasets.CIFAR10, "fmnist": datasets.FashionMNIST,
               "mnist": datasets.MNIST}
    root = project_path(root)
    counts = {}
    for name in names:
        cls = classes[name]
        counts[name] = {
            "train": len(cls(root=str(root), train=True, download=download)),
            "test": len(cls(root=str(root), train=False, download=download))}
        print(f"{name}: {counts[name]} ({root})", flush=True)
    return counts


def main():
    """解析数据集和目录；离线模式仅验证现有数据，不发起下载。"""
    parser = argparse.ArgumentParser(description="Download or verify classification datasets")
    parser.add_argument("datasets", nargs="+", choices=("cifar10", "fmnist", "mnist"))
    parser.add_argument("--root", type=Path, default=Path("../dataset"))
    parser.add_argument("--verify-only", action="store_true", help="仅检查已有数据，不下载")
    args = parser.parse_args()
    prepare_datasets(args.datasets, args.root, download=not args.verify_only)


if __name__ == "__main__":
    main()
