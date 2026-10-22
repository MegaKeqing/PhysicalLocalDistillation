"""三类图像数据集的固定划分和逐样本预处理。"""
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def normalize_sample(value):
    """按单张图像最大值归一化；全零图像保持全零。"""
    maximum = value.max()
    return value / maximum if maximum > 0 else value


def seed_worker(worker_id):
    """从 DataLoader 的独立生成器派生 Python 与 NumPy 子进程种子。"""
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def build_transform(size, augmentation):
    """保留灰度、缩放、转张量、增强、逐样本归一化的执行顺序。"""
    steps = [transforms.Grayscale(1), transforms.Resize((size, size)),
             transforms.ToTensor()]
    choices = {
        "horizontal_flip": transforms.RandomHorizontalFlip(),
        "vertical_flip": transforms.RandomVerticalFlip(),
        "crop": transforms.RandomCrop(size, padding=max(1, size // 8))}
    for name in augmentation:
        if name not in choices:
            raise ValueError(f"Unknown augmentation: {name}")
        steps.append(choices[name])
    return transforms.Compose([*steps, transforms.Lambda(normalize_sample)])


def build_loaders(config):
    """训练集末尾划验证集，测试集保持官方划分；各加载器随机流独立。"""
    classes = {"mnist": datasets.MNIST, "fmnist": datasets.FashionMNIST,
               "cifar10": datasets.CIFAR10}
    cls = classes[config["dataset"]]
    common = dict(root=config["root"], download=config.get("download", False))
    train_transform = build_transform(config["size"], config["augmentation"])
    eval_transform = build_transform(config["size"], [])
    full_train = cls(train=True, transform=train_transform, **common)
    full_eval = cls(train=True, transform=eval_transform, **common)
    test = cls(train=False, transform=eval_transform, **common)
    split = len(full_train) - config["valid_size"]
    if not 0 < split < len(full_train):
        raise ValueError("valid_size must leave nonempty train and validation sets")
    train_count = config.get("train_size") or split
    if not 0 < train_count <= split:
        raise ValueError("train_size exceeds the training split")
    partitions = [Subset(full_train, range(train_count)),
                  Subset(full_eval, range(train_count)),
                  Subset(full_eval, range(split, len(full_eval))), test]
    loaders = []
    for index, dataset in enumerate(partitions):
        loaders.append(DataLoader(
            dataset, batch_size=config["batch_size"], shuffle=index == 0,
            num_workers=config["num_workers"], pin_memory=config["pin_memory"],
            drop_last=False, worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(config["shuffle_seed"] + index)))
    return dict(zip(("train", "train_eval", "valid", "test"), loaders))
