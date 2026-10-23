"""发布入口检查：不下载真实数据，不读取实验权重。"""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from PIL import Image
import main as entry
from pld.data import build_loaders
from pld.losses import FeatureLoss, linear_cka
from pld.utils import PROJECT_ROOT, provenance, validate_config
from scripts.prepare_data import prepare_datasets


class SyntheticImages:
    """以固定小图替代 torchvision 数据集，保留真实变换和加载器路径。"""

    def __init__(self, root, train=True, transform=None, download=False):
        """保存调用参数；本类不访问磁盘或网络。"""
        self.train, self.transform = train, transform

    def __len__(self):
        """返回足以区分训练和验证划分的小样本数。"""
        return 14 if self.train else 4

    def __getitem__(self, index):
        """生成非负图像及固定标签，由被测代码执行预处理。"""
        image = Image.new("RGB", (10, 10), color=(index + 1, 30, 70))
        return (self.transform(image) if self.transform else image), index % 10


class ReleaseChecks(unittest.TestCase):
    """覆盖可移植入口、无 Git、数据准备和损失选项约束。"""

    def test_missing_git_keeps_source_hashes(self):
        """无 Git 可执行文件时仍生成完整源码摘要和依赖记录。"""
        with patch("pld.utils.subprocess.run", side_effect=FileNotFoundError("git")):
            result = provenance(PROJECT_ROOT, torch.device("cpu"))
        self.assertIsNone(result["git_commit"])
        self.assertIsNone(result["git_status"])
        self.assertEqual(len(result["source_sha256"]["main.py"]), 64)
        self.assertIn("pillow", result)

    def test_source_archive_without_repository(self):
        """无 .git 的源码目录也可记录环境，Git 字段为空。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").write_text("# standalone source\n")
            result = provenance(root, torch.device("cpu"))
        self.assertIsNone(result["git_commit"])
        self.assertIsNone(result["git_status"])
        self.assertIn("main.py", result["source_sha256"])

    def test_all_example_configs_validate(self):
        """所有公开示例均保留可识别的任务、种子和数据配置。"""
        paths = list((PROJECT_ROOT / "configs").glob("*.json"))
        self.assertEqual(len(paths), 14)
        for path in paths:
            validate_config(json.loads(path.read_text()))

    def test_teacher_only_cka(self):
        """CKA 与教师对齐公式相同，拒绝不支持的标签和图像目标。"""
        x = torch.arange(24, dtype=torch.float32).reshape(4, 6).requires_grad_()
        y = torch.flip(x.detach(), dims=[0])
        loss = FeatureLoss({"type": "cka", "weight_feat": 0.7}, (6,), (6,))
        torch.testing.assert_close(loss(x, y), 0.7 * (1 - linear_cka(x, y)), rtol=0, atol=0)
        for field in ("weight_label", "weight_image"):
            with self.assertRaises(ValueError):
                FeatureLoss({"type": "cka", field: 1}, (6,), (6,))

    def test_prepare_three_datasets_and_offline_mode(self):
        """三个数据集均准备两种划分；离线检查不会请求下载。"""
        for download in (False, True):
            with patch("scripts.prepare_data.datasets.CIFAR10") as cifar, \
                 patch("scripts.prepare_data.datasets.FashionMNIST") as fashion, \
                 patch("scripts.prepare_data.datasets.MNIST") as mnist:
                for factory in (cifar, fashion, mnist):
                    factory.return_value.__len__.return_value = 4
                result = prepare_datasets(["cifar10", "fmnist", "mnist"], "../dataset", download)
                self.assertEqual(set(result), {"cifar10", "fmnist", "mnist"})
                for factory in (cifar, fashion, mnist):
                    self.assertEqual(factory.call_count, 2)
                    self.assertEqual([c.kwargs["train"] for c in factory.call_args_list], [True, False])
                    for call in factory.call_args_list:
                        self.assertEqual(call.kwargs["download"], download)
                        self.assertEqual(Path(call.kwargs["root"]), PROJECT_ROOT.parent / "dataset")

    def test_real_loaders_with_synthetic_three_datasets(self):
        """替换数据源后验证真实预处理、固定划分和独立打乱流的可重复性。"""
        with patch("pld.data.datasets.CIFAR10", SyntheticImages), \
             patch("pld.data.datasets.FashionMNIST", SyntheticImages), \
             patch("pld.data.datasets.MNIST", SyntheticImages):
            for name in ("cifar10", "fmnist", "mnist"):
                config = dict(dataset=name, root="unused", size=8, batch_size=3,
                              valid_size=4, num_workers=0, pin_memory=False,
                              augmentation=[], shuffle_seed=74001, download=False)
                first, second = build_loaders(config), build_loaders(config)
                for key, count in (("train", 10), ("train_eval", 10), ("valid", 4), ("test", 4)):
                    self.assertEqual(len(first[key].dataset), count)
                    for (images, labels), (other_images, other_labels) in zip(first[key], second[key]):
                        self.assertEqual(tuple(images.shape[1:]), (1, 8, 8))
                        self.assertTrue(torch.equal(images.amax(dim=(1, 2, 3)), torch.ones(len(images))))
                        self.assertTrue(torch.equal(images, other_images))
                        self.assertTrue(torch.equal(labels, other_labels))

    def test_main_outside_project_preserves_json_and_teacher_path(self):
        """从外部目录执行教师、BP、PLD 小样本检查，确认路径稳定且原 JSON 不变。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "project"
            (root / "configs").mkdir(parents=True)
            outside = Path(directory) / "outside"
            outside.mkdir()
            batch = [(torch.ones(4, 1, 8, 8), torch.arange(4))]
            base = {
                "task": "teacher", "seed": 74101, "device": "cpu", "threads": 2,
                "output_dir": "outputs/teacher",
                "data": dict(dataset="mnist", root="../dataset", size=8, batch_size=4,
                             valid_size=4, num_workers=0, pin_memory=False, augmentation=[],
                             shuffle_seed=74102, download=False),
                "model": dict(type="mlp", init_seed=74103, size=8, hidden_dims=[16, 16]),
                "training": dict(epochs=2, learning_rate=0.001, weight_decay=0.01)}
            original_cwd = Path.cwd()
            try:
                os.chdir(outside)
                with patch("pld.utils.PROJECT_ROOT", root):
                    for task in ("teacher", "bp", "pld"):
                        config = copy.deepcopy(base)
                        config["task"] = task
                        config["output_dir"] = "outputs/" + task
                        if task == "pld":
                            config["pld"] = dict(teacher_checkpoint="outputs/teacher/best.pt",
                                                mapping=[0, 1, 2], projection_seed=74104,
                                                losses=[{"type": "pmse"}, {"type": "pmse"},
                                                        {"type": "kl_ce"}])
                        path = root / "configs" / (task + ".json")
                        path.write_text(json.dumps(config))
                        args = ["main.py", "--config", "configs/" + task + ".json", "--check-epochs", "1"]
                        output = root / config["output_dir"]
                        if task == "bp":
                            args += ["--output", "overrides/bp"]
                            output = root / "overrides/bp"
                        with patch("sys.argv", args), patch.object(entry, "build_loaders") as loaders:
                            loaders.return_value = dict(train=batch, train_eval=batch, valid=batch)
                            entry.main()
                            self.assertEqual(loaders.call_args.args[0]["root"], str(root.parent / "dataset"))
                        self.assertEqual(json.loads((output / "config.json").read_text()), config)
                        result = json.loads((output / "check_result.json").read_text())
                        self.assertEqual(result["test_evaluations"], 0)
                        self.assertEqual(result["epochs_ran"], 1)
                        self.assertFalse((output / "result.json").exists())
                        if task == "pld":
                            manifest = json.loads((output / "manifest.json").read_text())
                            self.assertEqual(len(manifest["teacher_checkpoint_sha256"]), 64)
                        with patch("sys.argv", args), self.assertRaises(FileExistsError):
                            entry.main()
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
