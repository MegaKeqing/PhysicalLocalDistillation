"""逐层及注入训练检查：合成小样本，无真实数据集访问。"""
import argparse
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from pld.models import build_model
from pld.staged import PLDSequenceTrainer, TeacherInjectionTrainer
from pld.utils import seed_everything, state_sha256, validate_config

DEVICE = torch.device("cpu")


def configuration(task):
    """返回固定种子的小型 DPNN 阶段训练配置。"""
    return {
        "task": task, "seed": 75001, "device": str(DEVICE), "threads": 2,
        "output_dir": "unused",
        "data": dict(dataset="cifar10", root="unused", size=8, batch_size=4, valid_size=6,
                     num_workers=0, pin_memory=False, augmentation=[], shuffle_seed=75002),
        "model": dict(type="dpnn", size=8, num_blocks=2, init_seed=75003, tm_seed=75004),
        "training": dict(epochs=5, learning_rate=.001, weight_decay=.01, early_stopping=None),
        "pld": dict(mapping=[0, 1, 2], projection_seed=75005, teacher_checkpoint="unused",
                    losses=[dict(type="pmse" if task=="pld-seq" else "mse"),
                            dict(type="pmse" if task=="pld-seq" else "mse"),
                            dict(type="kl_ce", weight=.5, label_weight=.5, temperature=3.)])}


def make_trainer(config):
    """构造散射介质不同、结构相同的 DPNN 教师和学生。"""
    teacher_config = dict(config["model"], init_seed=75006, tm_seed=75007)
    factory = PLDSequenceTrainer if config["task"] == "pld-seq" else TeacherInjectionTrainer
    return factory(build_model(config["model"]), build_model(teacher_config),
                   config["training"], config["pld"], 8, DEVICE)


def make_loaders():
    """构造带尾批、固定顺序和可区分样本的三种划分。"""
    generator = torch.Generator().manual_seed(75008)
    loaders = {}
    for split, count in (("train_eval", 10), ("valid", 6), ("test", 6)):
        images = torch.rand(count, 1, 8, 8, generator=generator)
        labels = torch.arange(count) % 10
        loaders[split] = DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)
    return loaders


class StagedChecks(unittest.TestCase):
    """验证测量次数、注入语义、冻结与前五轮小样本确定性。"""

    def setUp(self):
        """重置随机流，使测试与执行顺序无关。"""
        seed_everything(75001, 2)

    def test_sequence_measure_once_and_restore_all_best(self):
        """每个学生块每个样本只测一次，阶段更新和验证中禁止光学前向。"""
        config = configuration("pld-seq")
        trainer, loaders = make_trainer(config), make_loaders()
        calls = [[], []]
        originals = [b.optical for b in trainer.model.blocks[:-1]]
        def observed(index):
            """包装光学调用并记录采集样本数。"""
            def optical(value):
                """记录一次实际光学采集。"""
                calls[index].append(len(value))
                return originals[index](value)
            return optical
        with patch.object(trainer.model.blocks[0], "optical", observed(0)), \
             patch.object(trainer.model.blocks[1], "optical", observed(1)):
            original_epoch = trainer.stage_epoch
            def checked_epoch(*args, **kwargs):
                """任何训练或验证 epoch 都不得新增测量。"""
                before = [list(c) for c in calls]
                result = original_epoch(*args, **kwargs)
                self.assertEqual(before, calls)
                return result
            with patch.object(trainer, "stage_epoch", checked_epoch), tempfile.TemporaryDirectory() as directory:
                result = trainer.fit(loaders, config, directory)
                self.assertEqual([sum(c) for c in calls], [22, 22])
                self.assertEqual(result["test_evaluations"], 1)
                for stage, block in enumerate(trainer.model.blocks):
                    saved = torch.load(Path(directory)/f"block_{stage}_best.pt", weights_only=True)
                    self.assertEqual(state_sha256(saved["block"]), state_sha256(block.state_dict()))
                    self.assertEqual(state_sha256(saved["loss"]), state_sha256(trainer.losses[stage].state_dict()))
                final = torch.load(Path(directory)/"cache/block_1_test_features.pt", weights_only=True)["features"]
                images = loaders["test"].dataset.tensors[0]
                with torch.no_grad():
                    expected = trainer.transform(lambda x: trainer.model.features(x)[1], images)
                torch.testing.assert_close(final, expected, rtol=0, atol=0)
                self.assertEqual(state_sha256(trainer.teacher.state_dict()), trainer.teacher_hash)
                self.assertTrue(all(not p.requires_grad for p in trainer.model.parameters()))

    def test_injection_inputs_validation_and_final_chain(self):
        """隐藏层和验证使用教师输入，头用 S1(t0)，测试不允许访问教师。"""
        config = configuration("ti-pld")
        trainer, loaders = make_trainer(config), make_loaders()
        with tempfile.TemporaryDirectory() as directory:
            original_test = trainer.test_student
            def checked_test(inputs, all_loaders):
                """最终测试只允许学生前向，教师前向一旦调用即失败。"""
                with patch.object(trainer.teacher, "forward", side_effect=AssertionError("teacher test")), \
                     patch.object(trainer.teacher, "features", side_effect=AssertionError("teacher test")):
                    return original_test(inputs, all_loaders)
            with patch.object(trainer, "test_student", checked_test):
                result = trainer.fit(loaders, config, directory)
            self.assertEqual(result["valid_mode"], "teacher_injection")
            self.assertEqual(result["test_mode"], "student_chain")
            cache = Path(directory)/"cache"
            self.assertFalse(any("test" in p.name for p in cache.iterdir()))
            for split in ("train", "valid"):
                images = loaders["train_eval" if split == "train" else "valid"].dataset.tensors[0]
                with torch.no_grad():
                    teacher_z0 = trainer.transform(lambda x: trainer.teacher.features(x)[0], images)
                    expected = trainer.transform(trainer.model.blocks[1].optical, teacher_z0)
                    cached = torch.load(cache/f"block_1_{split}_speckles.pt", weights_only=True)["intensity"]
                    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
                    head = torch.load(cache/f"head_{split}_features.pt", weights_only=True)["features"]
                    torch.testing.assert_close(head, trainer.transform(trainer.model.blocks[1].digital, cached), rtol=0, atol=0)
                    chain = trainer.transform(lambda x: trainer.model.features(x)[1], images)
                    self.assertFalse(torch.equal(head, chain))
            self.assertEqual(result["optical_measurements"]["block_0/test"], 6)
            self.assertEqual(result["optical_measurements"]["block_1/test"], 6)

    def test_five_epoch_stage_repeatability(self):
        """相同配置独立训练两次，每阶段前五轮 loss 和最优参数逐位一致。"""
        for task in ("pld-seq", "ti-pld"):
            runs = []
            for repeat in range(2):
                config = configuration(task)
                seed_everything(config["seed"], 2)
                trainer = make_trainer(config)
                with tempfile.TemporaryDirectory() as directory:
                    result = trainer.fit(make_loaders(), config, directory)
                    with (Path(directory)/"history.csv").open() as handle:
                        history = [{k:v for k,v in row.items() if k != "seconds"}
                                   for row in csv.DictReader(handle)]
                self.assertEqual(len(history), 15)
                runs.append((history, result["best_model_sha256"]))
            self.assertEqual(runs[0], runs[1])

    def test_main_dispatch_with_generated_teacher(self):
        """真实 main 入口能加载教师并完成两种阶段训练，数据源用合成样本替代。"""
        import main as entry
        for task in ("pld-seq", "ti-pld"):
            config = configuration(task)
            config["training"]["epochs"] = 1
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                teacher_config = dict(config["model"], init_seed=75006, tm_seed=75007)
                teacher = build_model(teacher_config)
                checkpoint = path / "teacher.pt"
                torch.save({"format_version": 1, "model": teacher.state_dict(),
                            "config": {"model": teacher_config, "data": config["data"]}}, checkpoint)
                config["pld"]["teacher_checkpoint"] = str(checkpoint)
                config["output_dir"] = str(path / "run")
                source = path / "config.json"
                source.write_text(json.dumps(config))
                with patch("sys.argv", ["main.py", "--config", str(source)]), \
                     patch.object(entry, "build_loaders", return_value=make_loaders()):
                    entry.main()
                result = json.loads((path / "run/result.json").read_text())
                self.assertEqual(result["task"], task)
                self.assertEqual(result["test_evaluations"], 1)
                self.assertEqual(len(result["stages"]), 3)

    def test_invalid_protocols(self):
        """拒绝增强、错误损失、同散射介质及非逐层同构映射。"""
        config = configuration("ti-pld")
        for field in ("augmentation", "loss", "mapping"):
            bad = copy.deepcopy(config)
            if field == "augmentation":
                bad["data"]["augmentation"] = ["crop"]
                with self.assertRaises(ValueError):
                    validate_config(bad)
            elif field == "loss":
                bad["pld"]["losses"][0]["type"] = "pmse"
                with self.assertRaises(ValueError):
                    validate_config(bad)
            else:
                bad["pld"]["mapping"] = [1, 0, 2]
                with self.assertRaises(ValueError):
                    make_trainer(bad)
        with self.assertRaises(ValueError):
            TeacherInjectionTrainer(build_model(config["model"]), build_model(config["model"]),
                                    config["training"], config["pld"], 8, DEVICE)

    def test_cnn_sequence_teacher_and_tail_coverage(self):
        """CNN 可提供异形 PMSE 目标，拒绝漏尾批的固定缓存采集。"""
        config = configuration("pld-seq")
        config["pld"]["mapping"] = [0, 3, 4]
        teacher = build_model(dict(type="cnn", channels=[4, 8], init_seed=75009))
        trainer = PLDSequenceTrainer(build_model(config["model"]), teacher,
                                    config["training"], config["pld"], 8, DEVICE)
        loaders = make_loaders()
        with tempfile.TemporaryDirectory() as directory:
            trainer.fit(loaders, config, directory)
        bad = DataLoader(loaders["valid"].dataset, batch_size=4, drop_last=True)
        with self.assertRaises(ValueError):
            trainer.collect(bad, True)


def main():
    """指定 CPU 或 GPU 运行阶段训练器的独立检查。"""
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    unittest.main(argv=["check_staged"], verbosity=2)


if __name__ == "__main__":
    main()
