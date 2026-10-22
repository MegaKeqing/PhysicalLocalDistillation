"""无需训练数据的核心检查：python -m tests.check_core --device cpu。"""
import argparse
import unittest
import csv
import tempfile
from pathlib import Path
import torch
from pld.losses import FeatureLoss, linear_cka
from pld.models import build_model
from pld.trainers import BPTrainer, PLDTrainer, fit
from scripts.reproduce import compare_losses
from pld.utils import seed_everything, state_sha256

DEVICE = torch.device("cpu")


def student_config(kind="dpnn"):
    """返回小型但保持完整传播链的测试模型配置。"""
    if kind == "dpnn":
        return dict(type=kind, init_seed=73001, tm_seed=73002, size=8, num_blocks=2)
    return dict(type=kind, init_seed=73001, size=8, hidden_dims=[64, 64])


def teacher_config():
    """返回与学生输入尺寸匹配的三块 MLP 教师。"""
    return dict(type="mlp", init_seed=73003, size=8, hidden_dims=[64, 64])


def pld_config(loss="pmse"):
    """构造三块映射及独立投影种子，用于逐块优化检查。"""
    return dict(mapping=[0, 1, 2], projection_seed=73004,
                losses=[dict(type=loss), dict(type=loss),
                        dict(type="kl_ce", weight=0.5, label_weight=0.5, temperature=3.0)])


class CoreChecks(unittest.TestCase):
    """检查可重复性、梯度路径、冻结教师、损失边界及配置错误。"""

    def setUp(self):
        """每项检查重置所有随机状态，避免检查顺序影响结论。"""
        seed_everything(73005, 2)
        self.images = torch.rand(8, 1, 8, 8, device=DEVICE)
        self.labels = torch.arange(8, device=DEVICE) % 10
        self.optimizer = dict(learning_rate=0.001, weight_decay=0.01)

    def test_deterministic_construction(self):
        """同一配置独立构造两次，应得到逐位相同的参数及介质。"""
        for kind in ("dpnn", "mlp"):
            self.assertEqual(state_sha256(build_model(student_config(kind)).state_dict()),
                             state_sha256(build_model(student_config(kind)).state_dict()))

    def test_batch_independent_normalization(self):
        """加入尺度不同的另一张图，不应改变既有样本的 DPNN 输出。"""
        model = build_model(student_config()).to(DEVICE).eval()
        first = model(self.images[:1])
        paired = model(torch.cat((self.images[:1], self.images[1:2] * 100)))[0:1]
        torch.testing.assert_close(first, paired, rtol=1e-5, atol=1e-5)

    def test_losses_and_teacher_detach(self):
        """五种损失在 CPU/GPU 上都产生有限学生梯度且不回传到教师。"""
        for kind in ("mse", "pmse", "cka", "mi", "kl_ce"):
            x = torch.rand(8, 10, device=DEVICE, requires_grad=True)
            y = torch.rand(8, 10, device=DEVICE, requires_grad=True)
            loss = FeatureLoss(dict(type=kind), (10,), (10,)).to(DEVICE)
            value = loss(x, y, self.labels)
            value.backward()
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(x.grad).all())
            self.assertIsNone(y.grad)

    def test_disabled_loss_has_no_gradient(self):
        """权重为零时不创建梯度，避免 AdamW 仅凭权重衰减更新禁用块。"""
        for kind in ("mse", "pmse", "mi", "kl_ce", "cka"):
            cfg = dict(type=kind)
            if kind == "cka":
                cfg.update(weight_feat=0)
            else:
                cfg["weight"] = 0
                if kind == "kl_ce":
                    cfg["label_weight"] = 0
            loss = FeatureLoss(cfg, (10,), (10,)).to(DEVICE)
            value = loss(torch.rand(8, 10, device=DEVICE, requires_grad=True),
                         torch.rand(8, 10, device=DEVICE), self.labels)
            self.assertFalse(value.requires_grad)
            self.assertEqual(value.item(), 0)

    def test_cka_degenerate(self):
        """常量特征不能使 CKA 梯度成为 NaN。"""
        x = torch.ones(8, 12, device=DEVICE, requires_grad=True)
        value = linear_cka(x, torch.ones_like(x))
        value.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertEqual(value.item(), 0)

    def test_bp_updates(self):
        """BP 应更新各隐藏块，而固定介质保持不变。"""
        model = build_model(student_config()).to(DEVICE)
        trainer = BPTrainer(model, self.optimizer, DEVICE)
        before = [block.weight.detach().clone() for block in model.blocks[:-1]]
        tm = model.blocks[0].tm.detach().clone()
        trainer.train_batch(self.images, self.labels)
        for old, block in zip(before, model.blocks[:-1]):
            self.assertFalse(torch.equal(old, block.weight))
        self.assertTrue(torch.equal(tm, model.blocks[0].tm))

    def test_pld_detaches_between_blocks(self):
        """通过块输入钩子确认后续块输入不携带上游计算图。"""
        trainer = PLDTrainer(build_model(student_config()), build_model(teacher_config()),
                             self.optimizer, pld_config(), 8, DEVICE)
        inputs = []
        handles = [b.register_forward_pre_hook(lambda module, args: inputs.append(args[0].requires_grad))
                   for b in trainer.model.blocks]
        frozen = state_sha256(trainer.teacher.state_dict())
        projections = [loss.projection.weight.detach().clone() for loss in trainer.losses[:-1]]
        trainer.train_batch(self.images, self.labels)
        for handle in handles:
            handle.remove()
        self.assertEqual(inputs, [False, False, False])
        self.assertEqual(frozen, state_sha256(trainer.teacher.state_dict()))
        self.assertTrue(all(p.grad is None for p in trainer.teacher.parameters()))
        for before, loss in zip(projections, trainer.losses):
            self.assertFalse(torch.equal(before, loss.projection.weight))
        parameter_sets = [{id(p) for group in opt.param_groups for p in group["params"]}
                          for opt in trainer.optimizers]
        self.assertFalse(any(parameter_sets[i] & parameter_sets[j]
                             for i in range(3) for j in range(i)))

    def test_repeated_updates(self):
        """两次独立构造并执行相同多步更新，模型和投影头必须逐位一致。"""
        for kind in ("dpnn", "mlp"):
            for method in ("bp", "pld"):
                hashes = []
                for repeat in range(2):
                    seed_everything(73005, 2)
                    model = build_model(student_config(kind))
                    trainer = (BPTrainer(model, self.optimizer, DEVICE) if method == "bp"
                               else PLDTrainer(model, build_model(teacher_config()),
                                               self.optimizer, pld_config(), 8, DEVICE))
                    for _ in range(3):
                        trainer.train_batch(self.images, self.labels)
                    hashes.append((state_sha256(model.state_dict()),
                                   state_sha256(trainer.losses.state_dict())))
                self.assertEqual(hashes[0], hashes[1])

    def test_invalid_mapping_and_mse(self):
        """拒绝错误层号以及会引发广播的 MSE 形状。"""
        bad = pld_config()
        bad["mapping"] = [0, 1, 9]
        with self.assertRaises(ValueError):
            PLDTrainer(build_model(student_config()), build_model(teacher_config()),
                       self.optimizer, bad, 8, DEVICE)
        with self.assertRaises(ValueError):
            FeatureLoss(dict(type="mse"), (1, 8, 8), (64,))

    def test_cnn_freeze_including_bn(self):
        """冻结 CNN 教师时，BatchNorm 统计和 Dropout 状态也必须冻结。"""
        teacher = build_model(dict(type="cnn", init_seed=73006, channels=[4, 8],
                                   dropout=0.2, sigmoid=True))
        config = pld_config()
        config["mapping"] = [0, 3, 4]
        trainer = PLDTrainer(build_model(student_config()), teacher, self.optimizer, config, 8, DEVICE)
        before = state_sha256(teacher.state_dict())
        trainer.train_batch(self.images, self.labels)
        self.assertEqual(before, state_sha256(teacher.state_dict()))
        self.assertFalse(teacher.training)


    def test_five_epoch_check_preserves_prefix_without_test(self):
        """短检查保持正常训练的前五轮 loss；没有测试 loader 时也应完整退出。"""
        training = dict(self.optimizer, epochs=6, early_stopping=None)
        batch = [(self.images, self.labels)]
        for method in ("bp", "pld"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for checking in (False, True):
                    seed_everything(73005, 2)
                    model = build_model(student_config())
                    trainer = (BPTrainer(model, training, DEVICE) if method == "bp"
                               else PLDTrainer(model, build_model(teacher_config()), training,
                                               pld_config(), 8, DEVICE))
                    loaders = dict(train=batch, valid=batch, train_eval=batch)
                    if not checking:
                        loaders["test"] = batch
                    output = root / ("check" if checking else "full")
                    output.mkdir()
                    result = fit(trainer, loaders, {"training": training}, output,
                                 check_epochs=5 if checking else None)
                    self.assertEqual(result["epochs_ran"], 5 if checking else 6)
                    self.assertEqual(result["test_evaluations"], 0 if checking else 1)
                    if checking:
                        self.assertFalse((output / "result.json").exists())
                first, second = root / "full/history.csv", root / "check/history.csv"
                self.assertTrue(compare_losses(first, second)["passed"])
                with second.open() as handle:
                    rows = list(csv.DictReader(handle))
                rows[0]["optimization_loss"] = str(float(rows[0]["optimization_loss"]) + 0.01)
                with second.open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                self.assertFalse(compare_losses(first, second)["passed"])


def main():
    """选择设备并运行可独立复用的核心单元检查。"""
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    DEVICE = torch.device(args.device)
    unittest.main(argv=["check_core"], verbosity=2)


if __name__ == "__main__":
    main()
