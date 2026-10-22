"""BP 与逐块局部蒸馏；验证集选模，训练结束后测试一次。"""
import csv
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from .losses import FeatureLoss
from .utils import state_sha256, write_json


def make_optimizer(parameters, config):
    """构建 AdamW 及默认 betas、eps，不启用学习率调度。"""
    return torch.optim.AdamW(parameters, lr=config["learning_rate"],
                            weight_decay=config["weight_decay"])


@torch.no_grad()
def evaluate(model, loader, device):
    """按样本数统计交叉熵和准确率，验证及测试均不进行数据增强。"""
    model.eval()
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        correct += (logits.argmax(1) == labels).sum().item()
        count += len(labels)
    if count == 0:
        raise ValueError("Empty evaluation loader")
    metrics = {"loss": loss_sum / count, "accuracy": correct / count, "count": count}
    if not torch.isfinite(torch.tensor(metrics["loss"])):
        raise FloatingPointError("Non-finite evaluation loss")
    return metrics


class BPTrainer:
    """对 CNN、DPNN 或 MLP 执行完整链路反向传播。"""

    def __init__(self, model, config, device):
        """将模型放到指定设备并创建单个优化器。"""
        self.model, self.device = model.to(device), device
        self.optimizer = make_optimizer(model.parameters(), config)
        self.losses = nn.ModuleList()

    def train_batch(self, images, labels):
        """对一个批次执行交叉熵、反向传播和更新。"""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BP loss")
        loss.backward()
        self.optimizer.step()
        return [loss.item()]

    def optimizer_states(self):
        """返回优化器状态，随最佳 checkpoint 一同保存。"""
        return [self.optimizer.state_dict()]


class PLDTrainer:
    """冻结教师，各学生块仅接收自己的局部损失梯度。"""

    def __init__(self, model, teacher, config, pld_config, size, device):
        """验证层映射，以确定性虚拟输入推断投影维度，再逐块建优化器。"""
        self.model, self.teacher, self.device = model.to(device), teacher.to(device), device
        self.teacher.requires_grad_(False).eval()
        self.mapping = pld_config["mapping"]
        if len(self.mapping) != len(model.blocks) or len(pld_config["losses"]) != len(model.blocks):
            raise ValueError("One mapping and loss are required per student block")
        if any(type(i) is not int or not 0 <= i < len(teacher.blocks) for i in self.mapping):
            raise ValueError("Teacher block index out of range")
        model.eval()
        with torch.no_grad():
            dummy = torch.zeros(2, 1, size, size, device=device)
            students, targets = model.features(dummy), teacher.features(dummy)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(pld_config["projection_seed"])
            self.losses = nn.ModuleList([
                FeatureLoss(spec, tuple(student.shape[1:]), tuple(targets[index].shape[1:]))
                for spec, student, index in zip(pld_config["losses"], students, self.mapping)
            ]).to(device)
        self.optimizers = [
            make_optimizer([*block.parameters(), *loss.parameters()], config)
            for block, loss in zip(model.blocks, self.losses)]
        self.model.train()

    def train_batch(self, images, labels):
        """块内立即更新；下一块使用该次前向的 detach 输出，不跨块反传。"""
        self.model.train()
        self.losses.train()
        self.teacher.eval()
        with torch.no_grad():
            targets = self.teacher.features(images)
        value, metrics = images.detach(), []
        for block, loss_fn, optimizer, index in zip(
                self.model.blocks, self.losses, self.optimizers, self.mapping):
            optimizer.zero_grad(set_to_none=True)
            output = block(value)
            loss = loss_fn(output, targets[index], labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PLD loss")
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            value = output.detach()
            metrics.append(loss.item())
        return metrics

    def optimizer_states(self):
        """分别保存每个块及其投影头的优化器状态。"""
        return [optimizer.state_dict() for optimizer in self.optimizers]


def fit(trainer, loaders, config, output, check_epochs=None):
    """按验证准确率选模；检查模式仅截取前 N 轮，不改变训练步骤、不访问测试集。"""
    output = Path(output)
    training = config["training"]
    early = training.get("early_stopping")
    best_accuracy, best_epoch = -float("inf"), 0
    best_path = output / "best.pt"
    initial_hash = state_sha256(trainer.model.state_dict())
    teacher_hash = (state_sha256(trainer.teacher.state_dict())
                    if isinstance(trainer, PLDTrainer) else None)
    started = time.monotonic()
    epoch_limit = check_epochs if check_epochs is not None else training["epochs"]
    for epoch in range(1, epoch_limit + 1):
        epoch_start = time.monotonic()
        sums, count = None, 0
        for images, labels in loaders["train"]:
            images, labels = images.to(trainer.device), labels.to(trainer.device)
            losses = trainer.train_batch(images, labels)
            if sums is None:
                sums = [0.0] * len(losses)
            sums = [total + value * len(labels) for total, value in zip(sums, losses)]
            count += len(labels)
        if count == 0:
            raise ValueError("Empty training loader")
        valid = evaluate(trainer.model, loaders["valid"], trainer.device)
        train = evaluate(trainer.model, loaders["train_eval"], trainer.device)
        row = dict(epoch=epoch, optimization_loss=sum(sums) / count,
                   train_loss=train["loss"], train_accuracy=train["accuracy"],
                   valid_loss=valid["loss"], valid_accuracy=valid["accuracy"],
                   seconds=time.monotonic() - epoch_start)
        row.update({f"block_{i}_loss": total / count for i, total in enumerate(sums)})
        with (output / "history.csv").open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        if valid["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = valid["accuracy"], epoch
            torch.save({
                "format_version": 1, "model": trainer.model.state_dict(),
                "losses": trainer.losses.state_dict(),
                "optimizers": trainer.optimizer_states(), "epoch": epoch,
                "valid": valid, "config": config}, best_path)
        write_json(output / "status.json", {
            "status": "RUNNING", "epoch": epoch, "best_epoch": best_epoch,
            "best_valid_accuracy": best_accuracy, "last": row})
        print(f"epoch={epoch} train={train['accuracy']:.4f} valid={valid['accuracy']:.4f} "
              f"best={best_accuracy:.4f} seconds={row['seconds']:.1f}", flush=True)
        if early and epoch >= early["min_epochs"] and epoch - best_epoch >= early["patience"]:
            break
    if check_epochs is not None:
        if teacher_hash is not None and teacher_hash != state_sha256(trainer.teacher.state_dict()):
            raise RuntimeError("Frozen teacher changed")
        result = {
            "status": "CHECK_COMPLETED", "epochs_ran": epoch, "test_evaluations": 0,
            "initial_model_sha256": initial_hash, "teacher_state_sha256": teacher_hash,
            "elapsed_seconds": time.monotonic() - started}
        write_json(output / "check_result.json", result)
        write_json(output / "status.json", result)
        print(f"Check completed: {epoch} epochs; test set not evaluated", flush=True)
        return result
    checkpoint = torch.load(best_path, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(checkpoint["model"])
    trainer.losses.load_state_dict(checkpoint["losses"])
    test = evaluate(trainer.model, loaders["test"], trainer.device)
    if teacher_hash is not None and teacher_hash != state_sha256(trainer.teacher.state_dict()):
        raise RuntimeError("Frozen teacher changed")
    result = {
        "status": "COMPLETED", "epochs_ran": epoch, "best_epoch": best_epoch,
        "valid": checkpoint["valid"], "test": test, "test_evaluations": 1,
        "initial_model_sha256": initial_hash,
        "best_model_sha256": state_sha256(trainer.model.state_dict()),
        "best_losses_sha256": state_sha256(trainer.losses.state_dict()),
        "teacher_state_sha256": teacher_hash, "elapsed_seconds": time.monotonic() - started}
    write_json(output / "result.json", result)
    write_json(output / "status.json", result)
    print(f"Completed: test_accuracy={test['accuracy']:.6f}, best_epoch={best_epoch}", flush=True)
    return result
