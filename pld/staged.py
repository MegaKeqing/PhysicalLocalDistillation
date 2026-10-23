"""固定散斑的逐层蒸馏与教师注入蒸馏；训练循环仅更新数字层。"""
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from .losses import FeatureLoss
from .models import DPNN
from .trainers import evaluate, make_optimizer
from .utils import file_sha256, seed_everything, state_sha256, write_json


class CachedDistillation:
    """复用缓存、逐阶段优化和选模；输入来源由具体训练器确定。"""

    def __init__(self, model, teacher, training, options, size, device):
        """冻结教师并构造局部损失；不执行学生光学前向来探测维度。"""
        if not isinstance(model, DPNN):
            raise ValueError("Fixed-speckle training requires a DPNN student")
        self.model, self.teacher = model.to(device), teacher.to(device)
        self.device, self.training = device, training
        self.hidden = len(model.blocks) - 1
        self.teacher.requires_grad_(False).eval()
        self.mapping = options["mapping"]
        if len(self.mapping) != self.hidden + 1 or len(options["losses"]) != self.hidden + 1:
            raise ValueError("One mapping and loss are required per student block")
        if any(type(i) is not int or not 0 <= i < len(teacher.blocks) for i in self.mapping):
            raise ValueError("Teacher block index out of range")
        if self.mapping[-1] != len(teacher.blocks) - 1:
            raise ValueError("The classifier must target teacher logits")
        with torch.no_grad():
            targets = teacher.features(torch.zeros(2, 1, size, size, device=device))
        shapes = [(1, size, size)] * self.hidden + [(model.blocks[-1][-1].out_features,)]
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(options["projection_seed"])
            self.losses = nn.ModuleList([
                FeatureLoss(spec, shape, tuple(targets[index].shape[1:]))
                for spec, shape, index in zip(options["losses"], shapes, self.mapping)
            ]).to(device)
        self.teacher_hash = state_sha256(teacher.state_dict())
        self.cache_records, self.measurements = {}, {}

    @torch.no_grad()
    def collect(self, loader, with_teacher):
        """按固定顺序收集样本和目标，拒绝漏尾批或随机采集顺序。"""
        if loader.drop_last or not isinstance(loader.sampler, SequentialSampler):
            raise ValueError("Acquisition requires a sequential loader with drop_last=False")
        count, offset, store = len(loader.dataset), 0, {}
        if count == 0:
            raise ValueError("Empty acquisition split")
        self.teacher.eval()
        for images, labels in loader:
            batch = {"images": images, "labels": labels}
            if with_teacher:
                features = self.teacher.features(images.to(self.device))
                batch.update({f"target_{i}": features[index] for i, index in enumerate(self.mapping)})
            for key, value in batch.items():
                if key not in store:
                    store[key] = torch.empty((count, *value.shape[1:]), dtype=value.dtype)
                store[key][offset:offset + len(images)] = value.detach().cpu()
            offset += len(images)
        if offset != count:
            raise ValueError("Acquisition did not cover every sample")
        return store

    @torch.no_grad()
    def transform(self, function, values):
        """按固定批大小转换 CPU 缓存，输出不携带计算图。"""
        result = None
        for start in range(0, len(values), self.batch_size):
            value = function(values[start:start + self.batch_size].to(self.device))
            if result is None:
                result = torch.empty((len(values), *value.shape[1:]), dtype=value.dtype)
            result[start:start + len(value)] = value.detach().cpu()
        return result

    def record_cache(self, name, **tensors):
        """保存本次采集及摘要；缓存属于运行产物，不是源码或跨运行共享状态。"""
        path = self.output / "cache" / (name + ".pt")
        torch.save(tensors, path)
        self.cache_records[name] = {
            "file": str(path.relative_to(self.output)), "sha256": file_sha256(path),
            "tensors": {key: list(value.shape) for key, value in tensors.items()}}
        write_json(self.output / "cache_manifest.json", self.cache_records)

    def measure(self, stage, split, values):
        """每阶段每个 split 仅采集一次光学强度，记录实际采集样本数。"""
        key = f"block_{stage}/{split}"
        if key in self.measurements:
            raise RuntimeError("Repeated optical acquisition: " + key)
        self.model.eval()
        intensity = self.transform(self.model.blocks[stage].optical, values)
        self.measurements[key] = len(values)
        self.record_cache(f"block_{stage}_{split}_speckles", intensity=intensity)
        return intensity

    def stage_output(self, stage, inputs):
        """隐藏阶段只走数字层；分类头直接消费上一阶段的固定特征。"""
        block = self.model.blocks[stage]
        return block.digital(inputs) if stage < self.hidden else block(inputs)

    def stage_loader(self, inputs, targets, labels, shuffle, seed):
        """把输入、目标、标签绑定在同一数据集，打乱时保持逐样本对应。"""
        if len(inputs) != len(targets) or len(inputs) != len(labels):
            raise ValueError("Cached input, target and label counts differ")
        return DataLoader(TensorDataset(inputs, targets, labels), batch_size=self.batch_size,
                          shuffle=shuffle, drop_last=False, num_workers=0,
                          generator=torch.Generator().manual_seed(seed))

    def stage_epoch(self, stage, loader, optimizer=None):
        """执行一个数字训练或验证轮次；隐藏层不计算无定义的分类准确率。"""
        training = optimizer is not None
        self.model.eval()
        self.model.blocks[stage].train(training)
        self.losses[stage].train(training)
        total, count, ce, correct = 0.0, 0, 0.0, 0
        with torch.set_grad_enabled(training):
            for inputs, target, labels in loader:
                inputs, target, labels = (value.to(self.device) for value in (inputs, target, labels))
                if training:
                    optimizer.zero_grad(set_to_none=True)
                output = self.stage_output(stage, inputs)
                loss = self.losses[stage](output, target, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite stage loss")
                if training and loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                total += loss.item() * len(labels)
                count += len(labels)
                if stage == self.hidden:
                    ce += F.cross_entropy(output.detach(), labels, reduction="sum").item()
                    correct += (output.detach().argmax(1) == labels).sum().item()
        if count == 0:
            raise ValueError("Empty stage loader")
        metrics = {"distill_loss": total / count, "count": count}
        if stage == self.hidden:
            metrics.update(loss=ce / count, accuracy=correct / count)
        return metrics

    def train_stage(self, stage, inputs, data, config):
        """当前块独立早停，恢复验证最优的数字层及 PMSE 投影，再冻结该阶段。"""
        seed_everything((config["seed"] + stage) % 2**32, config.get("threads", 4))
        self.model.requires_grad_(False)
        self.losses.requires_grad_(False)
        block, loss = self.model.blocks[stage], self.losses[stage]
        block.requires_grad_(True)
        loss.requires_grad_(True)
        optimizer = make_optimizer([*block.parameters(), *loss.parameters()], self.training)
        shuffle_seed = (config["data"]["shuffle_seed"] + stage) % 2**32
        train = self.stage_loader(inputs["train"], data["train"][f"target_{stage}"],
                                  data["train"]["labels"], True, shuffle_seed)
        valid = self.stage_loader(inputs["valid"], data["valid"][f"target_{stage}"],
                                  data["valid"]["labels"], False, shuffle_seed)
        head = stage == self.hidden
        monitor = "accuracy" if head else "distill_loss"
        best, best_epoch, best_valid = (-float("inf") if head else float("inf")), 0, None
        path = self.output / f"block_{stage}_best.pt"
        early = self.training.get("early_stopping")
        for epoch in range(1, self.training["epochs"] + 1):
            start = time.monotonic()
            train_metrics = self.stage_epoch(stage, train, optimizer)
            valid_metrics = self.stage_epoch(stage, valid)
            value = valid_metrics[monitor]
            better = value > best if head else value < best
            if better:
                best, best_epoch, best_valid = value, epoch, valid_metrics
                torch.save({"block": block.state_dict(), "loss": loss.state_dict(),
                            "optimizer": optimizer.state_dict(), "stage": stage,
                            "epoch": epoch, "valid": valid_metrics}, path)
            row = dict(stage=stage, epoch=epoch, train_distill_loss=train_metrics["distill_loss"],
                       valid_distill_loss=valid_metrics["distill_loss"],
                       train_loss=train_metrics.get("loss"), train_accuracy=train_metrics.get("accuracy"),
                       valid_loss=valid_metrics.get("loss"), valid_accuracy=valid_metrics.get("accuracy"),
                       seconds=time.monotonic() - start)
            with (self.output / "history.csv").open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if stage == 0 and epoch == 1:
                    writer.writeheader()
                writer.writerow(row)
            write_json(self.output / "status.json", dict(status="RUNNING", stage=stage, epoch=epoch,
                       best_epoch=best_epoch, monitor=monitor, best_value=best, last=row))
            print(f"stage={stage} epoch={epoch} train_local={row['train_distill_loss']:.6f} "
                  f"valid_local={row['valid_distill_loss']:.6f} "
                  f"valid_acc={row['valid_accuracy']} seconds={row['seconds']:.1f}", flush=True)
            if early and epoch >= early["min_epochs"] and epoch - best_epoch >= early["patience"]:
                break
        saved = torch.load(path, map_location=self.device, weights_only=True)
        block.load_state_dict(saved["block"])
        loss.load_state_dict(saved["loss"])
        block.requires_grad_(False).eval()
        loss.requires_grad_(False).eval()
        return dict(stage=stage, epochs_ran=epoch, best_epoch=best_epoch,
                    monitor=monitor, valid=best_valid, checkpoint=path.name)

    def fit(self, loaders, config, output):
        """采集固定输入，逐阶段训练并只在所有阶段选定后评估测试集一次。"""
        if config["data"]["augmentation"]:
            raise ValueError("Fixed-speckle caches require augmentation=[]")
        self.output, self.batch_size = Path(output), config["data"]["batch_size"]
        (self.output / "cache").mkdir()
        started = time.monotonic()
        initial_hash = state_sha256(self.model.state_dict())
        data = {split: self.collect(loaders["train_eval" if split == "train" else split], True)
                for split in ("train", "valid")}
        for split, values in data.items():
            self.record_cache(f"teacher_{split}", **{k: v for k, v in values.items() if k != "images"})
        inputs = self.prepare_inputs(data, loaders)
        stages = []
        for stage in range(self.hidden):
            intensity = self.hidden_inputs(stage, inputs, data)
            stages.append(self.train_stage(stage, intensity, data, config))
            inputs = self.finish_hidden(stage, intensity, inputs)
        stages.append(self.train_stage(self.hidden, inputs, data, config))
        torch.save({"format_version": 1, "model": self.model.state_dict(),
                    "losses": self.losses.state_dict(), "config": config, "stages": stages},
                   self.output / "best.pt")
        if state_sha256(self.teacher.state_dict()) != self.teacher_hash:
            raise RuntimeError("Frozen teacher changed")
        # 所有阶段权重在这里已经固定，测试指标不参与任何选择。
        test = self.test_student(inputs, loaders)
        result = dict(status="COMPLETED", task=config["task"], stages=stages,
                      epochs_ran=sum(stage["epochs_ran"] for stage in stages),
                      valid=stages[-1]["valid"], valid_mode=self.validation_mode,
                      test=test, test_mode="student_chain", test_evaluations=1,
                      initial_model_sha256=initial_hash,
                      best_model_sha256=state_sha256(self.model.state_dict()),
                      teacher_state_sha256=self.teacher_hash,
                      optical_measurements=self.measurements,
                      elapsed_seconds=time.monotonic() - started)
        write_json(self.output / "result.json", result)
        write_json(self.output / "status.json", result)
        print(f"Completed: task={config['task']} test_accuracy={test['accuracy']:.6f}", flush=True)
        return result


class PLDSequenceTrainer(CachedDistillation):
    """逐层训练：当前数字层收敛后，才用其输出采集下一层散斑。"""

    validation_mode = "student_chain"

    def __init__(self, model, teacher, training, options, size, device):
        """要求隐藏层 PMSE，允许教师和学生的特征维度不同。"""
        if any(spec["type"] != "pmse" for spec in options["losses"][:-1]):
            raise ValueError("PLD sequence hidden blocks require PMSE")
        if options["losses"][-1]["type"] != "kl_ce":
            raise ValueError("The classifier requires KL+CE")
        super().__init__(model, teacher, training, options, size, device)

    def prepare_inputs(self, data, loaders):
        """收集测试图像用于逐层测量，但不计算教师测试特征或测试指标。"""
        test = self.collect(loaders["test"], False)
        self.test_labels = test["labels"]
        return {"train": data["train"]["images"], "valid": data["valid"]["images"],
                "test": test["images"]}

    def hidden_inputs(self, stage, inputs, data):
        """对上一阶段冻结的学生输出，分别采集三种划分的散斑。"""
        return {split: self.measure(stage, split, values) for split, values in inputs.items()}

    def finish_hidden(self, stage, intensity, previous):
        """恢复最优数字层后生成并保存 z_k，再作为下一层的固定输入。"""
        outputs = {split: self.transform(self.model.blocks[stage].digital, values)
                   for split, values in intensity.items()}
        for split, values in outputs.items():
            self.record_cache(f"block_{stage}_{split}_features", features=values)
        return outputs

    def test_student(self, inputs, loaders):
        """测试已缓存的完整学生链末端特征，避免重复采集任何光学块。"""
        loader = DataLoader(TensorDataset(inputs["test"], self.test_labels),
                            batch_size=self.batch_size, shuffle=False)
        return evaluate(self.model.blocks[-1], loader, self.device)


class TeacherInjectionTrainer(CachedDistillation):
    """教师注入：隐藏层输入预先确定，阶段验证也采用教师输入。"""

    validation_mode = "teacher_injection"

    def __init__(self, model, teacher, training, options, size, device):
        """要求同构且介质不同的 DPNN、逐层一一对应和隐藏层直接 MSE。"""
        if not isinstance(model, DPNN) or not isinstance(teacher, DPNN):
            raise ValueError("Teacher injection requires DPNN teacher and student")
        if len(model.blocks) != len(teacher.blocks) or len(model.blocks) < 3:
            raise ValueError("Teacher injection requires matching DPNNs with at least two optical blocks")
        if options["mapping"] != list(range(len(model.blocks))):
            raise ValueError("Isomorphic teacher injection requires identity layer mapping")
        for student, target in zip(model.blocks[:-1], teacher.blocks[:-1]):
            signature = lambda b: (b.size, b.alpha, b.norm.eps, b.bias is None, b.dropout.p)
            if signature(student) != signature(target) or torch.equal(student.tm, target.tm):
                raise ValueError("Teacher/student must be isomorphic and use different scattering matrices")
        if any(spec["type"] != "mse" for spec in options["losses"][:-1]):
            raise ValueError("Teacher injection hidden blocks require MSE without a projection")
        if options["losses"][-1]["type"] != "kl_ce":
            raise ValueError("The classifier requires KL+CE")
        super().__init__(model, teacher, training, options, size, device)

    def prepare_inputs(self, data, loaders):
        """训练开始前一次采集各层 train/valid 散斑，测试集不进入注入缓存。"""
        self.intensities = {}
        for stage in range(self.hidden):
            self.intensities[stage] = {
                split: self.measure(stage, split, values["images"] if stage == 0
                                    else values[f"target_{stage - 1}"])
                for split, values in data.items()}
        return {}

    def hidden_inputs(self, stage, inputs, data):
        """直接读取该层预采集的教师注入散斑，不依赖已训练的学生上游块。"""
        return self.intensities[stage]

    def finish_hidden(self, stage, intensity, previous):
        """最后隐藏层输出 S_{L-1}(z^t_{L-2})，用于分类头训练和验证。"""
        if stage != self.hidden - 1:
            return previous
        outputs = {split: self.transform(self.model.blocks[stage].digital, values)
                   for split, values in intensity.items()}
        for split, values in outputs.items():
            self.record_cache(f"head_{split}_features", features=values)
        return outputs

    def test_student(self, inputs, loaders):
        """测试时移除教师注入，原始图像通过完整学生链；每块各测量一次。"""
        metrics = evaluate(self.model, loaders["test"], self.device)
        for stage in range(self.hidden):
            self.measurements[f"block_{stage}/test"] = metrics["count"]
        return metrics
