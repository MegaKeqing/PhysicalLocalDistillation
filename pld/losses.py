"""局部蒸馏损失；PMSE 投影头与对应学生块一起优化。"""
import math
import warnings
import torch
from torch import nn
from torch.nn import functional as F


def linear_cka(first, second, eps=1e-8):
    """计算中心化样本 Gram 的余弦；根号内加 eps 保证退化输入梯度有限。"""
    x, y = first.flatten(1), second.flatten(1)
    x, y = x - x.mean(0, keepdim=True), y - y.mean(0, keepdim=True)
    gx, gy = x @ x.T, y @ y.T
    dot = torch.einsum("ij,ij->", gx, gy)
    nx = torch.einsum("ij,ij->", gx, gx)
    ny = torch.einsum("ij,ij->", gy, gy)
    return dot / torch.sqrt(nx * ny + eps)


def cosine_gram(features, eps=1e-8):
    """构造按批次样本数缩放的余弦核；零向量导致 trace 偏离 1 时发出提示。"""
    flat = features.flatten(1).float()
    unit = flat / (torch.norm(flat, p=2, dim=1, keepdim=True) + eps)
    gram = unit @ unit.T / len(unit)
    if abs(float(gram.trace().detach()) - 1.0) > 1e-3:
        warnings.warn("MI kernel trace differs from 1; check zero/near-zero features")
    return gram


def matrix_entropy(gram, alpha=None, trace_norm=False, eps=1e-8):
    """计算矩阵 Shannon 熵（nat）或 Rényi 熵（bit），仅联合核做迹归一化。"""
    if alpha is not None and (not math.isfinite(alpha) or alpha <= 0 or abs(alpha - 1) < 1e-3):
        raise ValueError("alpha must be positive and at least 1e-3 away from 1")
    matrix = (gram + gram.T) * 0.5
    if trace_norm:
        matrix = matrix / (matrix.trace() + eps)
    symmetric = 0.5 * (matrix + matrix.T)
    eye = torch.eye(len(matrix), device=matrix.device, dtype=matrix.dtype)
    # 对角线加入固定抖动以稳定特征值分解；分解失败直接报错。
    values = torch.linalg.eigvalsh(symmetric + 1e-8 * eye).clamp_min(0)
    if alpha is None:
        return -(values * torch.log(values + eps)).nansum()
    return torch.log2((values + eps).pow(alpha).nansum()) / (1 - alpha)


class FeatureLoss(nn.Module):
    """MSE、PMSE、CKA、MI 和分类 KL+CE 的统一逐块接口。"""

    def __init__(self, config, student_shape, teacher_shape):
        """校验损失类型及维度；PMSE 显式创建可训练投影矩阵。"""
        super().__init__()
        self.config = dict(config)
        self.kind = self.config.pop("type").lower()
        allowed = {
            "mse": {"weight"},
            "pmse": {"weight", "bias"},
            "cka": {"weight_feat", "eps"},
            "mi": {"weight", "alpha", "eps"},
            "kl_ce": {"weight", "label_weight", "temperature"}}
        if self.kind not in allowed:
            raise ValueError(f"Unknown loss: {self.kind}")
        unknown = set(self.config) - allowed[self.kind]
        if unknown:
            raise ValueError(f"Unknown {self.kind} loss options: {unknown}")
        self.weight = float(self.config.get("weight", 1.0))
        if self.kind == "mse" and student_shape != teacher_shape:
            raise ValueError("MSE requires equal feature shapes; use PMSE for unequal shapes")
        if self.kind == "kl_ce" and (student_shape != teacher_shape or len(student_shape) != 1):
            raise ValueError("KL+CE requires matching classification logits")
        if self.kind == "pmse":
            self.projection = nn.Linear(math.prod(student_shape), math.prod(teacher_shape),
                                        bias=self.config.get("bias", False))
        if self.kind == "kl_ce" and self.config.get("temperature", 3.0) <= 0:
            raise ValueError("temperature must be positive")

    def forward(self, student, teacher, labels=None):
        """教师目标始终 detach；所有损失均采用最小化约定。"""
        teacher = teacher.detach()
        if self.kind in {"mse", "pmse", "mi"} and self.weight == 0:
            return student.new_zeros(())
        if self.kind == "kl_ce" and self.weight == 0 and self.config.get("label_weight", 0.5) == 0:
            return student.new_zeros(())
        if self.kind == "mse":
            return self.weight * F.mse_loss(student, teacher)
        if self.kind == "pmse":
            return self.weight * F.mse_loss(self.projection(student.flatten(1)), teacher.flatten(1))
        if self.kind == "kl_ce":
            temperature = self.config.get("temperature", 3.0)
            loss = self.weight * temperature**2 * F.kl_div(
                F.log_softmax(student / temperature, dim=-1),
                F.softmax(teacher / temperature, dim=-1), reduction="batchmean")
            label_weight = self.config.get("label_weight", 0.5)
            if label_weight:
                if labels is None:
                    raise ValueError("labels are required for cross entropy")
                loss = loss + label_weight * F.cross_entropy(student, labels)
            return loss
        if self.kind == "cka":
            weight = self.config.get("weight_feat", 1.0)
            if weight == 0:
                return student.new_zeros(())
            return weight * (1 - linear_cka(student, teacher, self.config.get("eps", 1e-8)))
        eps, alpha = self.config.get("eps", 1e-8), self.config.get("alpha")
        gx, gy = cosine_gram(student, eps), cosine_gram(teacher, eps)
        information = (matrix_entropy(gx, alpha, eps=eps) + matrix_entropy(gy, alpha, eps=eps)
                       - matrix_entropy(gx * gy, alpha, trace_norm=True, eps=eps))
        return -self.weight * information
