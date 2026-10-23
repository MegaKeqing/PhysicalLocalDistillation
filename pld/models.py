"""DPNN、定制 VGG 型 CNN 与逐块 MLP。"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class BlockNetwork(nn.Module):
    """按块顺序执行网络；最后一个块始终输出分类 logits。"""

    def forward(self, value):
        """执行完整前向；训练器负责决定块间是否截断梯度。"""
        for block in self.blocks:
            value = block(value)
        return value

    def features(self, value):
        """返回每个块的输出，供冻结教师提供逐层目标。"""
        outputs = []
        for block in self.blocks:
            value = block(value)
            outputs.append(value)
        return outputs


class OpticalBlock(nn.Module):
    """相位编码、固定散射、强度探测和可训练数字变换。"""

    def __init__(self, tm, size, alpha, eps, bias, dropout):
        """注册固定介质，并按方差为输入维度倒数的正态分布初始化隐藏数字层。"""
        super().__init__()
        dim = size * size
        self.size, self.alpha = size, alpha
        self.register_buffer("tm", tm)
        self.weight = nn.Parameter(torch.randn(dim, dim) / math.sqrt(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def optical(self, value):
        """仅计算固定光学段的散斑强度，供逐层训练一次采集后复用。"""
        if value.ndim != 4 or tuple(value.shape[1:]) != (1, self.size, self.size):
            raise ValueError("DPNN requires [batch, 1, size, size]")
        if not torch.isfinite(value).all() or (value < 0).any():
            raise ValueError("DPNN input must be finite and non-negative")
        maximum = value.amax(dim=(1, 2, 3), keepdim=True)
        normalized = value / torch.where(maximum > 0, maximum, torch.ones_like(maximum))
        field = torch.exp(1j * normalized * self.alpha * torch.pi)
        scattered = torch.einsum("bcm,cmn->bcn", field.flatten(2), self.tm)
        return scattered.abs().square().flatten(1)

    def digital(self, intensity):
        """对散斑执行可训练数字变换，输出下一光学块需要的非负图像。"""
        hidden = F.linear(self.norm(intensity), self.weight, self.bias).abs()
        return self.dropout(hidden).reshape(-1, 1, self.size, self.size)

    def forward(self, value):
        """完整前向仍为光学段后接数字段，参数名称与检查点格式保持一致。"""
        return self.digital(self.optical(value))


class DPNN(BlockNetwork):
    """相位编码 DPNN：各光学块使用相同固定介质，末端为独立数字分类头。"""

    def __init__(self, size=32, num_blocks=2, num_classes=10, tm_seed=31001,
                 encoding_alpha=1.0, layer_norm_eps=1e-5, bias=True, dropout=0.0):
        """使用独立 CPU 生成器创建复高斯介质，避免推进初始化随机状态。"""
        super().__init__()
        if size <= 0 or num_blocks <= 0:
            raise ValueError("size and num_blocks must be positive")
        dim = size * size
        generator = torch.Generator().manual_seed(tm_seed)
        real = torch.randn(1, dim, dim, generator=generator) / math.sqrt(dim)
        imag = torch.randn(1, dim, dim, generator=generator) / math.sqrt(dim)
        tm = torch.complex(real, imag)
        self.blocks = nn.ModuleList([
            *[OpticalBlock(tm, size, encoding_alpha, layer_norm_eps, bias, dropout)
              for _ in range(num_blocks)],
            nn.Sequential(nn.Flatten(1), nn.LayerNorm(dim, eps=layer_norm_eps),
                          nn.Linear(dim, num_classes, bias=bias))
        ])


class CNN(BlockNetwork):
    """定制 VGG 型教师，四个卷积块加池化及两个分类块。"""

    def __init__(self, channels=(32, 64, 128, 256), num_classes=10,
                 dropout=0.2, sigmoid=True):
        """构建卷积、池化和分类块；卷积无偏置，分类层有偏置。"""
        super().__init__()
        blocks = []
        incoming = 1
        for outgoing in channels:
            blocks.append(nn.Sequential(
                nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True),
                nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False),
                nn.BatchNorm2d(outgoing),
                nn.Sigmoid() if sigmoid else nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2)))
            incoming = outgoing
        width = max(incoming, 128)
        blocks.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Sequential(nn.Identity(), nn.Flatten(1), nn.Linear(incoming, width),
                          nn.ReLU(inplace=True), nn.Dropout(dropout)),
            nn.Sequential(nn.Linear(width, num_classes))])
        self.blocks = nn.ModuleList(blocks)


class MLP(BlockNetwork):
    """由 LN、Linear、ReLU 隐藏块和独立分类头组成的 MLP。"""

    def __init__(self, size=32, hidden_dims=(1024, 1024), num_classes=10, dropout=0.0):
        """按 hidden_dims 创建可独立蒸馏的隐藏块。"""
        super().__init__()
        if not hidden_dims or min(hidden_dims) <= 0:
            raise ValueError("hidden_dims must contain positive dimensions")
        blocks, dim = [], size * size
        for index, width in enumerate(hidden_dims):
            blocks.append(nn.Sequential(
                nn.Flatten(1) if index == 0 else nn.Identity(),
                nn.LayerNorm(dim), nn.Linear(dim, width), nn.ReLU(inplace=True),
                nn.Dropout(dropout)))
            dim = width
        blocks.append(nn.Linear(dim, num_classes))
        self.blocks = nn.ModuleList(blocks)


def build_model(config):
    """在独立 CPU 随机上下文中构造模型，初始化种子必须显式提供。"""
    options = dict(config)
    kind = options.pop("type")
    seed = options.pop("init_seed")
    constructors = {"dpnn": DPNN, "cnn": CNN, "mlp": MLP}
    if kind not in constructors:
        raise ValueError(f"Unknown model type: {kind}")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        return constructors[kind](**options)
