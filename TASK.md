# 训练命令索引

在项目根目录执行；先准备数据和训练教师，再运行相应蒸馏。
完整步骤见 [README](README.md)，字段说明见 [配置说明](configs/README.md)。

| 数据集 | CNN 教师 | DPNN BP | DPNN PLD |
|---|---|---|---|
| CIFAR-10 | `python main.py --config configs/cifar10_teacher.json` | `python main.py --config configs/cifar10_bp.json` | `python main.py --config configs/cifar10_pld.json` |
| Fashion-MNIST | `python main.py --config configs/fmnist_teacher.json` | `python main.py --config configs/fmnist_bp.json` | `python main.py --config configs/fmnist_pld.json` |
| MNIST | `python main.py --config configs/mnist_teacher.json` | `python main.py --config configs/mnist_bp.json` | `python main.py --config configs/mnist_pld.json` |

MLP 学生使用同数据集的 `configs/<dataset>_mlp_pld.json`。
独立前五轮检查在同一命令后追加 `--check-epochs 5 --output result/<新的目录名>`。
检查结束不评估测试集；每次输出目录必须不存在。
