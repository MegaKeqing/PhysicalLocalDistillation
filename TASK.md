# 训练命令索引

在项目根目录执行；先准备数据和训练教师，再运行相应蒸馏。
完整步骤见 [README](README.md)，字段说明见 [配置说明](configs/README.md)。

| 数据集 | CNN 教师 | DPNN BP | DPNN PLD |
|---|---|---|---|
| CIFAR-10 | `python main.py --config configs/cifar10_teacher.json` | `python main.py --config configs/cifar10_bp.json` | `python main.py --config configs/cifar10_pld.json` |
| Fashion-MNIST | `python main.py --config configs/fmnist_teacher.json` | `python main.py --config configs/fmnist_bp.json` | `python main.py --config configs/fmnist_pld.json` |
| MNIST | `python main.py --config configs/mnist_teacher.json` | `python main.py --config configs/mnist_bp.json` | `python main.py --config configs/mnist_pld.json` |

MLP 学生使用同数据集的 `configs/<dataset>_mlp_pld.json`。
teacher、bp、pld 的独立前五轮检查在同一命令后追加 `--check-epochs 5 --output result/<新的目录名>`。
检查结束不评估测试集；每次输出目录必须不存在。

## 固定散斑的阶段训练

| 任务 | 教师来源 | main 命令 |
|---|---|---|
| CIFAR-10 PLD-seq | CIFAR-10 CNN 教师 | python main.py --config configs/cifar10_pld_seq.json |
| CIFAR-10 TI-PLD | CNN 蒸馏所得 CIFAR-10 DPNN | python main.py --config configs/cifar10_ti_pld.json |

两项均使用每阶段独立早停；局部隐藏层验证损失选模，分类头验证准确率选模。
TI-PLD 的阶段验证采用教师注入，最终测试采用完整学生链。

两项不支持 --check-epochs。PLD-seq 需要先训练或下载 CNN 教师；TI-PLD 需要先完成 cifar10_pld.json，
生成 result/cifar10_pld/best.pt 作为 DPNN 教师。
