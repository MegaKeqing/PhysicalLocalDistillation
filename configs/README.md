# 配置说明

main.py 只接受 JSON，不依赖私有配置或预训练权重。先用 `<dataset>_teacher.json` 训练 CNN 教师，
再用同数据集的 _pld.json 或 _mlp_pld.json 蒸馏；_bp.json 不需要教师。
CIFAR-10/Fashion-MNIST 也可使用 README 中的已训练教师附件，放入配置指定的路径后直接蒸馏。
dataset 可取 cifar10、fmnist、mnist。修改配置后属于新的运行条件，应使用新的输出目录。

## 通用字段

| 字段 | 类型、含义 |
|---|---|
| task | teacher、bp 或 pld；前两者都使用全网络交叉熵 BP |
| seed | 全局整数种子，范围 [0, 2^32) |
| device | cpu 或 cuda:0 等 PyTorch 设备名 |
| threads | PyTorch CPU 线程数，默认 4 |
| output_dir | 本次输出目录；必须不存在 |
| data、model、training | 下表中的配置对象 |
| pld | 仅蒸馏时需要 |

相对路径以项目根目录为基准，绝对路径保持其位置。--output 覆盖本次输出位置，
不改变保存的原始 JSON 或其他配置中的教师路径。--check-epochs N 限制本次检查轮数，
不修改 JSON，不评估测试集；N 必须在 1 到 training.epochs 之间。

## data

| 字段 | 类型、默认值及含义 |
|---|---|
| dataset | cifar10、fmnist、mnist |
| root | 数据目录；示例 ../dataset，三个数据集可共用 |
| size | 输入方形边长；示例 32，DPNN/MLP 必须与 model.size 一致 |
| batch_size | 同时用于训练、训练集评估、验证和测试的批大小 |
| valid_size | 官方训练集末尾用于验证的样本数；示例 10000，不随机划分 |
| train_size | 可选，只取训练划分前 N 个样本；省略或 null 使用整个训练划分，不能用于声称完整样本实验 |
| download | 是否允许 torchvision 下载，默认 false；推荐先运行 prepare_data.py |
| num_workers | DataLoader 工作进程数；排查平台问题可设 0，但应记录变化 |
| pin_memory | 是否使用锁页内存 |
| augmentation | 按数组顺序应用增强；[] 为无增强，可选 horizontal_flip、vertical_flip、crop |
| shuffle_seed | 数据流种子；四个加载器分别使用此值 + 0、1、2、3 |

水平和垂直翻转概率各为 0.5；crop 的 padding=max(1, size//8)，输出仍为 size×size。
仅训练加载器打乱数据，所有加载器 drop_last=false。不要用验证/测试样本训练。

## model

所有模型必需 type 与 init_seed；type 为 dpnn、cnn 或 mlp。
初始化使用独立 CPU 随机上下文，不推进训练随机流。所有示例 num_classes=10。

| 模型 | 字段与默认值 |
|---|---|
| 共用 | num_classes=10；dropout：DPNN/MLP 默认 0，CNN 默认 0.2 |
| dpnn | size=32，num_blocks=2，tm_seed=31001，encoding_alpha=1.0，layer_norm_eps=1e-5，bias=true |
| cnn | channels=[32,64,128,256]，sigmoid=true |
| mlp | size=32，hidden_dims=[1024,1024] |

DPNN 的 tm_seed 控制固定复高斯介质，所有光学块共享同一介质；
init_seed 控制可训练参数。每个光学块执行逐样本归一化、exp(jπαx)、散射、
强度探测、LayerNorm、Linear、绝对值和 Dropout，末端为 Flatten、LayerNorm、Linear 分类头。
DPNN 输入必须有限、非负且形状为 [batch,1,size,size]。

CNN 每个卷积块是 Conv-BN-ReLU、Conv-BN-Sigmoid、MaxPool；
sigmoid=false 时第二个激活改为 ReLU。它是定制网络，不是 torchvision 标准 VGG16。
默认四个卷积块后有自适应池化、隐藏分类块、输出分类块，共七块。
32×32 输入适配默认四次池化；增加块数需保证空间尺寸能继续池化。
CNN 不接收 size 字段，空间尺寸由 data.size 决定。

MLP 每个隐藏块使用 LayerNorm、Linear、ReLU、Dropout，最后为 Linear 分类头。
改变模型块数或宽度时应重新设置蒸馏映射和损失形状。

## training

| 字段 | 含义 |
|---|---|
| epochs | 最大轮数，正整数 |
| learning_rate | AdamW 学习率 |
| weight_decay | AdamW 权重衰减，作用于各优化器全部参数 |
| early_stopping | null 表示关闭，否则必须包含 min_epochs 和 patience 两个正整数 |

AdamW 使用 betas=(0.9,0.999)、eps=1e-8，无学习率调度或混合精度。
验证准确率严格增大时才保存 best.pt；持平不会替换最优权重。
达到 min_epochs 且距最佳轮数至少 patience 轮时停止。
前 N 轮检查也保留此规则；若提前停止不足 N 轮，重复性检查不会通过。
完整训练只在结束后用验证最优模型评估一次官方测试集。

## pld 与损失

| 字段 | 含义 |
|---|---|
| teacher_checkpoint | 本项目训练生成的教师 best.pt |
| mapping | 每个学生块对应的教师块索引，从 0 开始，含末端分类头 |
| losses | 每个学生块的损失对象，长度与 mapping、学生块数一致 |
| projection_seed | PMSE 投影层的独立初始化种子 |

默认两块学生加分类头共三块，教师默认七块，mapping=[1,5,6]：
第一光学/隐藏块对齐教师第二卷积块，第二块对齐教师隐藏分类块，学生分类头对齐教师 logits。
教师和学生的 dataset、size、valid_size 必须一致；教师增强可不同。
不会缓存教师特征或把教师特征作为学生下一块输入。

每个 losses 对象含 type 与对应选项；不支持的选项会报错：

| type | 选项及默认值 | 最小化目标 |
|---|---|---|
| mse | weight=1.0 | weight × mean((student−teacher)²)，特征形状必须完全相同 |
| pmse | weight=1.0，bias=false | 学生展平特征经可训练 Linear 投影后的 MSE；允许维度不同 |
| cka | weight_feat=1.0，eps=1e-8 | weight_feat × (1−linear CKA(student,teacher))；只对齐教师特征 |
| mi | weight=1.0，alpha=null，eps=1e-8 | 负的矩阵式互信息代理量；允许特征维度不同 |
| kl_ce | weight=1.0，label_weight=0.5，temperature=3.0 | weight × T² × KL(teacher/T ∥ student/T) + label_weight × CE(student,label) |

KL+CE 用于末层分类 logits，教师与学生类别数必须一致；
示例 weight=0.5、label_weight=0.5、T=3。标签交叉熵不使用温度缩放。
零权重损失不产生梯度；KL 和 CE 两项都为零时，该块不执行优化器更新。

CKA 先对样本特征按批次中心化，用样本 Gram 计算相似度，分母根号内加 eps。
MI 用按样本数缩放的余弦 Gram；联合核经迹归一化，特征值分解加 1e-8 对角抖动。
alpha=null 使用 Shannon 熵（nat）；正数 alpha 且 |alpha−1|>=1e-3 使用 Rényi 熵（bit）。
零或近零特征导致 Gram 迹偏离 1 时会提示。MI 是有限样本代理量，不是绝对互信息测量。

## 示例范围与输出

CIFAR-10/Fashion-MNIST 的教师、DPNN BP、DPNN PLD 配置用于自动复现队列；
MNIST 和三个 MLP 配置提供可运行接口示例，未建立相同的完整准确率验收。
原始配置、模型状态和环境会随每次运行保存；不提供自动续训或测试集调参分支。
