# PhysicalLocalDistillation

图像分类仿真的精简实现，包含相位编码 DPNN、定制 VGG 型 CNN、MLP、BP 和物理局部蒸馏（PLD）。
DPNN 执行相位编码、固定复高斯散射、强度探测和可训练数字变换，末端为独立分类头。
支持 CIFAR-10、Fashion-MNIST（配置名 fmnist）和 MNIST；源码仓库不附带数据或训练结果，已训练教师通过 Release 附件提供。

## 安装

使用 **Python 3.10**，在项目根目录创建独立环境。以下命令适用于 Linux x86_64；
Windows 可用 Python 3.10 创建 venv，并将激活命令改为 PowerShell 的
`.venv\Scripts\Activate.ps1`。自动复现队列仅面向 Linux NVIDIA GPU。

~~~bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip --index-url https://pypi.org/simple
~~~

根据机器选择下面 **一组** PyTorch 安装命令。先仅从官方索引获取 PyTorch/torchvision
二进制包，然后必须执行后续 requirements 安装以补全依赖。

CPU（无需 GPU，适合核心检查）：

~~~bash
python -m pip install --no-deps torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
~~~

NVIDIA GPU（CUDA 12.4，需兼容的 NVIDIA 驱动）：

~~~bash
python -m pip install --no-deps torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
~~~

然后执行：

~~~bash
python -m pip install -r requirements.txt --index-url https://pypi.org/simple
python -m pip check
python -m tests.check_core --device cpu
python -m unittest tests.check_release -v
~~~

版本组合及索引见 [PyTorch 官方历史版本安装说明](https://docs.pytorch.org/get-started/previous-versions/)。
直接依赖固定为 PyTorch 2.5.1、torchvision 0.20.1、NumPy 2.2.6 和 Pillow 9.4.0；
Pillow 版本保留图像预处理条件。requirements 不是所有平台的完整间接依赖锁文件。
记录自己的实际环境可运行 `python -m pip freeze > environment.txt`。
核心检查使用合成小样本，不需要下载数据，也不衡量完整训练准确率。
GPU 环境可额外运行 `python -m tests.check_core --device cuda:0`。

## 准备数据

默认目录是项目外的 `../dataset`，示例 JSON 均关闭自动下载。先运行：

~~~bash
python scripts/prepare_data.py cifar10 fmnist mnist --root ../dataset
python scripts/prepare_data.py cifar10 fmnist mnist --root ../dataset --verify-only
~~~

可只列出需要的数据集。脚本调用 torchvision 官方数据集下载与完整性检查。
网络无法连接数据源时，可在能联网的机器上准备，再复制整个数据目录到运行机器。
离线布局如下（MNIST/FashionMNIST 的 raw 中需保留解压后的 IDX 文件）：

~~~text
dataset/
  cifar-10-batches-py/
    data_batch_1 ... data_batch_5
    test_batch
    batches.meta
  FashionMNIST/raw/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
  MNIST/raw/
    train-images-idx3-ubyte
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte
    t10k-labels-idx1-ubyte
~~~

自定义目录时同步修改各 JSON 的 `data.root`。也可设 `data.download=true` 由训练入口准备，
但建议先单独完成下载。代码的 MIT 许可证不替代各数据集自身的使用条件。

## 已训练教师

CIFAR-10 和 Fashion-MNIST 的教师模型作为 [v1.0.0 Release 附件](https://github.com/MegaKeqing/PhysicalLocalDistillation/releases/tag/v1.0.0) 提供。
附件仅含教师参数与加载模型所需的配置，不含优化器、指标、训练日志或学生权重。
也可以按下文从头训练教师。MNIST 暂不提供已训练权重。

使用已训练教师时，在项目根目录执行：

~~~bash
mkdir -p checkpoint/teacher_model/cifar10 checkpoint/teacher_model/fmnist
curl --fail --location https://github.com/MegaKeqing/PhysicalLocalDistillation/releases/download/v1.0.0/cifar10_teacher.pt -o checkpoint/teacher_model/cifar10/best.pt
curl --fail --location https://github.com/MegaKeqing/PhysicalLocalDistillation/releases/download/v1.0.0/fmnist_teacher.pt -o checkpoint/teacher_model/fmnist/best.pt
~~~

Windows 可用 curl.exe 下载到相同目录。下载后核对 SHA-256：

| 附件 | SHA-256 |
|---|---|
| cifar10_teacher.pt | `c9bce9c1e3ef80d8e851ffd0d032c57dc645850216cec5bba21668bbb0311ec3` |
| fmnist_teacher.pt | `d6e1362f38b579e5eab8bf92c27b854ccdc08b395904bbca1de72880d11e2127` |

Linux 可运行 `sha256sum checkpoint/teacher_model/{cifar10,fmnist}/best.pt`；
PowerShell 可运行 `Get-FileHash checkpoint/teacher_model/*/best.pt -Algorithm SHA256`。

数据准备完成后，可直接运行对应的 DPNN/MLP PLD 配置，无需再次执行教师训练命令。
从头训练教师时使用尚不存在的输出目录，不要先在该目录下载附件。
下载的教师不包含完整运行记录，不能单独用于 `reproduce.py --checks-only`：
该选项仍要求本地已有全部六项完整运行。
自动复现队列会从头训练教师，因此应在其六项输出目录均不存在时启动。

## 训练入口

[配置说明](configs/README.md) 列出全部字段、模型结构和损失定义；
[命令索引](TASK.md) 列出三个数据集的入口。以下两种方式任选其一，避免重复创建相同输出目录。

**方式 A：逐项运行。** 以 CIFAR-10 为例，先从头训练教师：

~~~bash
python main.py --config configs/cifar10_teacher.json
python main.py --config configs/cifar10_bp.json
python main.py --config configs/cifar10_pld.json
~~~

文件名前缀改为 fmnist 或 mnist 可切换数据集；MLP 学生使用 `<dataset>_mlp_pld.json`。
teacher、bp 使用 BPTrainer，pld 使用 PLDTrainer。教师也可改为本项目的 DPNN 或 MLP，
同时调整学生的教师层映射和损失维度。教师 checkpoint 必须由本项目入口产生；
PLD 根据 checkpoint 中的配置构建教师。仅加载自己生成或可信来源的 checkpoint。

JSON 默认 `device="cuda:0"`。选择物理 GPU 可在命令前加 `CUDA_VISIBLE_DEVICES=0`。
CPU 训练需将配置副本的 device 改为 cpu，建议 num_workers=0、pin_memory=false。
CPU 能运行相同入口，但完整实验耗时较长，准确率门槛不作为 CPU 核心检查内容。

所有入口中的相对配置路径、data.root、output_dir、teacher_checkpoint 和 --output
均相对项目根目录解析，支持绝对路径和 ~。从其他目录调用时，使用 main.py 的实际路径。
保存的 config.json 保留原始 JSON 值；解析路径不会改写训练配置。
已有输出目录会被拒绝。--output 只改变当前运行输出；教师输出位置改变后，必须相应修改
学生的 pld.teacher_checkpoint。

**方式 B：自动复现队列。** 在六项输出目录均不存在时执行：

~~~bash
python scripts/reproduce.py --gpu 0 --output result/reproduction
~~~

脚本需要 nvidia-smi；检查指定物理 GPU 没有计算进程，依次训练 CIFAR-10 和 Fashion-MNIST
的教师、BP 和 PLD，再对这六项各重复前五轮。该队列不包含 MNIST 或 MLP。
--output 只指定队列日志与短检查目录，完整训练输出仍取自各 JSON。

如果方式 A 已完成上述 **全部六项**，保留相应 config.json、result.json、history.csv 和 best.pt，
可只执行前五轮检查：

~~~bash
python scripts/reproduce.py --checks-only --gpu 0 --output result/reproduction_checks
~~~

此时六份保存配置必须与当前 JSON 一致，输出目录需为新目录。它不是单项检查命令。
单项检查可执行（示例仍要求已训练好的教师）：

~~~bash
python main.py --config configs/cifar10_pld.json --check-epochs 5 --output result/cifar10_pld_check
~~~

## 训练协议和复现范围

三个数据集均转为 32×32 灰度图、按单张图像最大值归一化。官方训练集末尾 10,000 张用于验证，
其余用于训练；测试集采用官方划分。CIFAR-10 训练集为 40,000 张，另两个数据集为 50,000 张。
训练增强顺序是灰度、缩放、转张量、配置中的增强、逐样本归一化；验证和测试不增强。

示例学生使用两个光学块或两个 MLP 隐藏块，batch 64、AdamW lr=0.001、weight_decay=0.01，
最多 100 轮，至少 20 轮且连续 10 轮验证准确率未严格改善时停止。
教师训练 200 轮、无早停；CIFAR-10 教师 batch 32、weight_decay=0.01，水平/垂直翻转和裁剪；
Fashion-MNIST 教师 batch 64、weight_decay=0.0005，水平翻转和裁剪。
MNIST 沿用 Fashion-MNIST 的训练结构，是功能示例，未建立完整准确率验收结果。

PLD 冻结教师参数、BatchNorm 统计和 Dropout。每个学生块及自己的 PMSE 投影头独立优化。
下一块使用当前块这次前向的 detach 输出；更新后不重新计算输出，不跨块反向传播。
教师按批次在线提供目标。示例损失为 PMSE、PMSE、KL+CE，映射为 [1, 5, 6]。
支持 MSE、PMSE、教师特征 CKA、矩阵式 MI 和末层 KL+交叉熵。

全局、数据打乱、模型初始化、散射矩阵及投影初始化种子均在 JSON 显式设置。
BP/PLD 的介质和初始化实例可以不同；单组准确率差异不能解释为控制所有随机因素的因果效应。
严格启用确定性算法，关闭 TF32 与 cuDNN benchmark。
前五轮比较要求同一 JSON、同一教师权重、软件版本、设备类型和线程数。
不同 CPU/GPU 或软件构建之间不保证 loss 逐位一致。

自动队列的准确率验收条件为：

| 数据集 | 教师 | DPNN BP | DPNN PLD |
|---|---|---|---|
| CIFAR-10 | >79% | >43% 且 <47% | >47% 且高于 BP |
| Fashion-MNIST | >90% | >88% | >88% |

这些是验收条件，不是跨硬件、跨环境的保证。未通过时保留真实记录，不通过换种子或测试集调参追求达标。
五轮检查逐项比较 optimization_loss、train_loss、valid_loss 和所有 block_*_loss，要求数值完全相同；
只验证前五轮 loss，不推断完整轨迹或最终权重逐位一致。短检查不评估测试集。
MSE、CKA、MI 和 MLP 的核心数值/梯度检查不等同于完整准确率复现。

## 输出与代码结构

每次运行保存 config.json、manifest.json、history.csv、status.json 和验证最优 best.pt。
manifest 包含代码哈希、Git 信息、依赖环境及蒸馏所用教师摘要；无 Git 的源码压缩包也可运行，
此时 Git 字段为 null，代码哈希仍保存。
完整训练结束后加载验证最优权重，评估测试集一次并写 result.json；
短检查只写 check_result.json。检查点不提供中断续训入口。
教师 checkpoint 内含配置、参数、损失头及优化器状态，不需单独提供教师构造文件。

| 路径 | 用途 |
|---|---|
| main.py | 统一 JSON 入口 |
| pld/models.py | DPNN、CNN、MLP |
| pld/trainers.py | BP、逐块 PLD、验证选模 |
| pld/losses.py | 五种局部损失 |
| pld/data.py、pld/utils.py | 数据预处理、随机状态和运行记录 |
| configs/ | 12 份示例 JSON 和字段说明 |
| scripts/ | 数据准备和复现队列 |
| tests/ | 无真实数据的 CPU/GPU 核心检查及入口回归检查 |

权重、数据、运行结果和本地环境由 .gitignore 排除。代码采用 [MIT License](LICENSE)。
