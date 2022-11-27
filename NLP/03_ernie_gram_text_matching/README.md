# 概述

本文重点介绍如何利用**PaddleNLP**，完成基于ERNIE-Gram的文本匹配任务。


## 文章目录结构
- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleNLP安装
- 2 数据准备
  - 2.1 数据格式
  - 2.2 lcqmc公开数据集
- 3 模型介绍
- 4 模型训练、验证与预测
  - 4.1 模型训练
  - 4.2 训练配置参数
  - 4.3 模型预测
- 5 模型部署
  - 5.1 模型导出
  - 5.2 模型推理
  - 5.3 推理配置参数

# 1 环境安装

## 1.1 PaddlePaddle安装

### 1.1.1 安装对应版本PaddlePaddle

根据系统和设备的cuda环境，选择对应的安装包，这里默认使用pip在linux设备上进行安装。

<img width="821" alt="image" src="https://user-images.githubusercontent.com/48433081/176497642-0abf3de1-86d5-43af-afe8-f97db46b7fd9.png">

在终端执行

```sh
pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 1.1.2 验证安装是否成功

``` bash
# 安装完成后您可以使用 python进入python解释器，
python
# 继续输入
import paddle
# 再输入
paddle.utils.run_check()
```
如果出现PaddlePaddle is installed successfully!，说明您已成功安装。

## 1.2 训练相关依赖安装

首先使用下面的命令，下载PaddleNLP的代码。

```bash
git clone https://github.com/PaddlePaddle/PaddleNLP.git -b develop
```

接下来安装PaddleNLP依赖，下面2种方式均可。

* 方式1

```bash
# 这里会使用最新release出的whl包
pip install paddlenlp
```

* 方式2

```bash
# 这里会基于最基于最新代码安装whl包
pip install -r requirements.txt
pip install -e .
```


本项目依赖scikit-learn，因此安装完PaddleNLP后，还需要使用下面的命令安装scikit-learn。

```sh
python3 -m  pip install scikit-learn==1.0.2
```

## 1.3 推理相关依赖安装

模型转换与ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，Paddle2ONNX支持将Paddle静态图模型转化为ONNX模型格式，算子目前稳定支持导出ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

如果基于GPU部署，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:

```sh
python -m pip install onnxruntime-gpu onnx onnxconverter-common
```

如果基于CPU部署，请使用如下命令安装所需依赖:

```sh
python -m pip install onnxruntime
```

此外，推荐安装faster-tokenizer可以得到更极致的文本处理效率，进一步提升端到端推理性能。

```sh
pip install faster-tokenizer
```

# 2 数据准备

## 2.1 数据格式

指定格式本地数据集目录结构：

```text
data/
├── train.txt # 训练数据集文件
├── dev.txt # 开发数据集文件
├── test.txt # 可选，测试数据集文件
```

**训练、开发、测试数据集** 文件中2个输入文本内容与是否匹配用tab符`'\t'`分隔开。

```text
- train.txt/dev.txt/test.txt 文件格式：
```text
<文本>'\t'<文本>'\t'<标签>
<文本>'\t'<文本>'\t'<标签>
...
```

示例如下。

```text
开初婚未育证明怎么弄？	初婚未育情况证明怎么开？	1
谁知道她是网络美女吗？	爱情这杯酒谁喝都会醉是什么歌	0
人和畜生的区别是什么？	人与畜生的区别是什么！	1
男孩喝女孩的尿的故事	怎样才知道是生男孩还是女孩	0
```


## 2.2 lcqmc公开数据集

本文我们将以lcqmc公开数据集为例进行演示，paddlenlp中已经集成好，后续训练过程中，会自动下载该数据,使用逻辑如下所示。

```py
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])
```

上述python命令会把数据集下载到`~/.paddlenlp/datasets/LCQMC`目录中。


如果希望更直观地使用或者查看下载的数据集，也可以直接使用下面的命令下载并解压该数据集。

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/lcqmc.zip
unzip lcqmc.zip
```


# 3 模型介绍

粗粒度的语言信息，如命名实体或短语，有助于在预训练时进行充分的表征学习。以前的工作主要集中在扩展BERT的掩码语言建模(MLM)目标，从屏蔽单个标记到n个连续序列的标记。

作者认为这种连续掩码方法忽略了粗粒度语言信息的内在依赖和相互关系的建模。作为替代方案，论文中提出了ERNIE-gram，一种显式的n-gram掩蔽方法，以增强粗粒度信息集成到预训练。示意图如下所示。


<img width="842" alt="image" src="https://user-images.githubusercontent.com/14270174/202895553-3bc91c19-7321-4b57-81e5-25a05ff23135.png">


在ERNIE-Gram中，n-gram直接使用n-gram标识来屏蔽和预测，而不是使用n个连续标记序列。此外，ERNIE-Gram采用一个生成器模型来采样似然的n-gram标识，作为可选的n-gram掩码并以粗粒度和细粒度方式预测它们，以实现综合的n-gram预测和关系建模。

ERNIE-Gram在英语和汉语文本语料库上进行预训练，并对19个下游任务进行了微调。实验结果表明，ERNIE-Gram的表现明显优于以往的预训练模型如XLNet和RoBERTa，并与目前最先进的方法取得了相当的结果。

更多关于ERNIE-gram的详细解释，请参考：[ERNIE-Gram: Pre-Training with Explicitly N-Gram Masked Language Modeling for Natural Language Understanding](https://arxiv.org/pdf/2010.12148.pdf)。



# 4 模型训练、验证与预测

## 4.1 模型训练

首先进入工作目录，并把数据移动到当前目录。

```sh
cd PaddleNLP/examples/text_matching/ernie_matching
```


使用下面的命令可以完成单卡的模型训练

```sh
python train_pointwise.py \
    --device gpu \
    --save_dir ./checkpoints \
    --batch_size 32 \
    --learning_rate 2E-5
```

部分训练日志如下所示。

```
[2022-11-20 21:56:14,759] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.modeling.ErnieModel'> to load 'ernie-3.0-medium-zh'.
[2022-11-20 21:56:14,759] [    INFO] - Downloading https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh.pdparams and saved to /home/aistudio/.paddlenlp/models/ernie-3.0-medium-zh
100%|███████████████████████████████████████████████████████████████████████████████████| 320029/320029 [00:39<00:00, 8078.25it/s]
W1120 21:56:54.480510   881 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W1120 21:56:54.484766   881 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
[2022-11-20 21:56:56,686] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load 'ernie-3.0-medium-zh'.
[2022-11-20 21:56:56,686] [    INFO] - Downloading https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh_vocab.txt and saved to /home/aistudio/.paddlenlp/models/ernie-3.0-medium-zh
[2022-11-20 21:56:56,687] [    INFO] - Downloading ernie_3.0_medium_zh_vocab.txt from https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_medium_zh_vocab.txt
100%|██████████████████████████████████████████████████████████████████████████████████████████| 182k/182k [00:00<00:00, 3.00MB/s]
[2022-11-20 21:56:56,849] [    INFO] - tokenizer config file saved in /home/aistudio/.paddlenlp/models/ernie-3.0-medium-zh/tokenizer_config.json
[2022-11-20 21:56:56,849] [    INFO] - Special tokens file saved in /home/aistudio/.paddlenlp/models/ernie-3.0-medium-zh/special_tokens_map.json
global step 10, epoch: 1, batch: 10, loss: 0.55552, accu: 0.60938, speed: 6.10 step/s
global step 20, epoch: 1, batch: 20, loss: 0.32429, accu: 0.70781, speed: 20.07 step/s
global step 30, epoch: 1, batch: 30, loss: 0.20013, accu: 0.76146, speed: 21.28 step/s
global step 40, epoch: 1, batch: 40, loss: 0.14908, accu: 0.79531, speed: 19.68 step/s
global step 50, epoch: 1, batch: 50, loss: 0.25992, accu: 0.81375, speed: 19.65 step/s
global step 60, epoch: 1, batch: 60, loss: 0.52279, accu: 0.82292, speed: 20.63 step/s
global step 70, epoch: 1, batch: 70, loss: 0.24669, accu: 0.83170, speed: 18.52 step/s
global step 80, epoch: 1, batch: 80, loss: 0.09662, accu: 0.84102, speed: 21.80 step/s
global step 90, epoch: 1, batch: 90, loss: 0.39619, accu: 0.84306, speed: 21.28 step/s
global step 100, epoch: 1, batch: 100, loss: 0.30451, accu: 0.84688, speed: 19.30 step/s
eval dev loss: 0.49129, accu: 0.78437
global step 110, epoch: 1, batch: 110, loss: 0.21181, accu: 0.90000, speed: 1.19 step/s
global step 120, epoch: 1, batch: 120, loss: 0.22850, accu: 0.88906, speed: 20.17 step/s
global step 130, epoch: 1, batch: 130, loss: 0.18250, accu: 0.89479, speed: 18.87 step/s
global step 140, epoch: 1, batch: 140, loss: 0.28898, accu: 0.90469, speed: 20.97 step/s
global step 150, epoch: 1, batch: 150, loss: 0.30832, accu: 0.90375, speed: 20.40 step/s
global step 160, epoch: 1, batch: 160, loss: 0.31764, accu: 0.90156, speed: 19.73 step/s
global step 170, epoch: 1, batch: 170, loss: 0.25749, accu: 0.90357, speed: 19.43 step/s
global step 180, epoch: 1, batch: 180, loss: 0.35511, accu: 0.90430, speed: 20.45 step/s
global step 190, epoch: 1, batch: 190, loss: 0.20937, accu: 0.90451, speed: 20.19 step/s
global step 200, epoch: 1, batch: 200, loss: 0.33983, accu: 0.90344, speed: 18.91 step/s
eval dev loss: 0.46684, accu: 0.80425
```

如果在CPU环境下训练，可以指定`nproc_per_node`参数进行多核训练：

```sh
python -m paddle.distributed.launch --nproc_per_node 8 --backend "gloo" \
    train_pointwise.py \
    --device gpu \
    --save_dir ./checkpoints \
    --batch_size 32 \
    --learning_rate 2E-5
```

如果在GPU环境中使用，可以指定`gpus`参数进行单卡/多卡训练。使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况:

```sh
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus "0,1" train_pointwise.py \
    --device gpu \
    --save_dir ./checkpoints \
    --batch_size 32 \
    --learning_rate 2E-5
```

## 4.2 训练配置参数

训练过程中，可支持配置的参数如下所示。

* `save_dir`：可选，保存训练模型的目录；默认保存在当前目录checkpoints文件夹下。
* `max_seq_length`：可选，ERNIE-Gram 模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：可选，批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：可选，Fine-tune的最大学习率；默认为5e-5。
* `weight_decay`：可选，控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `epochs`: 训练轮次，默认为3。
* `warmup_proption`：可选，学习率warmup策略的比例，如果0.1，则学习率会在前10%训练step的过程中从0慢慢增长到learning_rate, 而后再缓慢衰减，默认为0.0。
* `init_from_ckpt`：可选，模型参数路径，热启动模型训练；默认为None。
* `seed`：可选，随机种子，默认为1000.
* `device`: 选用什么设备进行训练，可选cpu或gpu。如使用gpu训练则参数gpus指定GPU卡号。

代码示例中使用的预训练模型是 ERNIE-Gram，如果想要使用其他预训练模型如 ERNIE, BERT，RoBERTa，Electra等，只需更换`model` 和 `tokenizer`即可。


程序运行时将会自动进行训练，评估。同时训练过程中会自动保存模型在指定的`save_dir`中，如：

```text
checkpoints/
├── model_100
│   ├── model_state.pdparams
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:**

* 如需恢复模型训练，则可以设置`init_from_ckpt`， 如`init_from_ckpt=checkpoints/model_100/model_state.pdparams`。
* 如需使用ernie-tiny模型，则需要提前先安装sentencepiece依赖，如`pip install sentencepiece`


## 4.3 模型预测

本文档将 LCQMC 的测试集作为预测数据,  测试数据示例如下，：

```text
谁有狂三这张高清的  这张高清图，谁有
英雄联盟什么英雄最好    英雄联盟最好英雄是什么
这是什么意思，被蹭网吗  我也是醉了，这是什么意思
现在有什么动画片好看呢？    现在有什么好看的动画片吗？
请问晶达电子厂现在的工资待遇怎么样要求有哪些    三星电子厂工资待遇怎么样啊
```

首先通过下面的命令，将`test.csv`拷贝到当前目录。

```sh
cp ~/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/test.tsv .
```

启动预测：

```shell
python predict_pointwise.py \
        --device gpu \
        --params_path "./checkpoints/model_4400/model_state.pdparams"\
        --batch_size 128 \
        --max_seq_length 64 \
        --input_file 'test.tsv'
```

输出预测结果如下:

```text
{'query': '谁有狂三这张高清的', 'title': '这张高清图，谁有', 'pred_label': 1}
{'query': '英雄联盟什么英雄最好', 'title': '英雄联盟最好英雄是什么', 'pred_label': 1}
{'query': '这是什么意思，被蹭网吗', 'title': '我也是醉了，这是什么意思', 'pred_label': 1}
{'query': '现在有什么动画片好看呢？', 'title': '现在有什么好看的动画片吗？', 'pred_label': 1}
{'query': '请问晶达电子厂现在的工资待遇怎么样要求有哪些', 'title': '三星电子厂工资待遇怎么样啊', 'pred_label': 0}
```

# 5 模型部署

## 5.1 模型导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，静态图模型将用于**后续的推理部署工作**。具体代码见[静态图导出脚本](export_model.py)，静态图参数保存在`output_path`指定路径中。运行方式如下。

```sh
python export_model.py --params_path checkpoints/model_300/model_state.pdparams --output_path=./output
```

程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
output/
├── inference.pdiparams
├── inference.pdiparams.info
└── inference.pdmodel
```

## 5.2 模型推理

模型ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，推理前首先需要参考1.3章节安装依赖。

使用下面的命令完推理过程。

```sh
python deploy/python/predict.py --model_dir ./output
```

输出

```
Data: {'query': '叉车证怎么查', 'title': '叉车证怎么考'}         Label: dissimilar
Data: {'query': '行车记录仪什么牌子的比较好？', 'title': '什么牌子的行车记录仪比较好?'}          Label: similar
Data: {'query': '墙壁开关怎么拆', 'title': '墙壁开关如何接？'}   Label: dissimilar
Data: {'query': '最近什么歌好听啊？', 'title': '最近什么歌好听'}         Label: similar
Data: {'query': '少年阿宾怎么下载？', 'title': '怎么下载少年阿宾'}       Label: similar
Data: {'query': '移动硬盘问题', 'title': '我的移动硬盘出问题了'}         Label: similar
```

## 5.3 推理配置参数

推理过程中，可支持配置的参数如下。

* `model_dir`: inference模型路径，必填。
* `max_seq_length`: 最大的序列长度，默认为128。
* `batch_size`: 预测时候的批大小，默认为32。
* `device`: 设备，可选项为`['cpu', 'gpu', 'xpu']`，默认为`gpu`。
* `use_tensorrt`: 是否使用TensorRT，默认为`False`。
* `precision`: 精度设置，可选项为`["fp32", "fp16", "int8"]`，默认为`fp32`。
* `cpu_threads`: CPU线程数，默认为10。
* `enable_mkldnn`: 是否使用mkldnn，默认为False。
* `benchmark`: 是否使用benchmark，默认为False。
* `save_log_path`: 保存的日志路径，默认为`log_output`。

