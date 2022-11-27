# 概述

本文重点介绍如何利用**PaddleNLP**，完成基于Seq2Seq的机器翻译任务。

## 文章目录结构
- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleNLP安装
- 2 数据准备
  - 2.1 数据格式
  - 2.2 公开数据集
- 3 模型介绍
- 4 模型训练、验证与预测
  - 4.1 模型训练
  - 4.2 训练配置参数
  - 4.3 模型预测
  - 4.4 预测配置参数
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

假设需要将语言a翻译成语言b，则需要准备以下文件。




```text
data/
├── train.a # 训练数据集文件
├── train.b # 训练数据集文件
├── dev.a # 开发数据集文件
├── dev.b # 开发数据集文件
├── test.a # 开发数据集文件
├── test.b # 开发数据集文件
├── vocab.a # 字典文件
├── vocab.b # 字典文件
```

**训练、开发、测试数据集** `*.a`与`*.b`中的每一行文本均对应。

示例如下所示。

* train.a 文件内容

```text
Rachel Pike : The science behind a climate headline
In 4 minutes , atmospheric chemist Rachel Pike provides a glimpse of the massive scientific effort behind the bold headlines on climate change , with her team -- one of thousands who contributed -- taking a risky flight over the rainforest in pursuit of data on a key molecule .
I &apos;d like to talk to you today about the scale of the scientific effort that goes into making the headlines you see in the paper .
```

* train.b 文件内容

```text
Khoa học đằng sau một tiêu đề về khí hậu
Trong 4 phút , chuyên gia hoá học khí quyển Rachel Pike giới thiệu sơ lược về những nỗ lực khoa học miệt mài đằng sau những tiêu đề táo bạo về biến đổi khí hậu , cùng với đoàn nghiên cứu của mình -- hàng ngàn người đã cống hiến cho dự án này -- một chuyến bay mạo hiểm qua rừng già để tìm kiếm thông tin về một phân tử then chốt .
Tôi muốn cho các bạn biết về sự to lớn của những nỗ lực khoa học đã góp phần làm nên các dòng tít bạn thường thấy trên báo .
```

**字典文件** `vocab.*`中为该语言的字典文件，示例文件如下所示。

* 字典a

```text
<unk>
<s>
</s>
Rachel
:
The
science
behind
a
climate
```

* 字典b

```text
<unk>
<s>
</s>
Khoa
học
đằng
sau
một
tiêu
đề
```


## 2.2 公开数据集

本教程使用IWSLT'15 English-Vietnamese data 数据集中的英语到越南语的数据作为训练语料，tst2012的数据作为开发集，tst2013的数据作为测试集。使用下面的命令可以直接下载该数据集。

```shell
wget https://bj.bcebos.com/paddlenlp/datasets/iwslt15.en-vi.tar.gz
tar -xf iwslt15.en-vi.tar.gz
```

解压后，数据集的目录结构如下所示。

```
|------iwslt15.en-vi
    |-----train.en
    |-----train.vi
    |-----tst2012.en
    |-----tst2012.vi
    |-----tst2013.en
    |-----tst2013.vi
    |-----vocab.en
    |-----vocab.vi
```

其中`vocab.vi`是越南语的字典，`vocab.en`是英语的字典，`train.*`是训练集合，`tst2012.*`是开发集合，`tst2013.*`是测试集合。



PaddleNLP中也支持直接调用API，加载该数据集。

```py
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset('iwslt15', splits=('train', 'dev'))
```

如果希望了解加载数据集，并创建Dataloader的核心逻辑，请参考：[链接](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/seq2seq/data.py)。


# 3 模型介绍

本文展示的是Seq2Seq的一个经典样例：机器翻译，带Attention机制的翻译模型。Seq2Seq翻译模型，模拟了人类在进行翻译类任务时的行为：先解析源语言，理解其含义，再根据该含义来写出目标语言的语句。

本文档演示的模型中，在编码器方面，我们采用了基于LSTM的多层的RNN encoder；在解码器方面，我们使用了带注意力（Attention）机制的RNN decoder，在预测时我们使用柱搜索（beam search）算法来生成翻译的目标语句

更多详细介绍，请参考：[Machine Translation using Seq2Seq with Attention](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/machine_translation/seq2seq/README.md)。


# 4 模型训练、验证与预测

## 4.1 模型训练

首先进入工作目录。

```sh
cd PaddleNLP/examples/machine_translation/seq2seq/
```

使用下面的命令可以完成单卡的模型训练

```sh
python train.py \
    --num_layers 2 \
    --hidden_size 512 \
    --batch_size 128 \
    --dropout 0.2 \
    --init_scale  0.1 \
    --max_grad_norm 5.0 \
    --device gpu \
    --model_path ./attention_models
```

各参数的具体说明请参阅 `args.py` 。训练程序会在每个epoch训练结束之后，save一次模型。

**NOTE:** 如需恢复模型训练，则`init_from_ckpt`只需指定到文件名即可，不需要添加文件尾缀。如`--init_from_ckpt=attention_models/5`即可，程序会自动加载模型参数`attention_models/5.pdparams`，也会自动加载优化器状态`attention_models/5.pdopt`。

## 4.2 训练配置参数

训练过程中，可支持配置的参数如下。

* `learning_rate`: 训练过程中的学习率，默认为0.001。
* `num_layers`: encoder和decoder的层数，默认为1。
* `hidden_size`: 隐含层的大小，默认为100。
* `batch_size`: 批大小，默认为1。
* `max_epoch`: 总的迭代轮数，默认为12。
* `max_len`: 源与目标的最大句子长度，默认为50。
* `dropout`: dropout的概率，默认为0.2。
* `init_scale`: 初始参数的scale，默认为0。
* `max_grad_norm`: 梯度norm的最大值，默认为5.0，一般用于防止梯度发散。
* `log_freq`: 打印日志的频率，默认为100，表示每100个iter会打印一次。
* `model_path`: inference模型的保存路径，默认为`model`。
* `infer_output_file`: 输出的预测文件路径，默认为`infer_output`。
* `device`: 运行设备，默认为`gpu`，可选为`gpu`, `cpu`或者`xpu`。
* `init_from_ckpt`: 待加载的checkpoints模型路径，默认为`None`。




程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
attention_models/
├── 10.pdparams
└── 10.pdopt
```


## 4.3 模型预测

训练完成之后，可以使用保存的模型（由 `--init_from_ckpt` 指定）对测试集的数据集进行beam search解码。生成的翻译结果位于`--infer_output_file`指定的路径，预测命令如下：


```sh
python predict.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/9 \
     --infer_output_file infer_output.txt \
     --beam_size 10 \
     --device gpu
```

最终会输出

```text
W1123 01:47:06.191763  5324 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W1123 01:47:06.195843  5324 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
BLEU score is 0.23349343944555087.
```

取第10个epoch的结果，用取beam_size为10的beam search解码，`predict.py`脚本在生成翻译结果之后，会调用`paddlenlp.metrics.BLEU`计算翻译结果的BLEU指标，最终计算出的BLEU分数为0.23349343944555087。

同时，所有的预测结果也保存在`infer_output.txt`文件中。


## 4.4 预测配置参数

该部分与4.2章节的训练配置参数完全相同，可以直接参考4.2章节。


# 5 模型部署


# 5 模型部署

## 5.1 模型导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，静态图模型将用于**后续的推理部署工作**。具体代码见[静态图导出脚本](export_model.py)，运行方式如下所示。

```shell
python export_model.py \
     --num_layers 2 \
     --hidden_size 512 \
     --batch_size 128 \
     --dropout 0.2 \
     --init_scale  0.1 \
     --max_grad_norm 5.0 \
     --init_from_ckpt attention_models/9.pdparams \
     --beam_size 10 \
     --export_path ./infer_model/model
```

这里指定的参数`export_path` 表示导出预测模型文件的前缀。保存时会添加后缀（`pdiparams`，`pdiparams.info`，`pdmodel`）。


程序运行时将会自动导出模型到指定的 `export_path` 中，保存模型文件结构如下所示：

```text
infer_model/
├── model.pdiparams
├── model.pdiparams.info
└── model.pdmodel
```

## 5.2 模型推理


使用下面的命令完推理过程。

```sh
cd deploy/python
python infer.py \
    --export_path ../../infer_model/model \
    --device gpu \
    --batch_size 128 \
    --infer_output_file infer_output.txt
```

输出内容如下所示。

```text
I1123 01:53:52.348434  6302 naive_executor.cc:102] ---  skip [feed], feed -> src_length
I1123 01:53:52.348476  6302 naive_executor.cc:102] ---  skip [feed], feed -> src
I1123 01:53:52.349395  6302 naive_executor.cc:102] ---  skip [transpose_6.tmp_0], fetch -> fetch
W1123 01:53:53.020433  6302 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W1123 01:53:53.024305  6302 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
W1123 01:53:53.516913  6302 rnn_kernel.cu.cc:243] If the memory space of the Input WeightList is not continuous, less efficient calculation will be called. Please call flatten_parameters() to make the input memory continuous.
BLEU score is 0.23349343944555087.
```

所有的预测结果也保存在`infer_output.txt`文件中。可以看出，BLEU结果与动态图预测结果相同。


## 5.3 推理配置参数

该部分与4.2章节的训练配置参数完全相同，可以直接参考4.2章节。