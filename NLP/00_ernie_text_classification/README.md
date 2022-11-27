# 概述

本文重点介绍如何利用**PaddleNLP**，完成基于ERNIE-3.0的文本多分类任务。

## 文章目录结构
- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleNLP安装
- 2 数据准备
  - 2.1 数据格式
  - 2.2 公开数据集
  - 2.3 数据标注工具
- 3 模型介绍
- 4 模型训练、验证与预测
  - 4.1 模型训练
  - 4.2 训练配置参数
  - 4.3 模型评估
  - 4.4 模型预测
  - 4.5 预测配置参数
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
├── label.txt # 分类标签文件
└── data.txt # 待预测数据文件
```

**训练、开发、测试数据集** 文件中文本与标签类别名用tab符`'\t'`分隔开。

```text
- train.txt/dev.txt/test.txt 文件格式：
```text
<文本>'\t'<标签>
<文本>'\t'<标签>
...
```

label.txt(分类标签文件)记录数据集中所有标签集合，每一行为一个标签名。

- label.txt 文件格式：

```text
<标签>
<标签>
...
```

- label.txt 文件样例：

```text
病情诊断
治疗方案
病因分析
指标解读
就医建议
...
```

**待预测数据**

data.txt(待预测数据文件)，需要预测标签的文本数据。

- data.txt 文件格式：

```text
<文本>
<文本>
...
```

- data.txt 文件样例：

```text
黑苦荞茶的功效与作用及食用方法
交界痣会凸起吗
检查是否能怀孕挂什么科
鱼油怎么吃咬破吃还是直接咽下去
...
```

## 2.2 CBLUE公开数据集

本文我们将以CBLUE公开数据集KUAKE-QIC任务为示例，演示多分类全流程方案使用。下载数据集：

```shell
wget https://paddlenlp.bj.bcebos.com/datasets/KUAKE_QIC.tar.gz
tar -zxvf KUAKE_QIC.tar.gz
mv KUAKE_QIC data
rm KUAKE_QIC.tar.gz
```

## 2.3 数据标注工具

如果你希望自己标注数据并参与训练，可以参考[文本分类任务doccano数据标注使用指南](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/applications/text_classification/doccano.md)进行文本分类数据标注。下面对其中的关键步骤进行介绍。

**注意：**： 该数据标注工具要求3.8+的python版本，使用的时候需要注意python版本。

首先安装doccano。

```sh
pip install doccano
```

安装以后，运行下面的命令，创建用户。

```sh
# Initialize database.
doccano init
# Create a super user.
doccano createuser --username admin --password pass
```

然后使用下面的命令，打开端口，启动服务。

```sh
doccano webserver --port 8000
```

此时，我们可以在浏览器中访问`http://127.0.0.1:8000`，进入doccano标注界面，如下所示。

<img width="1400" alt="image" src="https://user-images.githubusercontent.com/14270174/202880011-75ccf902-4c84-461a-a9a6-a18109f27c38.png">



点击右上角的登录按钮并登录之后，可以点击下面的`start annoataions`按钮，开始标注。

<img width="1298" alt="image" src="https://user-images.githubusercontent.com/14270174/202885313-fe42582f-41c5-4a28-9b94-96b9b23c0821.png">

点击上面的`Actions`并导入文本数据，没有标注的文本行数据以文本行的形式存储（每一行为一条样本），选择的模式如下所示。

<img width="800" alt="image" src="https://user-images.githubusercontent.com/63761690/176859861-b790288f-32d7-4ab0-8b5f-b30e97f8c306.png">

标注数据中文本行内容示例如下所示。

```txt
黑苦荞茶的功效与作用及食用方法
交界痣会凸起吗
检查是否能怀孕挂什么科
鱼油怎么吃咬破吃还是直接咽下去
幼儿挑食的生理原因是
```

点击标签-操作-创建标签，开始添加分类类别标签：
<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176860972-eb9cacf1-199a-4cec-9940-6858434cfb94.png height=300 hspace='15'/>
</div>

填入分类类别标签，选择标签颜色，建议不同标签选择不同颜色，最后点击保存或保存并添加下一个，保存标签：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176860977-55292e2a-8bf8-4316-a0f8-b925872e5023.png height=300 hspace='15'/>
</div>

文本分类标签构建示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176860996-542cd1f7-9770-4b22-9586-a5bf0e802970.png height=300 hspace='15'/>
</div>


分类标签构建完成后，在数据集中，选择对应的分类类别标签，输入回车（Enter）键确认：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176872684-4a19f592-be5c-4b86-8adf-eb0a7d7aa375.png height=200 hspace='10'/>
</div>

数据全部标注完成后，对数据集进行导出。

点击`数据集-操作-导出数据集`，将标注好的数据导出，我们默认所有数据集已经标注完成且正确：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176874195-d21615f4-8d53-4033-8f53-2106ebdf21f8.png height=250 hspace='20'/>
</div>

选择导出的文件类型为``JSONL``，导出数据：

<div align="center">
    <img src=https://user-images.githubusercontent.com/63761690/176873347-fd995e4e-5baf-4d13-92b9-800cabd1f0b1.png height=300 hspace='20'/>
</div>

导出数据示例：
```text
{"id": 23, "data": "黑苦荞茶的功效与作用及食用方法", "label": ["功效作用"]}
{"id": 24, "data": "交界痣会凸起吗", "label": ["疾病表述"]}
{"id": 25, "data": "检查是否能怀孕挂什么科", "label": ["就医建议"]}
{"id": 26, "data": "鱼油怎么吃咬破吃还是直接咽下去", "label": ["其他"]}
{"id": 27, "data": "幼儿挑食的生理原因是", "label": ["病因分析"]}
```

导出后，使用脚本`trans_data.py`，将导出的数据转换为用于训练的数据集格式，需要指定输入文件路径、输出文件路径以及标签文件路径（如果没有事先生成的话）。

```sh
python trans_data.py
```

# 3 模型介绍

BERT模型通过随机屏蔽15%的word，使用transformer的多层self-attention双向建模能力，在各项nlp下游任务中(如sentence pair classification task, singe sentence classification task, question answering task) 都取得了很好的成绩。但是BERT模型主要聚焦在针对字护着英文word粒度的完形填空学习上面，没有充分利用训练数据集的词法结构。

ERNIE中，通过对训练数据中的词法结构，语法结构，语义信息进行统一建模，极大地增强了通用语义表示能力，在多项任务中均取得了大幅度超越BERT的效果。

ERNIE-3.0的结构示意图如下所示。ERNIE-3.0设计了一个新的连续多范式统一预训练框架，即对不同的精心设计的完形填空任务采用共享的Transformer网络，并利用特定的self-attention mask来控制预测条件的内容。

<img width="1026" alt="image" src="https://user-images.githubusercontent.com/14270174/202886931-da9bb02a-5f56-43e9-badd-becccda27f42.png">

ERNIE-3.0中将骨干共享网络和特定任务网络称为通用表示模块和特定任务表示模块。具体来说，通用表示网络扮演着通用语义特征提取器的actor（例如，它可以是一个多层transformer），其中的参数在各种任务范式中都是共享的，包括自然语言理解、自然语言生成等等。而特定任务的表示网络承担着提取特定任务语义特征的特征，其中的参数是由特定任务的目标学习的。ERNIE-3.0不仅使模型能够区分不同任务范式的特定语义信息，而且缓解了大规模预训练模型在有限的时间和硬件资源下难以实现的困境，其中ERNIE-3.0允许模型只在微调阶段更新特定任务表示网络的参数。

更多详细介绍，请参考：[ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation](https://arxiv.org/abs/2107.02137)。


# 4 模型训练、验证与预测

## 4.1 模型训练

首先进入工作目录，并把数据移动到当前目录。

```sh
cd PaddleNLP/applications/text_classification/multi_class
# 把数据从之前下载的目录挪到当前工作目录
mv ../../../data/  .
```


使用下面的命令可以完成单卡的模型训练

```sh
python train.py \
    --dataset_dir "data" \
    --device "gpu" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --epochs 100
```

如果在CPU环境下训练，可以指定`nproc_per_node`参数进行多核训练：

```sh
python -m paddle.distributed.launch --nproc_per_node 8 --backend "gloo" train.py \
    --dataset_dir "data" \
    --device "cpu" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --epochs 100
```

如果在GPU环境中使用，可以指定`gpus`参数进行单卡/多卡训练。使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况:

```sh
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus "0,1" train.py \
    --dataset_dir "data" \
    --device "gpu" \
    --max_seq_length 128 \
    --model_name "ernie-3.0-medium-zh" \
    --batch_size 32 \
    --early_stop \
    --epochs 100
```

## 4.2 训练配置参数

训练过程中，可支持配置的参数如下。

* `device`: 选用什么设备进行训练，选择cpu、gpu、xpu、npu。如使用gpu训练，可使用参数--gpus指定GPU卡号；默认为"gpu"。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含train.txt，dev.txt和label.txt文件;默认为None。
* `save_dir`：保存训练模型的目录；默认保存在当前目录checkpoint文件夹下。
* `max_seq_length`：分词器tokenizer使用的最大序列长度，ERNIE模型最大不能超过2048。请根据文本长度选择，通常推荐128、256或512，若出现显存不足，请适当调低这一参数；默认为128。
* `model_name`：选择预训练模型,可选"ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"；默认为"ernie-3.0-medium-zh"。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `learning_rate`：训练最大学习率；默认为3e-5。
* `epochs`: 训练轮次，使用早停法时可以选择100；默认为10。
* `early_stop`：选择是否使用早停法(EarlyStopping)，模型在开发集经过一定epoch后精度表现不再上升，训练终止；默认为False。
* `early_stop_nums`：在设定的早停训练轮次内，模型在开发集上表现不再上升，训练终止；默认为4。
* `logging_steps`: 训练过程中日志打印的间隔steps数，默认5。
* `weight_decay`：控制正则项力度的参数，用于防止过拟合，默认为0.0。
* `warmup`：是否使用学习率warmup策略，使用时应设置适当的训练轮次（epochs）；默认为False。
* `warmup_steps`：学习率warmup策略的比例数，如果设为1000，则学习率会在1000steps数从0慢慢增长到learning_rate, 而后再缓慢衰减；默认为0。
* `init_from_ckpt`: 模型初始checkpoint参数地址，默认None。
* `seed`：随机种子，默认为3。
* `train_file`：本地数据集中训练集文件名；默认为"train.txt"。
* `dev_file`：本地数据集中开发集文件名；默认为"dev.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。


程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
checkpoint/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
└── vocab.txt
```

**NOTE:**

* 如需恢复模型训练，则可以设置 `init_from_ckpt` ， 如 `init_from_ckpt=checkpoint/model_state.pdparams` 。
* 如需训练英文文本分类任务，只需更换预训练模型参数 `model_name` 。英文训练任务推荐使用"ernie-2.0-base-en"、"ernie-2.0-large-en"。
* 英文和中文以外语言的文本分类任务，推荐使用基于96种语言（涵盖法语、日语、韩语、德语、西班牙语等几乎所有常见语言）进行预训练的多语言预训练模型"ernie-m-base"、"ernie-m-large"，详情请参见[ERNIE-M论文](https://arxiv.org/pdf/2012.15674.pdf)。


## 4.3 模型评估

训练后的模型我们可以使用 评估脚本 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`:

```sh
python analysis/evaluate.py --device "gpu" --max_seq_length 128 --batch_size 32 --bad_case_path "./bad_case.txt" --dataset_dir "data" --params_path "./checkpoint"
```

输出打印示例：

```text
[2022-08-10 06:28:37,219] [    INFO] - -----Evaluate model-------
[2022-08-10 06:28:37,219] [    INFO] - Train dataset size: 6931
[2022-08-10 06:28:37,220] [    INFO] - Dev dataset size: 1955
[2022-08-10 06:28:37,220] [    INFO] - Accuracy in dev dataset: 81.79%
[2022-08-10 06:28:37,221] [    INFO] - Top-2 accuracy in dev dataset: 92.48%
[2022-08-10 06:28:37,222] [    INFO] - Top-3 accuracy in dev dataset: 97.24%
[2022-08-10 06:28:37,222] [    INFO] - Class name: 病情诊断
[2022-08-10 06:28:37,222] [    INFO] - Evaluation examples in train dataset: 877(12.7%) | precision: 97.14 | recall: 96.92 | F1 score 97.03
[2022-08-10 06:28:37,222] [    INFO] - Evaluation examples in dev dataset: 288(14.7%) | precision: 80.32 | recall: 86.46 | F1 score 83.28
[2022-08-10 06:28:37,223] [    INFO] - ----------------------------
[2022-08-10 06:28:37,223] [    INFO] - Class name: 治疗方案
[2022-08-10 06:28:37,223] [    INFO] - Evaluation examples in train dataset: 1750(25.2%) | precision: 96.84 | recall: 99.89 | F1 score 98.34
[2022-08-10 06:28:37,223] [    INFO] - Evaluation examples in dev dataset: 676(34.6%) | precision: 88.46 | recall: 94.08 | F1 score 91.18
...
```

这里，预测错误的样本保存在bad_case.txt文件中：

```text
Confidence	Prediction	Label	Text
0.77	注意事项	其他	您好，请问一岁三个月的孩子可以服用复方锌布颗粒吗？
0.94	就医建议	其他	输卵管粘连的基本检查
0.78	病情诊断	其他	经常干呕恶心，这是生病了吗
0.79	后果表述	其他	吃左旋肉碱后的不良反应
...
```

## 4.4 模型预测

训练结束后，输入待预测数据(data.txt)和类别标签对照列表(label.txt)，使用训练好的模型进行，默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`：

```shell
python predict.py --device "gpu" --max_seq_length 128 --batch_size 32 --dataset_dir "data"
```

## 4.5 预测配置参数

预测过程中，可支持配置的参数如下。

* `device`: 选用什么设备进行预测，可选cpu、gpu、xpu、npu；默认为gpu。
* `dataset_dir`：必须，本地数据集路径，数据集路径中应包含data.txt和label.txt文件;默认为None。
* `params_path`：待预测模型的目录；默认为"./checkpoint/"。
* `max_seq_length`：模型使用的最大序列长度,建议与训练时最大序列长度一致, 若出现显存不足，请适当调低这一参数；默认为128。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `data_file`：本地数据集中未标注待预测数据文件名；默认为"data.txt"。
* `label_file`：本地数据集中标签集文件名；默认为"label.txt"。


# 5 模型部署

## 5.1 模型导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，静态图模型将用于**后续的推理部署工作**。具体代码见[静态图导出脚本](export_model.py)，静态图参数保存在`output_path`指定路径中。运行方式如下。

```sh
python export_model.py --params_path ./checkpoint/ --output_path ./export
```

可支持配置的参数：
* `multilingual`：是否为多语言任务（是否使用ERNIE M作为预训练模型）；默认为False。
* `params_path`：动态图训练保存的参数路径；默认为"./checkpoint/"。
* `output_path`：静态图图保存的参数路径；默认为"./export"。

程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
export/
├── float32.pdiparams
├── float32.pdiparams.info
└── float32.pdmodel
```

## 5.2 模型推理

模型ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，推理前首先需要参考1.3章节安装依赖。

使用下面的命令完推理过程。

```sh
python infer.py \
    --device "gpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```

## 5.3 推理配置参数

推理过程中，可支持配置的参数如下。

* `model_path_prefix`：必须，待推理模型路径前缀。
* `model_name_or_path`：选择预训练模型,可选"ernie-1.0-large-zh-cw","ernie-3.0-xbase-zh", "ernie-3.0-base-zh", "ernie-3.0-medium-zh", "ernie-3.0-micro-zh", "ernie-3.0-mini-zh", "ernie-3.0-nano-zh", "ernie-2.0-base-en", "ernie-2.0-large-en","ernie-m-base","ernie-m-large"；默认为"ernie-3.0-medium-zh",根据实际使用的预训练模型选择。
* `max_seq_length`：ERNIE/BERT模型使用的最大序列长度，最大不能超过512, 若出现显存不足，请适当调低这一参数；默认为128。
* `use_fp16`：选择是否开启FP16进行加速；默认为False。
* `batch_size`：批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为32。
* `device`: 选用什么设备进行训练，可选cpu、gpu。
* `device_id`: 选择GPU卡号；默认为0。
* `perf`：选择进行模型性能和精度评估；默认为False。
* `dataset_dir`：本地数据集地址，需包含data.txt, label.txt, test.txt/dev.txt(可选，如果启动模型性能和精度评估)；默认为None。
* `perf_dataset`：评估数据集，可选'dev'、'test'，选择在开发集或测试集评估模型；默认为"dev"。

在GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，在包括V100、T4、A10、A100、GTX 20系列和30系列显卡等设备上可以开启FP16进行加速，在CPU或者CUDA计算能力 (CUDA Compute Capability) 小于7.0时开启不会带来加速效果。可以使用如下命令开启ONNXRuntime的FP16进行推理加速：


```sh
python infer.py \
    --use_fp16 \
    --device "gpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```

输出内容示例如下所示。

<img width="1200" alt="image" src="https://user-images.githubusercontent.com/14270174/198861855-4cfe0a6f-04e0-46a0-8abc-09de5fb03a3a.png">


如果希望基于CPU部署，命令如下所示，将`device`参数修改为`cpu`即可。

请使用如下命令进行部署。

```sh
python infer.py \
    --device "cpu" \
    --model_path_prefix "../../export/float32" \
    --model_name_or_path "ernie-3.0-medium-zh" \
    --max_seq_length 128 \
    --batch_size 32 \
    --dataset_dir "../../data"
```
