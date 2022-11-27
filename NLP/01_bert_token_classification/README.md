# 概述

本文重点介绍如何利用**PaddleNLP**，完成基于bert的序列标注任务。

## 文章目录结构
- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleNLP安装
- 2 数据准备
  - 2.1 数据格式
  - 2.2 MSRA-NER公开数据集
  - 2.3 数据标注工具
- 3 模型介绍
- 4 模型训练、验证与预测
  - 4.1 模型训练
  - 4.2 训练配置参数
  - 4.3 模型评估
- 5 模型部署
  - 5.1 模型导出
  - 5.2 导出配置参数
  - 5.3 模型推理
  - 5.4 推理配置参数

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
pip install scikit-learn==1.0.2
```

## 1.3 推理相关依赖安装

模型转换与ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，Paddle2ONNX支持将Paddle静态图模型转化为ONNX模型格式，算子目前稳定支持导出ONNX Opset 7~15，更多细节可参考：[Paddle2ONNX](https://github.com/PaddlePaddle/Paddle2ONNX)。

如果基于GPU部署，请先确保机器已正确安装NVIDIA相关驱动和基础软件，确保CUDA >= 11.2，CuDNN >= 8.2，并使用以下命令安装所需依赖:

```sh
pip install onnxruntime-gpu onnx onnxconverter-common
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

## 2.1 数据标注

数据集标注以文本文件形式存储，每一行中，包含2个字段：文本内容以及标注结果，2个字段以`\t`分隔，而其中每个字段中的字符均以`\002`进行分割。示例格式如下。

```json
中^B共^B中^B央^B致^B中^B国^B致^B公^B党^B十^B一^B大^B的^B贺^B词^B各^B位^B代^B表^B、^B各^B位^B同^B志^B：^B在^B中^B国^B致^B公^B党^B第^B十^B一^B次^B全^B国^B代^B表^B大^B会^B隆^B重^B召^B开^B之^B际^B，^B中^B国^B共^B产^B党^B>中^B央^B委^B员^B会^B谨^B向^B大^B会^B表^B示^B热^B烈^B的^B祝^B贺^B，^B向^B致^B公^B党^B的^B同^B志^B们^B致^B以^B亲^B切^B的^B问^B候^B！    B-ORG^BI-ORG^BI-ORG^BI-ORG^BO^BB-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BB-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BO^BO^BO^BO^BO^BO^BO^BB-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BB-ORG^BI-ORG^BI-ORG^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO
在^B过^B去^B的^B五^B年^B中^B，^B致^B公^B党^B在^B邓^B小^B平^B理^B论^B指^B引^B下^B，^B遵^B循^B社^B会^B主^B义^B初^B级^B阶^B段^B的^B基^B本^B路^B线^B，^B努^B力^B实^B践^B致^B公^B党^B十^B大^B提^B出^B的^B发^B挥^B参^B政^B党^B>职^B能^B、^B加^B强^B自^B身^B建^B设^B的^B基^B本^B任^B务^B。    O^BO^BO^BO^BO^BO^BO^BO^BB-ORG^BI-ORG^BI-ORG^BO^BB-PER^BI-PER^BI-PER^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BB-ORG^BI-ORG^BI-ORG^BI-ORG^BI-ORG^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO^BO
```

最终数据集目录下面包含下面3个文件。其中`train.tsv`和`test.tsv`分别表示训练与测试文件标签。`label_map.json`

```text
|--- train.tsv
|--- test.tsv
|--- label_map.json
```

`label_map.json`为数据集中的标签与类别ID映射字典，示例如下。

```json
{
  "B-PER": 0,
  "I-PER": 1,
  "B-ORG": 2,
  "I-ORG": 3,
  "B-LOC": 4,
  "I-LOC": 5,
  "O": 6
}
```

本项目中，加载MSRA_NER数据集为BIO标注集：

* B-PER、I-PER代表人名首字、人名非首字。
* B-LOC、I-LOC代表地名首字、地名非首字。
* B-ORG、I-ORG代表组织机构名首字、组织机构名非首字。
* O代表该字不属于命名实体的一部分。

## 2.2 MSRA-NER公开数据集

本文我们将以MSRA-NER公开数据集为示例，演示序列标注任务的训练评估与模型推理过程。PaddleNLP也支持对该数据的自动加载

```py
import paddlenlp
train_ds, test_ds = paddlenlp.datasets.load_dataset('msra_ner', splits=["train", "test"])
```

## 2.3 数据标注工具

训练需要准备指定格式的本地数据集,如果没有已标注的数据集，可以参考[doccano](https://github.com/doccano/doccano)工具进行序列数据标注。

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

选择点击右上角的登录按钮并登录之后，选择Sequence Label任务，进入之后点击开始标注-数据集-上传数据，将未标注的文本行数据上传至网页中。

<img width="1427" alt="image" src="https://user-images.githubusercontent.com/14270174/202887587-88c44a65-73e0-41ea-bb60-d402b86d3f24.png">

接下来点击标签，并新增标签列表，这里以LOC和PER为例，新增结果如下所示。


<img width="1434" alt="image" src="https://user-images.githubusercontent.com/14270174/202888236-cb82f711-30c3-4240-9af4-bbd574aba3ab.png">


然后点击`数据集`，选择其中的某些文字，对其进行标注，示例如下所示。

<img width="1120" alt="image" src="https://user-images.githubusercontent.com/14270174/202888262-8e750b65-8c1f-4549-8c84-d3e773064c15.png">

所有数据标注完成之后，点击`操作-导出`

<img width="1313" alt="image" src="https://user-images.githubusercontent.com/14270174/202888479-7df0abac-2948-4f5e-89b5-43b3fbda9451.png">

选择jsonl格式，导出结果，如下所示。


<img width="1177" alt="image" src="https://user-images.githubusercontent.com/14270174/202888494-be840dbe-96a1-475f-a9bd-35133eb2529f.png">


最终保存的文件格式如下。

```txt
{"id": 14, "data": "我在人民广场", "label": [[2, 6, "LOC"], [0, 1, "PER"]]}
```

导出后，使用脚本`trans_data.py`，将导出的数据转换为用于训练的数据集格式，需要指定输入文件路径、输出文件路径以及标签文件路径（如果没有事先生成的话）。

```sh
python trans_data.py
```

转换后的脚本如下所示。

```txt
这^B位^B同^B志^B的^B家^B乡^B是^B安^B徽  O^BO^BB-PER^BI-PER^BO^BO^BO^BO^BB-LOC^BI-LOC
```

# 3 模型介绍

BERT(Bidirectional Encoder Representation from Transformers)是2018年10月由Google AI研究院提出的一种预训练模型，该模型在机器阅读理解顶级水平测试SQuAD1.1中表现出惊人的成绩: 全部两个衡量指标上全面超越人类，并且在11种不同NLP测试中创出SOTA表现，包括将GLUE基准推高至80.4% (绝对改进7.6%)，MultiNLI准确度达到86.7% (绝对改进5.6%)，成为NLP发展史上的里程碑式的模型成就。

BERT的网络架构使用的是《Attention is all you need》中提出的多层Transformer结构，如 图1 所示。其最大的特点是抛弃了传统的RNN和CNN，通过Attention机制将任意位置的两个单词的距离转换成1，有效的解决了NLP中棘手的长期依赖问题。Transformer的结构在NLP领域中已经得到了广泛应用。

BERT整体框架包含pre-train和fine-tune两个阶段。pre-train阶段模型是在无标注的标签数据上进行训练，fine-tune阶段，BERT模型首先是被pre-train模型参数初始化，然后所有的参数会用下游的有标注的数据进行训练。

<img width="1000" alt="image" src="https://ai-studio-static-online.cdn.bcebos.com/7b5e70561695477ea0c1b36f8ed6cbde577000b89d7748b99af4eeec1d1ab83a">


BERT是用了Transformer的encoder侧的网络，encoder中的Self-attention机制在编码一个token的时候同时利用了其上下文的token，其中‘同时利用上下文’即为双向的体现，而并非想Bi-LSTM那样把句子倒序输入一遍。


更多关于BERT的详细解读，请参考[BERT介绍¶](https://paddlepedia.readthedocs.io/en/latest/tutorials/pretrain_model/bert.html)，原始论文请参考：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)。

# 4 模型训练、验证与预测

## 4.1 模型训练

首先进入工作目录。

```sh
cd ry/model_docs/01_bert_token_classification
```

使用下面的命令可以完成单卡的模型训练

```sh
python run_token_cls.py \
  --task_name msra_ner \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --num_train_epochs 5
```

训练过程中输出如下所示。

```json
test: eval loss: 0.093072, precision: 0.534619, recall: 0.430576, f1: 0.476990
global step 5300, epoch: 0, batch: 5299, loss: 0.297391, speed: 2.98 step/s
test: eval loss: 0.286331, precision: 0.455952, recall: 0.430015, f1: 0.442604
global step 5400, epoch: 0, batch: 5399, loss: 0.158047, speed: 2.91 step/s
test: eval loss: 0.172951, precision: 0.571261, recall: 0.454528, f1: 0.506253
[2022-10-30 20:37:06,788] [    INFO] - Configuration saved in best_msra_ner_model/model_config.json
[2022-10-30 20:37:10,778] [    INFO] - tokenizer config file saved in best_msra_ner_model/tokenizer_config.json
[2022-10-30 20:37:10,778] [    INFO] - Special tokens file saved in best_msra_ner_model/special_tokens_map.json
global step 5500, epoch: 0, batch: 5499, loss: 0.338936, speed: 2.66 step/s
test: eval loss: 0.099727, precision: 0.488446, recall: 0.458832, f1: 0.473176
global step 5600, epoch: 0, batch: 5599, loss: 0.320394, speed: 3.02 step/s
test: eval loss: 0.187024, precision: 0.500308, recall: 0.455464, f1: 0.476834
global step 5700, epoch: 1, batch: 74, loss: 0.196108, speed: 3.16 step/s
test: eval loss: 0.053053, precision: 0.528410, recall: 0.447231, f1: 0.484443
global step 5800, epoch: 1, batch: 174, loss: 0.052263, speed: 3.10 step/s
test: eval loss: 0.105017, precision: 0.540921, recall: 0.452657, f1: 0.492869
global step 5900, epoch: 1, batch: 274, loss: 0.159725, speed: 3.10 step/s
test: eval loss: 0.232823, precision: 0.537400, recall: 0.466504, f1: 0.499449
global step 6000, epoch: 1, batch: 374, loss: 0.118922, speed: 3.16 step/s
test: eval loss: 0.104798, precision: 0.507723, recall: 0.473615, f1: 0.490076
global step 6100, epoch: 1, batch: 474, loss: 0.214611, speed: 3.17 step/s
```

如果在CPU环境下训练，可以指定`nproc_per_node`参数进行多核训练：

```sh
python -m paddle.distributed.launch --nproc_per_node 8 --backend "gloo" run_token_cls.py \
  --task_name msra_ner \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --num_train_epochs 5
```

如果在GPU环境中使用，可以指定`gpus`参数进行单卡/多卡训练。使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况:

```sh
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus "0,1" run_token_cls.py \
  --task_name msra_ner \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --num_train_epochs 5
```


## 4.2 训练配置参数

可支持配置的参数：

* `task_name`: 任务/数据集名称，默认为`msra_ner`，也可以指定为自己的数据集目录。
* `model_name_or_path`：模型名称，这里使用`bert-base-uncased`。
* `do_train`：是否进行模型训练。
* `num_train_epochs`: 迭代轮数，默认为3，这里建议设置得更大一些。
* `output_dir`: 模型保存的输出目录，默认为`best_msra_ner_model`
* `batch_size`: 模型训练/评估的batch size。
* `ckp`: 模型的权重路径，在评估的时候需要指定。


程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
best_msra_ner_model/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
└── vocab.txt
```

## 4.3 模型评估

训练后的模型我们可以使用 评估脚本 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`:

```sh
python run_token_cls.py \
  --task_name msra_ner  \
  --model_name_or_path bert-base-uncased \
  --do_eval \
  --ckp ./best_msra_ner_model/model_state.pdparams
```

输出打印示例：

```text
[2022-10-30 20:46:15,353] [    INFO] - Configuration saved in /home/aistudio/.paddlenlp/models/bert-base-uncased/model_config.json
W1030 20:46:15.355427 12614 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W1030 20:46:15.361291 12614 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
[2022-10-30 20:46:22,040] [ WARNING] - Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForTokenClassification: ['cls.predictions.transform.weight', 'cls.predictions.decoder_bias', 'cls.seq_relationship.bias', 'cls.predictions.decoder_weight', 'cls.seq_relationship.weight', 'cls.predictions.transform.bias', 'cls.predictions.layer_norm.bias', 'cls.predictions.layer_norm.weight']
- This IS expected if you are initializing BertForTokenClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForTokenClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[2022-10-30 20:46:22,040] [ WARNING] - Some weights of BertForTokenClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.weight', 'classifier.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
eval loss: 0.835900, precision: 0.559631, recall: 0.476796, f1: 0.514904
```

评估过程中的超参数与训练过程一样。

# 5 模型部署

## 5.1 模型导出

使用动态图训练结束之后，还可以将动态图参数导出成静态图参数，静态图模型将用于**后续的推理部署工作**。具体代码见 静态图导出脚本 ，静态图参数保存在`output_path`指定路径中。运行方式如下。

```sh
python export_model.py \
  --model_name bert-base-uncased \
  --ckp ./best_msra_ner_model/model_state.pdparams \
  --output_path export
```

## 5.2 导出配置参数

* `ckp`：动态图训练保存的参数路径；默认为"./best_msra_ner_model/model_state.pdparams"。
* `output_path`：静态图图保存的参数路径；默认为"./export"。

程序运行时将会自动导出模型到指定的 `output_path` 中，保存模型文件结构如下所示：

```text
export/
├── float32.pdiparams
├── float32.pdiparams.info
└── float32.pdmodel
```

## 5.3 模型推理

模型ONNXRuntime预测部署依赖Paddle2ONNX和ONNXRuntime，推理前首先需要参考1.3章节安装依赖。

使用下面的命令完CPU推理过程。

```sh
python infer_cpu.py \
  --task_name token_cls  \
  --model_path ../../export  \
  --model_name_or_path bert-base-uncased  \
  --text 北京欢迎你  \
  --precision_mode fp16
```

输出结果如下。

```json
input data: 北京欢迎你
The model detects all entities:
entity: 京   label: ORG   pos: [1, 1]
entity: 欢迎你   label:    pos: [2, 6]
```

## 5.4 推理配置参数

可支持配置的参数：

* `task_name`: 任务名称，这里使用`token_cls`即可
* `model_name_or_path`: 模型名称，这里使用`bert-base-uncased`
* `model_path`: 导出的inference模型路径
* `precision_mode`: 预测的精度，支持fp32、fp16、int8。

可以看出，模型预测结果不够准确，如果希望使用精度更高的模型，建议使用ERNIE-3.0系列。

如果希望使用GPU进行推理，可以使用下面的命令推理。

```sh
python infer_gpu.py \
  --task_name token_cls  \
  --model_path ../../export  \
  --model_name_or_path bert-base-uncased  \
  --text 北京欢迎你  \
  --precision_mode fp16
```


在GPU设备的CUDA计算能力 (CUDA Compute Capability) 大于7.0，在包括V100、T4、A10、A100、GTX 20系列和30系列显卡等设备上可以开启FP16进行加速，在CPU或者CUDA计算能力 (CUDA Compute Capability) 小于7.0时开启不会带来加速效果。可以使用如下命令开启ONNXRuntime的FP16进行推理加速：


```sh
python infer_gpu.py \
  --task_name token_cls  \
  --model_path ../../export  \
  --model_name_or_path bert-base-uncased  \
  --text 北京欢迎你  \
  --precision_mode fp16
```
