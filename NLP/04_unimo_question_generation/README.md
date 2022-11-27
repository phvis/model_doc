# 概述

本文重点介绍如何利用**PaddleNLP**，完成基于UNIMO的问题生成任务。

问题生成（Question Generation），指的是给定一段上下文，自动生成一个流畅且符合上下文主题的问句。问题生成通常可以分为，无答案问题生成和有答案问题生成，这里只关注应用更广的有答案问题生成。

问题生成技术在教育、咨询、搜索、推荐等多个领域均有着巨大的应用价值。具体来说，问题生成可广泛应用于问答系统语料库构建，事实性问题生成，教育行业题库生成，对话提问，聊天机器人意图理解，对话式搜索意图提问，闲聊机器人主动提问等等场景。

## 文章目录结构
- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleNLP安装
- 2 数据准备
  - 2.1 数据格式
  - 2.2 DuReader_QG公开数据集
- 3 模型介绍
- 4 模型训练、验证与预测
  - 4.1 模型训练
  - 4.2 训练配置参数
  - 4.3 模型预测
- 5 模型部署
  - 5.1 模型导出
  - 5.2 导出配置参数
  - 5.3 模型推理
  - 5.4 推理配置参数
  - 5.5 FAQ

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

使用下面的方式，再安装运行本项目的所需要的依赖。

```sh
cd PaddleNLP/examples/question_generation/unimo-text
pip install -r requirements.txt
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

数据集标注以文本文件形式存储，每一行的信息以字典的形式存储，包含4个字段，分别是`context`, `answer`, `question` 与 `id`，示例如下所示。

```json
{"context": "第35集雪见缓缓张开眼睛，景天又惊又喜之际，长卿和紫萱的仙船驶至，见众人无恙，也十分高兴。众人登船，用尽合力把自身的真气和水分输给她。雪见终于醒过来了，但却一脸木然，全无反应。众人向常胤求助，却发现人世界竟没有雪见的身世纪录。长卿询问清微的身世，清微语带双关说一切上了天界便有答案。长卿驾驶仙船，众人决定立马动身，往天界而去。众人来到一荒山，长卿指出，魔界和天界相连。由魔界进入通过神魔之井，便可登天。众人至魔界入口，仿若一黑色的蝙蝠洞，但始终无法进入。后来花楹发现只要有翅膀便能飞入。于是景天等人打下许多乌鸦，模仿重楼的翅膀，制作数对翅膀状巨物。刚佩戴在身，便被吸入洞口。众人摔落在地，抬头发现魔界守卫。景天和众魔套交情，自称和魔尊重楼相熟，众魔不理，打了起来。", "answer": "第35集", "question": "仙剑奇侠传3第几集上天界", "id": 0}
{"context": "选择燃气热水器时，一定要关注这几个问题：1、出水稳定性要好，不能出现忽热忽冷的现象2、快速到达设定的需求水温3、操作要智能、方便4、安全性要好，要装有安全报警装置 市场上燃气热水器品牌众多，购买时还需多加对比和仔细鉴别。方太今年主打的磁化恒温热水器在使用体验方面做了全面升级：9秒速热，可快速进入洗浴模式；水温持久稳定，不会出现忽热忽冷的现象，并通过水量伺服技术将出水温度精确控制在±0.5℃，可满足家里宝贝敏感肌肤洗护需求；配备CO和CH4双气体报警装置更安全（市场上一般多为CO单气体报警）。另外，这款热水器还有智能WIFI互联功能，只需下载个手机APP即可用手机远程操作热水器，实现精准调节水温，满足家人多样化的洗浴需求。当然方太的磁化恒温系列主要的是增加磁化功能，可以有效吸附水中的铁锈、铁屑等微小杂质，防止细菌滋生，使沐浴水质更洁净，长期使用磁化水沐浴更利于身体健康。", "answer": "方太", "question": "燃气热水器哪个牌子好", "id": 1}
{"context": "迈克尔.乔丹在NBA打了15个赛季。他在84年进入nba，期间在1993年10月6日第一次退役改打棒球，95年3月18日重新回归，在99年1月13日第二次退役，后于2001年10月31日复出，在03年最终退役。迈克尔·乔丹（Michael Jordan），1963年2月17日生于纽约布鲁克林，美国著名篮球运动员，司职得分后卫，历史上最伟大的篮球运动员。1984年的NBA选秀大会，乔丹在首轮第3顺位被芝加哥公牛队选中。 1986-87赛季，乔丹场均得到37.1分，首次获得分王称号。1990-91赛季，乔丹连夺常规赛MVP和总决赛MVP称号，率领芝加哥公牛首次夺得NBA总冠军。 1997-98赛季，乔丹获得个人职业生涯第10个得分王，并率领公牛队第六次夺得总冠军。2009年9月11日，乔丹正式入选NBA名人堂。", "answer": "15个", "question": "乔丹打了多少个赛季", "id": 2}
```

最终数据集目录下面包含下面3个文件。其中`train.json`和`dev.json`分别表示训练与开发集合标注文件。

```text
|--- train.json
|--- dev.json
```

## 2.2 DuReader_QG公开数据集

本文我们将以DuReader_QG公开数据集为示例，演示问题生成任务的训练与推理过程。

[DuReader_QG数据集](https://www.luge.ai/#/luge/dataDetail?id=8)是一个中文问题生成数据集，我们使用该数据集作为应用案例进行实验。DuReader_QG中的数据主要由由上下文、问题、答案3个主要部分组成，其任务描述为给定上下文p和答案a，生成自然语言表述的问题q，且该问题符合段落和上下文的限制。

为了方便用户快速测试，PaddleNLP Dataset API内置了DuReader_QG数据集，一键即可完成数据集加载，示例代码如下：

```py
from paddlenlp.datasets import load_dataset
train_ds, dev_ds = load_dataset('dureader_qg', splits=('train', 'dev'))
```

针对该数据集中的每个答案与上下文，我们需要推断出它的问题内容，以字符串形式给出。

在许多情况下，我们需要使用本地数据集来训练我们的文本分类模型，本项目支持使用固定格式本地数据集文件进行训练。 使用本地文件，只需要在模型训练时指定`train_file` 为本地训练数据地址，`predict_file` 为本地测试数据地址即可。

# 3 模型介绍

预训练模型可分为单模态（single-modal）与多模态（multi-modal），前者使用大量的文本/图像进行训练（CV:VGG ResNet, NLP:BERT GPT RoBERTa），后者使用图文对进行训练(Cross-modal: ViLBERT VisualBERT UNITER)。然而，multi-modal模型只能利用少量的图文对进行训练，只能学到image-text的表达，不能泛化迁移到single-modal的任务。因此，本文提出了一个UNIfied_MOdal 预训练模型，该模型可以同时适应single-modal和multi-modal的下游任务。


为了使用海量文本和图像数据，同时解决模态缺失带来的模型效果下降问题，UNIMO提出了统一模态预训练框架，可以同时支持文本、图像和文本-图像对三种不同类型的数据输入，使用统一堆叠的Transformer模型，将文本和图像表示映射在统一表示空间中，下面是UNIMO统一模态预训练框架图：

<img width="450" alt="image" src="https://user-images.githubusercontent.com/14270174/203980990-6575a365-be95-49cf-acaa-dcfc0ecd52ca.png">

为了将文本和图像表示映射在统一表示空间中，UNIMO模型输入主要包括三部分内容:

[1]文本输入：UNIMO文本输入和BERT类似，通过字节对编码（BPE, Byte Pair Encoder）将文本转化成token embedding，开始和结束分别添加【CLS】和【SEP】标志，得到序列{h[CLS], hw1, ..., hwn, h[SEP]}；

[2]图像输入：将图片采用Faster R-CNN算法提取兴趣图像区域的特征，通过自注意力机制得到上下文相关的区域特征embedding表征序列{h[IMG], hv1, ..., hvt}；

[3]文本-图像对输入：和传统多模态学习模型类似，将文本-图像对数据(V,W)分别得到文本对应的textual embed和图像对应的visual embed，然后进行拼接得到序列{[IMG], v1, ..., vt, [CLS], w1, ..., wn, [SEP]}。


UNIMO中使用对比学习，来优化跨模态场景的训练任务。传统的基于负例的对比学习主要是通过batch内采样获取负例，这样只能学习到粗粒度的文本和图像相似特征，而UNIMO将对比学习泛化到了跨模态层面，通过文本改写和文本/图像检索来提升正负例的质量和数量，结构框图如下。


<img width="618" alt="image" src="https://user-images.githubusercontent.com/14270174/203981090-c0f3f870-52c8-4e44-9300-be97a448ea84.png">

UNIMO模型提出的跨模态对比学习(Cross-Modal Contrastive Learning，CMCL)核心思想是将含义相同的文本-图像对数据作为正例，含义不同的文本-图像对数据作为负例，通过构造相似实例和不相似实例获得一个表示学习模型，通过这个模型可以让相似的实例在投影的向量空间中尽可能的接近，不相似的实例尽可能的远离。UNIMO为了提升CMCL中的正负例的质量，主要使用了文本改写和文本/图像检索两种策略：

(1)文本改写

为了增加CMCL中正负例的质量，UNIMO将图片的描述从语句、短语和词三个粒度进行改写。从语句粒度来看，通过回译技术增加正例，将图片对应的描述翻译成多条语义一致语言表示形式略有不同的样本从而达到样本增强的目的。同时基于tfidf相似度检索算法得到字面词汇重复率高但是语义不同的样本来增加负例；从短语和词粒度来看，随机替换语句中的object、attribute、relation和对应的组合信息从而增加负例；

(2)文本/图像检索

为了进一步增加CMCL正负例的质量，UNIMO从海量的单模数据中检索相似文本或者图像，从而组成弱相关文本-图像对数据用于对比学习，通过这种方式可以增加大量的训练语料。



更多关于UNIMO的介绍，请参考:

- 论文：[UNIMO: Towards Unified-Modal Understanding and Generation via Cross-Modal Contrastive Learning](https://arxiv.org/pdf/2012.15409.pdf)


# 4 模型训练、验证与预测

## 4.1 模型训练

首先进入工作目录。

```sh
cd PaddleNLP/examples/question_generation/unimo-text
```

使用下面的命令可以完成单卡的模型训练

```sh
python train.py \
    --dataset_name=dureader_qg \
    --model_name_or_path="unimo-text-1.0" \
    --save_dir=./unimo/finetune/checkpoints \
    --output_path ./unimo/finetune/predict.txt \
    --logging_steps=100 \
    --save_steps=500 \
    --epochs=20 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --template=1 \
    --device=gpu
```

训练过程中输出日志如下所示。

```json
[2022-11-25 16:30:16,495] [    INFO] - tokenizer config file saved in /home/aistudio/.paddlenlp/models/unimo-text-1.0/tokenizer_config.json
[2022-11-25 16:30:16,496] [    INFO] - Special tokens file saved in /home/aistudio/.paddlenlp/models/unimo-text-1.0/special_tokens_map.json

Epoch 1/20
step 100 - loss: 6.4865 - ppl: 656.2149 - lr: 0.0000028 - 0.444s/step
step 200 - loss: 4.3089 - ppl: 74.3609 - lr: 0.0000055 - 0.438s/step
```

如果在CPU环境下训练，可以指定`device`参数为`cpu`进行训练。

```sh
python train.py \
    --dataset_name=dureader_qg \
    --model_name_or_path="unimo-text-1.0" \
    --save_dir=./unimo/finetune/checkpoints \
    --output_path ./unimo/finetune/predict.txt \
    --logging_steps=100 \
    --save_steps=500 \
    --epochs=20 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --template=1 \
    --device=cpu
```

如果在GPU环境中使用，可以指定`gpus`参数进行单卡/多卡训练。使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。如果设备只有一个GPU卡号默认为0，可使用`nvidia-smi`命令查看GPU使用情况:

```sh
export CUDA_VISIBLE_DEVICES=0,1
python -m paddle.distributed.launch --gpus "0,1" train.py \
    --dataset_name=dureader_qg \
    --model_name_or_path="unimo-text-1.0" \
    --save_dir=./unimo/finetune/checkpoints \
    --output_path ./unimo/finetune/predict.txt \
    --logging_steps=100 \
    --save_steps=500 \
    --epochs=20 \
    --batch_size=16 \
    --learning_rate=1e-5 \
    --warmup_propotion=0.02 \
    --weight_decay=0.01 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_train \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --num_return_sequences=1 \
    --template=1 \
    --device=gpu
```


## 4.2 训练配置参数

可支持配置的参数：

- `gpus` 指示了训练所用的GPU，使用多卡训练可以指定多个GPU卡号，例如 --gpus "0,1"。
- `dataset_name` 数据集名称，默认为`dureader_qg`。
- `train_file` 本地训练数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为None。
- `predict_file` 本地测试数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为None。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。如果使用PaddleNLP提供的预训练模型，可以选择下面其中之一。
   | 可选预训练模型        |
   |---------------------------------|
   | unimo-text-1.0      |
   | unimo-text-1.0-large |

   <!-- | T5-PEGASUS |
   | ernie-1.0 |
   | ernie-gen-base-en |
   | ernie-gen-large-en |
   | ernie-gen-large-en-430g | -->

- `save_dir` 表示模型的保存路径。
- `output_path` 表示预测结果的保存路径。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `seed` 表示随机数生成器的种子。
- `epochs` 表示训练轮数。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `weight_decay` 表示AdamW优化器中使用的weight_decay的系数。
- `warmup_propotion` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数占总步数的比例。
- `max_seq_len` 模型输入序列的最大长度。
- `max_target_len` 模型训练时标签的最大长度。
- `min_dec_len` 模型生成序列的最小长度。
- `max_dec_len` 模型生成序列的最大长度。
- `do_train` 是否进行训练。
- `do_predict` 是否进行预测，在验证集上会自动评估。
- `device` 表示使用的设备，从gpu和cpu中选择。
- `template` 表示使用的模版，从[0, 1, 2, 3, 4]中选择，0表示不选择模版，1表示使用默认模版。

程序运行时将会自动进行训练和验证，训练过程中会自动保存模型在指定的`save_dir`中。如：

```text
./unimo/finetune/checkpoints
├── model_1000
│   ├── model_config.json
│   ├── model_state.pdparams
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.txt
└── ...
```

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

**NOTE:** 如需恢复模型训练，`model_name_or_path`配置本地模型的目录地址即可。

微调的baseline模型在dureader_qg验证集上有如下结果(指标为BLEU-4)：

|       model_name        | DuReaderQG |
| :-----------------------------: | :-----------: |
|    unimo-text-1.0-dureader_qg-template1    | 41.08 |


## 4.3 模型预测

训练后的模型我们可以使用 评估脚本 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在GPU环境下使用，在CPU环境下修改参数配置为`--device "cpu"`:

```shell
export CUDA_VISIBLE_DEVICES=0
python -u predict.py \
    --dataset_name=dureader_qg \
    --model_name_or_path=your_model_path \
    --output_path=./predict.txt \
    --logging_steps=100 \
    --batch_size=16 \
    --max_seq_len=512 \
    --max_target_len=30 \
    --do_predict \
    --max_dec_len=20 \
    --min_dec_len=3 \
    --template=1 \
    --device=gpu
```

关键参数释义如下：
- `output_path` 表示预测输出结果保存的文件路径，默认为./predict.txt。
- `model_name_or_path` 指示了finetune使用的具体预训练模型，可以是PaddleNLP提供的预训练模型，或者是本地的微调好的预训练模型。如果使用本地的预训练模型，可以配置本地模型的目录地址，例如: ./checkpoints/model_xx/，目录中需包含paddle预训练模型model_state.pdparams。


输出打印示例：

```text
W1125 18:06:51.002157 20480 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W1125 18:06:51.006407 20480 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.

Pred begin...
step 100 - 1.450s/step
Generation cost time: 176.39131212234497
```

生成的问题也保存在`predict.txt`文件中。

# 5 模型部署

## 5.1 模型导出

使用动态图训练结束之后，可以通过静态图导出脚本实现基于`FasterTransformer`的高性能预测加速，并将动态图参数导出成静态图参数，静态图参数保存在output_path指定路径中。运行方式：

```sh
python export_model.py \
    --model_name_or_path ./checkpoint \
    --inference_model_dir ./export_checkpoint \
    --max_dec_len 50 \
    --use_fp16_decoding
```

## 5.2 导出配置参数

导出过程中，配置参数如下所示。

* `model_name_or_path`：动态图训练保存的参数路径；默认为"./checkpoint"。
* `inference_model_dir`：静态图图保存的参数路径；默认为"./export_checkpoint"。
* `max_dec_len`：最大输出长度。
* `use_fp16_decoding`:是否使用fp16解码进行预测。

程序运行时将会自动导出模型到指定的 `export_checkpoint` 中，保存模型文件结构如下所示：

```text
export_checkpoint/
├── unimo_text.pdiparams
├── unimo_text.pdiparams.info
└── unimo_text.pdmodel
```

## 5.3 模型推理

使用下面的命令完GPU推理过程。

```sh
python deploy/paddle_inference/inference.py \
    --inference_model_dir ./export_checkpoint \
    --model_name_or_path "unimo-text-1.0" \
    --predict_file ~/.paddlenlp/datasets/DuReaderQG/dev.json \
    --output_path "infer.txt" \
    --device gpu
```

输出结果如下。


```text
Infer begin...
step 100 - 0.495s/step

Save inference result into: output_path_name
inference cost time: 62.26314830780029
```

推理生成的问题字符串也同步保存在`infer.txt`文件中。

如果希望使用CPU进行推理，可以使用下面的命令。


```sh
python deploy/paddle_inference/inference.py \
  --inference_model_dir ./export_checkpoint \
  --model_name_or_path "unimo-text-1.0" \
  --predict_file ~/.paddlenlp/datasets/DuReaderQG/dev.json \
  --output_path "infer.txt" \
  --device cpu
```


## 5.4 推理配置参数

推理过程中，可配置参数如下所示。

* `inference_model_dir`：用于高性能推理的静态图模型参数路径，默认为"./export_checkpoint"。
* `model_name_or_path`：tokenizer对应模型或路径，默认为"unimo-text-1.0"。
* `dataset_name`：数据集名称，默认为`dureader_qg`。
* `predict_file`：本地预测数据地址，数据格式必须与`dataset_name`所指数据集格式相同，默认为None，当为None时默认加载`dataset_name`的dev集。
* `output_path`：表示预测结果的保存路径。
* `device`：推理时使用的设备，可选项["gpu"]，默认为"gpu"。
* `batch_size`：进行推理时的批大小，默认为16。
* `precision`：当使用TensorRT进行加速推理时，所使用的TensorRT精度，可选项["fp32", "fp16"]，默认为"fp32"。


## 5.5 FAQ

**Q: 推理时，报错，提示类型不匹配怎么办？**

**A: 可以将export_model.py中的`attention_mask`导出参数的类型，由"float64"修改为"float32"** 

====
 