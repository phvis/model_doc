# 概述

本文重点介绍如何利用**PaddleNLP**，完成基于UIE的实体抽取任务。



## 文章目录结构
- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleNLP安装
- 2 数据准备
  - 2.1 数据格式
  - 2.2 数据标注工具
- 3 模型介绍
- 4 模型快速使用、微调与预测
  - 4.1 快速使用
  - 4.2 模型微调
  - 4.3 微调配置参数
  - 4.4 模型评估
  - 4.5 评估配置参数
- 5 模型部署
  - 5.1 模型导出
  - 5.2 模型推理

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


UIE的部署分为 CPU 和 GPU 两种情况，请根据你的部署环境安装对应的依赖。

- CPU端

CPU端的部署请使用如下命令安装所需依赖：

```shell
cd PaddleNLP/model_zoo/uie
pip install -r deploy/python/requirements_cpu.txt
```

- GPU端

为了在 GPU 上获得最佳的推理性能和稳定性，请先确保机器已正确安装 NVIDIA 相关驱动和基础软件，确保 **CUDA >= 11.2，cuDNN >= 8.1.1**，并使用以下命令安装所需依赖

```shell
cd PaddleNLP/model_zoo/uie
pip install -r deploy/python/requirements_gpu.txt
```

如果有模型推理加速、内存显存占用优化的需求，并且 GPU 设备的 CUDA 计算能力 (CUDA Compute Capability) 大于等于 7.0，例如 V100、T4、A10、A100/GA100、Jetson AGX Xavier 等显卡，推荐使用半精度（FP16）部署。直接使用微调后导出的 FP32 模型，运行时设置 `--use_fp16` 即可。

如果 GPU 设备的 CUDA 计算能力较低，低于 7.0，只支持 FP32 部署，微调后导出模型直接部署即可。

更多关于 CUDA Compute Capability 和精度支持情况请参考 NVIDIA 文档：[GPU 硬件与支持精度对照表](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-840-ea/support-matrix/index.html#hardware-precision-matrix)


# 2 数据准备

## 2.1 数据格式

基于UIE的实体抽取任务数据集以文本形式存储，包含训练集与开发集合，格式相同。每行文本存储一个字典(`content`)，包含文字内容(`prompt`)，关键字段以及提取得到的信息(`result_list`)。

数据集文件格式如下所示。

```text
data
|----train.txt
|----dev.txt
```

示例文本内容如下所示。

```text
{"content": "6月10日加班打车回家25元", "result_list": [{"text": "6月10日", "start": 0, "end": 5}], "prompt": "时间"}
{"content": "6月10日加班打车回家25元", "result_list": [{"text": "家", "start": 10, "end": 11}], "prompt": "目的地"}
{"content": "6月10日加班打车回家25元", "result_list": [{"text": "25", "start": 11, "end": 13}], "prompt": "费用"}
```

下面在`2.2章节`中详细介绍基于doccano，生成用于模型微调的数据格式。


## 2.2 数据标注工具

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

推荐使用数据标注平台[doccano](https://github.com/doccano/doccano) 进行数据标注，本示例也打通了从标注到训练的通道，即doccano导出数据后可通过[doccano.py](./doccano.py)脚本轻松将数据转换为输入模型时需要的形式，实现无缝衔接。标注方法的详细介绍请参考[doccano数据标注指南](doccano.md)。

原始数据示例：

```text
深大到双龙28块钱4月24号交通费
```

抽取的目标(schema)为：

```python
schema = ['出发地', '目的地', '费用', '时间']
```

标注步骤如下：

- 在doccano平台上，创建一个类型为``序列标注``的标注项目。
- 定义实体标签类别，上例中需要定义的实体标签有``出发地``、``目的地``、``费用``和``时间``。
- 使用以上定义的标签开始标注数据，下面展示了一个doccano标注示例：

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167336891-afef1ad5-8777-456d-805b-9c65d9014b80.png height=100 hspace='10'/>
</div>

- 标注完成后，在doccano平台上导出文件，并将其重命名为``doccano_ext.json``后，放入``./data``目录下。

- 这里我们提供预先标注好的文件[doccano_ext.json](https://bj.bcebos.com/paddlenlp/datasets/uie/doccano_ext.json)，可直接下载并放入`./data`目录。执行以下脚本进行数据转换，执行后会在`./data`目录下生成训练/验证/测试集文件。

```shell
python doccano.py \
    --doccano_file ./data/doccano_ext.json \
    --task_type ext \
    --save_dir ./data \
    --splits 0.8 0.2 0 \
    --schema_lang ch
```


可配置参数说明：

- ``doccano_file``: 从doccano导出的数据标注文件。
- ``save_dir``: 训练数据的保存目录，默认存储在``data``目录下。
- ``negative_ratio``: 最大负例比例，该参数只对抽取类型任务有效，适当构造负例可提升模型效果。负例数量和实际的标签数量有关，最大负例数量 = negative_ratio * 正例数量。该参数只对训练集有效，默认为5。为了保证评估指标的准确性，验证集和测试集默认构造全负例。
- ``splits``: 划分数据集时训练集、验证集所占的比例。默认为[0.8, 0.1, 0.1]表示按照``8:1:1``的比例将数据划分为训练集、验证集和测试集。
- ``task_type``: 选择任务类型，可选有抽取和分类两种类型的任务。
- ``options``: 指定分类任务的类别标签，该参数只对分类类型任务有效。默认为["正向", "负向"]。
- ``prompt_prefix``: 声明分类任务的prompt前缀信息，该参数只对分类类型任务有效。默认为"情感倾向"。
- ``is_shuffle``: 是否对数据集进行随机打散，默认为True。
- ``seed``: 随机种子，默认为1000.
- ``separator``: 实体类别/评价维度与分类标签的分隔符，该参数只对实体/评价维度级分类任务有效。默认为"##"。
- ``schema_lang``: 选择schema的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

备注：
- 默认情况下 [doccano.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/doccano.py) 脚本会按照比例将数据划分为 train/dev/test 数据集
- 每次执行 [doccano.py](https://github.com/PaddlePaddle/PaddleNLP/blob/develop/model_zoo/uie/doccano.py) 脚本，将会覆盖已有的同名数据文件
- 在模型训练阶段我们推荐构造一些负例以提升模型效果，在数据转换阶段我们内置了这一功能。可通过`negative_ratio`控制自动构造的负样本比例；负样本数量 = negative_ratio * 正样本数量。
- 对于从doccano导出的文件，默认文件中的每条数据都是经过人工正确标注的。

更多**不同类型任务（关系抽取、事件抽取、评价观点抽取等）的标注规则及参数说明**，请参考[doccano数据标注指南](doccano.md)。

此外，也可以通过数据标注平台 [Label Studio](https://labelstud.io/) 进行数据标注。本示例提供了 [labelstudio2doccano.py](./labelstudio2doccano.py) 脚本，将 label studio 导出的 JSON 数据文件格式转换成 doccano 导出的数据文件格式，后续的数据转换与模型微调等操作不变。

```shell
python labelstudio2doccano.py --labelstudio_file label-studio.json
```

可配置参数说明：

- ``labelstudio_file``: label studio 的导出文件路径（仅支持 JSON 格式）。
- ``doccano_file``: doccano 格式的数据文件保存路径，默认为 "doccano_ext.jsonl"。
- ``task_type``: 任务类型，可选有抽取（"ext"）和分类（"cls"）两种类型的任务，默认为 "ext"。

# 3 模型介绍

Yaojie Lu等人在ACL-2022中提出了通用信息抽取统一框架UIE。该框架实现了实体抽取、关系抽取、事件抽取、情感分析等任务的统一建模，并使得不同任务间具备良好的迁移和泛化能力。为了方便大家使用UIE的强大能力，PaddleNLP借鉴该论文的方法，基于ERNIE 3.0知识增强预训练模型，训练并开源了首个中文通用信息抽取模型UIE。该模型可以支持不限定行业领域和抽取目标的关键信息抽取，实现零样本快速冷启动，并具备优秀的小样本微调能力，快速适配特定的抽取目标。

<div align="center">
    <img src=https://user-images.githubusercontent.com/40840292/167236006-66ed845d-21b8-4647-908b-e1c6e7613eb1.png height=400 hspace='10'/>
</div>

UIE相比于之前的信息抽取方法，主要优势如下。

- **使用简单**：用户可以使用自然语言自定义抽取目标，无需训练即可统一抽取输入文本中的对应信息。**实现开箱即用，并满足各类信息抽取需求**。

- **降本增效**：以往的信息抽取技术需要大量标注数据才能保证信息抽取的效果，为了提高开发过程中的开发效率，减少不必要的重复工作时间，开放域信息抽取可以实现零样本（zero-shot）或者少样本（few-shot）抽取，**大幅度降低标注数据依赖，在降低成本的同时，还提升了效果**。

- **效果领先**：开放域信息抽取在多种场景，多种任务上，均有不俗的表现。

更多关于UIE的模型介绍，请参考[Universal Information Extraction](https://arxiv.org/pdf/2203.12277.pdf)。

# 4 模型快速使用、微调与预测

## 4.1 模型快速使用

命名实体识别（Named Entity Recognition，简称NER），是指识别文本中具有特定意义的实体。在开放域信息抽取中，抽取的类别没有限制，用户可以自己定义。
- 例如抽取的目标实体类型是"时间"、"选手"和"赛事名称", schema构造如下：
```text
['时间', '选手', '赛事名称']
```

调用示例：

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> schema = ['时间', '选手', '赛事名称'] # Define the schema for entity extraction
>>> ie = Taskflow('information_extraction', schema=schema)
>>> pprint(ie("2月8日上午北京冬奥会自由式滑雪女子大跳台决赛中中国选手谷爱凌以188.25分获得金牌！")) # Better print results using pprint
[{'时间': [{'end': 6,
          'probability': 0.9857378532924486,
          'start': 0,
          'text': '2月8日上午'}],
  '赛事名称': [{'end': 23,
            'probability': 0.8503089953268272,
            'start': 6,
            'text': '北京冬奥会自由式滑雪女子大跳台决赛'}],
  '选手': [{'end': 31,
          'probability': 0.8981548639781138,
          'start': 28,
          'text': '谷爱凌'}]}]
```


- 例如抽取的目标实体类型是"肿瘤的大小"、"肿瘤的个数"、"肝癌级别"和"脉管内癌栓分级", schema构造如下：

```text
['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
```

在上例中我们已经实例化了一个`Taskflow`对象，这里可以通过`set_schema`方法重置抽取目标。

调用示例：

```python
>>> schema = ['肿瘤的大小', '肿瘤的个数', '肝癌级别', '脉管内癌栓分级']
>>> ie.set_schema(schema)
>>> pprint(ie("（右肝肿瘤）肝细胞性肝癌（II-III级，梁索型和假腺管型），肿瘤包膜不完整，紧邻肝被膜，侵及周围肝组织，未见脉管内癌栓（MVI分级：M0级）及卫星子灶形成。（肿物1个，大小4.2×4.0×2.8cm）。"))
[{'肝癌级别': [{'end': 20,
            'probability': 0.9243267447402701,
            'start': 13,
            'text': 'II-III级'}],
  '肿瘤的个数': [{'end': 84,
            'probability': 0.7538413804059623,
            'start': 82,
            'text': '1个'}],
  '肿瘤的大小': [{'end': 100,
            'probability': 0.8341128043459491,
            'start': 87,
            'text': '4.2×4.0×2.8cm'}],
  '脉管内癌栓分级': [{'end': 70,
              'probability': 0.9083292325934664,
              'start': 67,
              'text': 'M0级'}]}]
```

- 例如抽取的目标实体类型是"person"和"organization"，schema构造如下：

```text
['person', 'organization']
```

英文模型调用示例：

```python
>>> from pprint import pprint
>>> from paddlenlp import Taskflow
>>> schema = ['Person', 'Organization']
>>> ie_en = Taskflow('information_extraction', schema=schema, model='uie-base-en')
>>> pprint(ie_en('In 1997, Steve was excited to become the CEO of Apple.'))
[{'Organization': [{'end': 53,
                    'probability': 0.9985840259877357,
                    'start': 48,
                    'text': 'Apple'}],
  'Person': [{'end': 14,
              'probability': 0.999631971804547,
              'start': 9,
              'text': 'Steve'}]}]
```


## 4.2 模型微调

推荐使用 [Trainer API ](../../docs/trainer.md) 对模型进行微调。只需输入模型、数据集等就可以使用 Trainer API 高效快速地进行预训练、微调和模型压缩等任务，可以一键启动多卡训练、混合精度训练、梯度累积、断点重启、日志显示等功能，Trainer API 还针对训练过程的通用训练配置做了封装，比如：优化器、学习率调度等。

使用下面的命令，使用 `uie-base` 作为预训练模型进行模型微调，将微调后的模型保存至`$finetuned_model`：

```sh
```shell
# 进入工作目录
cd model_zoo/uie/
export finetuned_model=./checkpoint/best_model

python finetune.py  \
    --device gpu \
    --logging_steps 10 \
    --save_steps 100 \
    --eval_steps 100 \
    --seed 42 \
    --model_name_or_path uie-base \
    --output_dir $finetuned_model \
    --train_path data/train.txt \
    --dev_path data/dev.txt  \
    --max_seq_length 512  \
    --per_device_eval_batch_size 16 \
    --per_device_train_batch_size  16 \
    --num_train_epochs 100 \
    --learning_rate 1e-5 \
    --label_names 'start_positions' 'end_positions' \
    --do_train \
    --do_eval \
    --do_export \
    --export_model_dir $finetuned_model \
    --overwrite_output_dir \
    --disable_tqdm True \
    --metric_for_best_model eval_f1 \
    --load_best_model_at_end  True \
    --save_total_limit 1
```



训练过程中输出如下所示。

```json
W1125 20:51:47.950181 12349 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
[2022-11-25 20:51:55,559] [    INFO] - global step 10, epoch: 1, loss: 0.00290, speed: 1.86 step/s
[2022-11-25 20:52:00,186] [    INFO] - global step 20, epoch: 2, loss: 0.00232, speed: 2.16 step/s
[2022-11-25 20:52:04,843] [    INFO] - global step 30, epoch: 3, loss: 0.00194, speed: 2.15 step/s
[2022-11-25 20:52:09,453] [    INFO] - global step 40, epoch: 4, loss: 0.00163, speed: 2.17 step/s
[2022-11-25 20:52:14,066] [    INFO] - global step 50, epoch: 5, loss: 0.00138, speed: 2.17 step/s
[2022-11-25 20:52:18,693] [    INFO] - global step 60, epoch: 6, loss: 0.00120, speed: 2.16 step/s
[2022-11-25 20:52:23,297] [    INFO] - global step 70, epoch: 7, loss: 0.00106, speed: 2.17 step/s
[2022-11-25 20:52:27,925] [    INFO] - global step 80, epoch: 8, loss: 0.00095, speed: 2.16 step/s
[2022-11-25 20:52:32,539] [    INFO] - global step 90, epoch: 9, loss: 0.00086, speed: 2.17 step/s
[2022-11-25 20:52:37,161] [    INFO] - global step 100, epoch: 10, loss: 0.00078, speed: 2.16 step/s
[2022-11-25 20:52:38,764] [    INFO] - Evaluation precision: 1.00000, recall: 1.00000, F1: 1.00000
[2022-11-25 20:52:38,764] [    INFO] - best F1 performence has been updated: 0.00000 --> 1.00000
```

由于该模型所需要的数据量很少，在使用GPU的情况下，很短的时间内就可以完成模型微调的过程。


## 4.3 微调配置参数

可支持配置的参数如下所示。

* `model_name_or_path`：必须，进行 few shot 训练使用的预训练模型。可选择的有 "uie-base"、 "uie-medium", "uie-mini", "uie-micro", "uie-nano", "uie-m-base", "uie-m-large"。
* `multilingual`：是否是跨语言模型，用 "uie-m-base", "uie-m-large" 等模型进微调得到的模型也是多语言模型，需要设置为 True；默认为 False。
* `output_dir`：必须，模型训练或压缩后保存的模型目录；默认为 `None` 。
* `device`: 训练设备，可选择 'cpu'、'gpu' 其中的一种；默认为 GPU 训练。
* `per_device_train_batch_size`：训练集训练过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
* `per_device_eval_batch_size`：开发集评测过程批处理大小，请结合显存情况进行调整，若出现显存不足，请适当调低这一参数；默认为 32。
* `learning_rate`：训练最大学习率，UIE 推荐设置为 1e-5；默认值为3e-5。
* `num_train_epochs`: 训练轮次，使用早停法时可以选择 100；默认为10。
* `logging_steps`: 训练过程中日志打印的间隔 steps 数，默认100。
* `save_steps`: 训练过程中保存模型 checkpoint 的间隔 steps 数，默认100。
* `seed`：全局随机种子，默认为 42。
* `weight_decay`：除了所有 bias 和 LayerNorm 权重之外，应用于所有层的权重衰减数值。可选；默认为 0.0；
* `do_train`:是否进行微调训练，设置该参数表示进行微调训练，默认不设置。
* `do_eval`:是否进行评估，设置该参数表示进行评估。


程序运行时将会自动进行训练，评估。同时训练过程中会自动保存开发集上最佳模型在指定的 `save_dir` 中，保存模型文件结构如下所示：

```text
model_best/
├── model_config.json
├── model_state.pdparams
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

## 4.4 模型评估

训练后的模型我们可以使用 评估脚本 对每个类别分别进行评估，并输出预测错误样本（bad case），默认在GPU环境下使用。

```sh
python evaluate.py  \
  --model_path ./checkpoint/model_best/  \
  --test_path ./data/dev.txt   \
  --batch_size 16  \
  --max_seq_len 512
```

输出打印示例：

```text
2022-11-25 20:59:59,121] [    INFO] - We are using <class 'paddlenlp.transformers.ernie.tokenizer.ErnieTokenizer'> to load './checkpoint/model_best/model_best/'.
W1125 20:59:59.148970 13772 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W1125 20:59:59.152618 13772 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
[2022-11-25 21:00:03,078] [    INFO] - -----------------------------
[2022-11-25 21:00:03,079] [    INFO] - Class Name: all_classes
[2022-11-25 21:00:03,079] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
```

由于数据集比较简单，这里F1达到了100%。

如果希望打印每个类别的F1信息，可以使用下面的，命令。

```sh
python evaluate.py \
    --model_path ./checkpoint/model_best \
    --test_path ./data/dev.txt \
    --debug
```

输出信息如下所示。


```text
W1125 21:03:49.595613 14367 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W1125 21:03:49.599402 14367 gpu_resources.cc:91] device: 0, cuDNN Version: 8.2.
[2022-11-25 21:03:52,989] [    INFO] - -----------------------------
[2022-11-25 21:03:52,989] [    INFO] - Class Name: 时间
[2022-11-25 21:03:52,989] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-25 21:03:53,116] [    INFO] - -----------------------------
[2022-11-25 21:03:53,116] [    INFO] - Class Name: 费用
[2022-11-25 21:03:53,116] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-25 21:03:53,185] [    INFO] - -----------------------------
[2022-11-25 21:03:53,186] [    INFO] - Class Name: 出发地
[2022-11-25 21:03:53,186] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
[2022-11-25 21:03:53,219] [    INFO] - -----------------------------
[2022-11-25 21:03:53,219] [    INFO] - Class Name: X的地
[2022-11-25 21:03:53,219] [    INFO] - Evaluation Precision: 1.00000 | Recall: 1.00000 | F1: 1.00000
```

## 4.5 评估配置参数


模型评估过程中，配置参数说明如下。

- `model_path`: 进行评估的模型文件夹路径，路径下需包含模型权重文件`model_state.pdparams`及配置文件`model_config.json`。
- `test_path`: 进行评估的测试集文件。
- `batch_size`: 批处理大小，请结合机器情况进行调整，默认为16。
- `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为512。
- `debug`: 是否开启debug模式对每个正例类别分别进行评估，该模式仅用于模型调试，默认关闭。
- `multilingual`: 是否是跨语言模型，默认关闭。
- `schema_lang`: 选择schema的语言，可选有`ch`和`en`。默认为`ch`，英文数据集请选择`en`。

# 5 模型部署

## 5.1 模型导出

在`4.1章节`中，模型微调时，已经自动进行了静态图的导出，保存路径`${finetuned_model}` 下应该有 `*.pdmodel`、`*.pdiparams` 模型文件可用于推理。


## 5.2 模型推理

- CPU端推理样例

    在CPU端，请使用如下命令进行部署

    ```shell
    python deploy/python/infer_cpu.py --model_path_prefix ${finetuned_model}/model
    ```

    可配置参数说明：

    - `model_path_prefix`: 用于推理的Paddle模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/model.pdiparams`，则传入`./export/model`。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率 0~1 之间，返回结果去掉小于这个阈值的结果，默认为 0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为 512。
    - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为 4。

  - GPU端推理样例

    在GPU端，请使用如下命令进行部署

    ```shell
    python deploy/python/infer_gpu.py --model_path_prefix export/model --use_fp16 --device_id 0
    ```

    可配置参数说明：

    - `model_path_prefix`: 用于推理的 Paddle 模型文件路径，需加上文件前缀名称。例如模型文件路径为`./export/model.pdiparams`，则传入`./export/model`。
    - `use_fp16`: FP32 模型是否使用 FP16 进行加速，使用 FP32、INT8 推理时不需要设置，默认关闭。
    - `position_prob`：模型对于span的起始位置/终止位置的结果概率0~1之间，返回结果去掉小于这个阈值的结果，默认为 0.5，span 的最终概率输出为起始位置概率和终止位置概率的乘积。
    - `max_seq_len`: 文本最大切分长度，输入超过最大长度时会对输入文本进行自动切分，默认为 512。
    - `batch_size`: 批处理大小，请结合机器情况进行调整，默认为 4。
    - `device_id`: GPU 设备 ID，默认为 0。


