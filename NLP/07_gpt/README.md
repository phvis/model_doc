# 概述

本文重点介绍如何利用**PaddleNLP**，完成基于GPT模型的训练过程，并得到语言生成模型。



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
- 4 模型训练与评估
  - 4.1 模型训练
  - 4.2 训练配置参数
  - 4.3 模型评估
  - 4.4 评估配置参数
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

此外，希望运行本项目内容的话，还需要安装以下的依赖包


```bash
pip install regex sentencepiece tqdm visualdl tool_helpers pybind11 lac zstandard
```


# 2 数据准备

[OpenWebTextCorpus](https://skylion007.github.io/OpenWebTextCorpus/)是一个开源的英文网页文本数据集，数据来源于Reddit，经过去重、清洗、提取，最终包含800多万个文档。
本示例采用EleutherAI清洗好的[OpenWebText2数据](https://openwebtext2.readthedocs.io/en/latest/index.html#download-plug-and-play-version)

下载以后通过以下命令解压：

```shell
wget https://mystic.the-eye.eu/public/AI/pile_preliminary_components/openwebtext2.jsonl.zst.tar
tar -xvf openwebtext2.json.zst.tar -C  /path/to/openwebtext
```

然后使用[preprocess](https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/gpt/../ernie-1.0/preprocess) 工具下的`create_pretraining_data.py`脚本进行数据集制作：
```
python -u  create_pretraining_data.py \
    --model_name gpt2-en \
    --tokenizer_name GPTTokenizer \
    --data_format JSON \
    --input_path /path/to/openwebtext/ \
    --append_eos \
    --output_prefix gpt_openwebtext  \
    --workers 40 \
    --log_interval 10000
```
处理时间约一个小时左右，就可以得到我们需要的`gpt_openwebtext_ids.npy`, `gpt_openwebtext_idx.npz`数据集文件。

为了方便用户运行测试本模型，本项目提供了处理好的300M的训练样本：
```shell
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_ids.npy
wget https://bj.bcebos.com/paddlenlp/models/transformers/gpt/data/gpt_en_dataset_300m_idx.npz
```

将所有预处理得到的文件统一放入一个文件夹中，以备训练使用：

```
mkdir data
mv gpt_en_dataset_300m_ids.npy ./data
mv gpt_en_dataset_300m_idx.npz ./data
```

# 3 模型介绍

Generative Pre-trained Transformer（GPT）系列是由OpenAI提出的非常强大的预训练语言模型，这一系列的模型可以在非常复杂的NLP任务中取得非常惊艳的效果，例如文章生成，代码生成，机器翻译，Q&A等，而完成这些任务并不需要有监督学习进行模型微调。而对于一个新的任务，GPT仅仅需要非常少的数据便可以理解这个任务的需求并达到接近或者超过state-of-the-art的方法。



[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) 是以[Transformer](https://arxiv.org/abs/1706.03762) 解码器为网络基本组件，使用自回归的方式在大规模无标注文本语料上进行预训练得到的语言生成模型。

GPT-2的目标旨在训练一个泛化能力更强的词向量模型，它并没有对GPT-1的网络进行过多的结构的创新与设计，只是使用了更多的网络参数和更大的数据集。下面我们对GPT-2展开详细的介绍。


GPT-2的学习目标是使用无监督的预训练模型做有监督的任务。因为文本数据的时序性，一个输出序列可以表示为一系列条件概率的乘积。而语言模型也可以表示为$P({output|input})$，对于一个有监督的任务，它可以建模为$P({output|input}, task)$。因此最终其实所有的任务都可以理解为分类任务，因此不需要针对每个子任务单独设计一个模型。

基于上面的思想，作者认为，当一个语言模型的容量足够大时，它就足以覆盖所有的有监督任务，也就是说所有的有监督学习都是无监督语言模型的一个子集。例如当模型训练完“Micheal Jordan is the best basketball player in the history”语料的语言模型之后，便也学会了(question：“who is the best basketball player in the history ?”，answer:“Micheal Jordan”)的Q&A任务。

基于上述思想，GPT-2的训练数据取自于Reddit上高赞的文章（保证语料的尽可能丰富），命名为WebText。数据集共有约800万篇文章，累计体积约40G。为了避免和测试集的冲突，WebText移除了涉及Wikipedia的文章。

模型参数如下。
- 使用字节对编码构建字典，字典的大小为50257；
- 滑动窗口的大小为$1024$；
- batchsize的大小为 512；
- Layer Normalization移动到了每一块的输入部分，在每个self-attention之后额外添加了一个Layer Normalization；
- 将残差层的初始化值用$1/\sqrt{N}$进行缩放，其中N是残差层的个数。

最终，在8个语言模型任务中，仅仅通过zero-shot学习，GPT-2就有7个超过了state-of-the-art的方法；此外，“LAMBADA”是测试模型捕捉长期依赖的能力的数据集，GPT-2将困惑度从99.8降到了8.6。

更多关于GPT全系列模型的解读与原始论文，请参考：

- [知乎解读：预训练语言模型之GPT-1，GPT-2和GPT-3](https://zhuanlan.zhihu.com/p/350017443)
- [GPT-1](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [GPT-3](https://arxiv.org/abs/2005.14165)



# 4 模型训练与评估

## 4.1 模型训练

使用下面的命令，训练GPU-2的英文模型。

```shell
cd PaddleNLP/model_zoo/gpt
CUDA_VISIBLE_DEVICES=0 python run_pretrain.py \
    --model_type gpt \
    --model_name_or_path gpt2-en \
    --input_dir "./data"\
    --output_dir "output"\
    --weight_decay 0.01\
    --grad_clip 1.0\
    --max_steps 500000\
    --save_steps 100000\
    --decay_steps 320000\
    --warmup_rate 0.01\
    --micro_batch_size 4\
    --device gpu
```

训练过程中的输出日志信息如下所示。

```json
[2022-11-25 23:31:05,459] [    INFO] - global step 1471, epoch: 0, batch: 1470, loss: 8.085629463, avg_reader_cost: 0.00024 sec, avg_batch_cost: 0.35901 sec, speed: 2.79 step/s, ips_total: 11409 tokens/s, ips: 11409 tokens/s, learning rate: 4.60000e-05
[2022-11-25 23:31:05,819] [    INFO] - global step 1472, epoch: 0, batch: 1471, loss: 7.663599491, avg_reader_cost: 0.00020 sec, avg_batch_cost: 0.35912 sec, speed: 2.78 step/s, ips_total: 11406 tokens/s, ips: 11406 tokens/s, learning rate: 4.60313e-05
[2022-11-25 23:31:06,180] [    INFO] - global step 1473, epoch: 0, batch: 1472, loss: 7.502598763, avg_reader_cost: 0.00023 sec, avg_batch_cost: 0.36036 sec, speed: 2.77 step/s, ips_total: 11366 tokens/s, ips: 11366 tokens/s, learning rate: 4.60625e-05
[2022-11-25 23:31:06,541] [    INFO] - global step 1474, epoch: 0, batch: 1473, loss: 7.493352890, avg_reader_cost: 0.00022 sec, avg_batch_cost: 0.36022 sec, speed: 2.78 step/s, ips_total: 11371 tokens/s, ips: 11371 tokens/s, learning rate: 4.60938e-05
[2022-11-25 23:31:06,899] [    INFO] - global step 1475, epoch: 0, batch: 1474, loss: 8.025818825, avg_reader_cost: 0.00019 sec, avg_batch_cost: 0.35756 sec, speed: 2.80 step/s, ips_total: 11455 tokens/s, ips: 11455 tokens/s, learning rate: 4.61250e-05
```

该模型训练过程中依赖了大量数据，并且训练迭代轮数很大，需要比较长的训练时间，因此建议使用多卡训练，使用方式如下，

```sh
cd PaddleNLP/model_zoo/gpt
python -u  -m paddle.distributed.fleet.launch \
    --gpus "0,1,2,3,4,5,6,7" \
    --log_dir "output/$task_name/log" run_pretrain_static.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --input_dir "./data" \
    --output_dir "output/$task_name" \
    --max_seq_len 1024 \
    --micro_batch_size 8 \
    --global_batch_size 32 \
    --sharding_degree 4\
    --mp_degree 2 \
    --dp_degree 1 \
    --pp_degree 1 \
    --use_sharding true \
    --use_amp true \
    --use_recompute true \
    --max_lr 0.00015 \
    --min_lr 0.00001 \
    --max_steps 500000 \
    --save_steps 100000 \
    --decay_steps 320000 \
    --weight_decay 0.01\
    --warmup_rate 0.01 \
    --grad_clip 1.0 \
    --logging_freq 1\
    --eval_freq 10000 \
    --device "gpu"
```

## 4.2 训练配置参数

训练过程中，可配置的参数如下所示。

- `model_name_or_path` 要训练的模型或者之前训练的checkpoint。
- `input_dir` 指定输入文件，可以使用目录，指定目录时将包括目录中的所有文件。
- `output_dir` 指定输出文件。
- `weight_decay` 权重衰减参数。
- `grad_clip` 梯度裁剪范围。
- `max_steps` 最大训练步数
- `save_steps` 保存模型间隔
- `mirco_batch_size` 训练的batch大小
- `device` 训练设备


训练过程中输出如下所示。


## 4.3 模型评估


我们提供了对[WikiText](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip)、[LAMBADA](https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl)两种数据集的评估脚本, 使用如下命令启动评估：

1. WikiText数据集评估
```bash
python run_eval.py --model_name gpt2-en \
    --eval_path ./wikitext-103/wiki.valid.tokens \
    --overlapping_eval 32 \
    --init_checkpoint_path ./output/model_1000/model_state.pdparams \
    --batch_size 8 \
    --device gpu
```

2. LAMBADA数据集评估

```bash
python run_eval.py --model_name gpt2-en \
    --eval_path ./lambada_test.jsonl \
    --cloze_eval \
    --init_checkpoint_path ./output/model_1000/model_state.pdparams \
    --batch_size 8 \
    --device gpu
```

输出内容如下所示。

```text
[2022-11-26 00:19:00,201] [   ERROR] - Using bos_token, but it is not set yet.
[2022-11-26 00:19:00,272] [    INFO] - tokenizer config file saved in /home/aistudio/.paddlenlp/models/gpt2-en/tokenizer_config.json
[2022-11-26 00:19:00,273] [    INFO] - Special tokens file saved in /home/aistudio/.paddlenlp/models/gpt2-en/special_tokens_map.json
[2022-11-26 00:19:06,083] [    INFO] - step 0, batch: 0, number correct: 0.000000, speed: 17.00 step/s
...
[2022-11-26 00:21:33,554] [    INFO] - step 600, batch: 600, number correct: 0.000000, speed: 4.05 step/s
[2022-11-26 00:21:44,210] [    INFO] -  validation results on ./lambada_test.jsonl | number correct: 0.0000E+00 | total examples: 5.1530E+03 | avg accuracy: 0.0000E+00
```


可以看出，由于训练时间较短，模型精度为0。这里repo中也提供了训练好的模型，我们不指定`init_checkpoint_path`参数，即可加载默认的预训练模型，命令如下。

```bash
python run_eval.py --model_name gpt2-en \
    --eval_path ./lambada_test.jsonl \
    --cloze_eval \
    --batch_size 8 \
    --device gpu
```

输出内容如下所示。


```text
[2022-11-26 00:25:22,324] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/gpt2-en/gpt-en-merges.txt
[2022-11-26 00:25:22,324] [   ERROR] - Using bos_token, but it is not set yet.
[2022-11-26 00:25:22,399] [    INFO] - tokenizer config file saved in /home/aistudio/.paddlenlp/models/gpt2-en/tokenizer_config.json
[2022-11-26 00:25:22,400] [    INFO] - Special tokens file saved in /home/aistudio/.paddlenlp/models/gpt2-en/special_tokens_map.json
[2022-11-26 00:25:28,439] [    INFO] - step 0, batch: 0, number correct: 4.000000, speed: 16.35 step/s
...
[2022-11-26 00:27:56,049] [    INFO] - step 600, batch: 600, number correct: 1561.000000, speed: 4.04 step/s
[2022-11-26 00:28:06,729] [    INFO] -  validation results on ./lambada_test.jsonl | number correct: 1.6780E+03 | total examples: 5.1530E+03 | avg accuracy: 3.2564E-01
```

可以看出，加载了repo中的预训练模型后，精度有明显提升。


## 4.4 评估配置参数

评估过程中，可支持配置的参数如下所示。

- `model_name` 使用的模型名称，如gpt2-en、gpt2-medium-en等。
- `eval_path` 数据集地址。
- `init_checkpoint_path` 模型参数地址。
- `batch_size` batch size大小。
- `device` 运行设备，cpu，gpu，xpu可选。
- `overlapping_eval` wikitext数据集参数。
- `cloze_eval` lambada数据参数，作为完型填空任务。

其中数据集WikiText采用的是PPL(perplexity)评估指标，LAMBADA采用的是ACC(accuracy)指标。

注：不设置`init_checkpoint_path` 参数时，可以评估默认预训练好的模型参数。


# 5 模型部署

## 5.1 模型导出

由于该模型训练时间很长，因此我们在这里直接使用repo提供的模型进行模型导出和推理。

使用下面的命令导出模型。

```sh
python export_model.py --model_type=gpt \
    --model_path=gpt2-en \
    --output_path=./infer_model/model
```


最终目录`infer_model`下的内容如下所示。`model.pdiparams`和`model.pdmodel`分别是保存得到的模型权重文件和模型结构文件。

```text
|----merges.txt
|----model.pdiparams
|----model.pdiparams.info
|----model.pdmodel
|----special_tokens_map.json
|----tokenizer_config.json
|----vocab.json
```


## 5.2 模型推理

使用下面的命令，可以完成模型的推理过程。


```sh
python deploy/python/inference.py --model_type gpt \
    --model_path ./infer_model/model
```
