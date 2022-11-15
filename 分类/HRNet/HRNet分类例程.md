# 概述

本文重点介绍如何利用飞桨图像分类套件**PaddleClas**在flowers102 数据集上，使用当前PaddleClas中包含的HRNet模型进行详细讲解。
HRNet模型是经典的高分辨率分类模型，模型结构如下。

![image-20221103190029574](docs.assets/model.png)

HRNet模型更详细的原理介绍请参考[官方论文](https://arxiv.org/abs/1904.04514)。

## 文章目录结构

- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleClas安装
    - 1.2.1 下载PaddleClas代码
    - 1.2.2 安装依赖项目
    - 1.2.3 验证安装是否成功
- 2 数据准备
  - 2.1 数据标注
    - 2.1.1 精灵标注安装
    - 2.1.2 精灵标注的使用
  - 2.2 数据格式转化
  - 2.3 数据划分
- 3 模型训练、验证与预测
  - 3.1 模型训练参数说明
    - 3.1.1 训练前数据准备
    - 3.1.2 开始训练
    - 3.1.3 训练参数解释
    - 3.1.4 恢复训练
    - 3.1.5 训练可视化
  - 3.2 模型验证参数说明
    - 3.2.1 评估操作
    - 3.2.2 评估方式说明
  - 3.3 模型预测
    - 3.3.1 输出说明
- 4 配置文件的说明
  - 4.1 整体配置文件格式综述
  - 4.2 数据路径与数据预处理说明
  - 4.3 模型说明
  - 4.4 优化器和损失函数说明
  - 4.5 其他参数说明

# 1 环境安装

## 1.1 PaddlePaddle安装

### 1.1.1 安装对应版本PaddlePaddle

根据系统和设备的cuda环境，选择对应的安装包，这里默认使用pip在linux设备上进行安装。

![176497642-0abf3de1-86d5-43af-afe8-f97db46b7fd9](docs.assets/176497642-0abf3de1-86d5-43af-afe8-f97db46b7fd9.png)

在终端执行

```bash
pip install paddlepaddle-gpu==2.3.0.post110 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
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

## 1.2 PaddleClas安装

### 1.2.1 下载PaddleClas代码

用户可以通过使用github或者gitee的方式进行下载，我们当前版本为PaddleClas的release v2.5版本。后续在使用时，需要对应版本进行下载。

![image-20221102120803146](docs.assets/image-20221102120803146.png)



``` bash
# github下载
git clone https://github.com/PaddlePaddle/PaddleClas.git
# gitee下载
git clone https://gitee.com/paddlepaddle/PaddleClas.git
```

### 1.2.2 安装依赖项目

* 方式一：
  通过直接pip install 安装，可以最高效率的安装依赖

``` bash
pip install paddleclas
```

* 方式二：
  下载PaddleClas代码后，进入PaddleClas代码文件夹目录下面

``` bash
cd PaddleClas
pip install -r requirements.txt
```

### 1.2.3 验证安装是否成功

在PaddleClas目录下执行如下命令，会进行简单的单卡训练和单卡预测。

等待几分钟查看执行输出的log，若没有报错，则验证安装成功，可以不用再继续运行该测试脚本。

``` bash
sh tests/test.sh
```

# 2 数据准备

## 2.1 数据标注

无论是图像分类，语义分割，全景分割，还是实例分割，我们都需要充足的训练数据。如果你想使用没有标注的原始数据集做分类任务，你必须先为原始图像作出标注。

### 2.1.1 精灵标注安装



用户在采集完用于训练、评估和预测的图片之后，需使用数据标注工具[精灵标注](http://www.jinglingbiaozhu.com)完成数据标注。

精灵标注支持在Windows/macOS/Linux三个系统上使用，且三个系统下的标注格式基本一致。具体的安装流程请参见[官方安装指南](http://www.jinglingbiaozhu.com)。

### 2.1.2 精灵标注的使用

安装好精灵标注后，双击打开软件会出现精灵标注的交互界面，选择新建即可选择标注任务，随后展开标注。

![image-20221102123008695](docs.assets/image-20221102123008695.png)

然后选择图像分类标注。

![image-20221102123257761](docs.assets/image-20221102123257761.png)

然后在右侧表单中依次选择图片文件夹、该分类任务的所有分类情况，用逗号(英文符号)隔开，配置每一张图像可以分配的种类上限。一般任务只需设置每张图像唯一分类即可。

![image-20221102123503156](docs.assets/image-20221102123503156.png)

点击创建后进入标注界面，在标注界面左侧可以选择上一张标注图像或下一张标注图像。

![image-20221102123711316](docs.assets/image-20221102123711316.png)

同时还可以点击设置，修改当前项目的一些基本属性。

![image-20221102123813945](docs.assets/image-20221102123813945.png)

然后，可以关闭设置界面，回到标注界面，在界面中间偏下有一个白色方框，点击后可选择当前图像的类别。

![image-20221102124021581](docs.assets/image-20221102124021581.png)

选择好当前图像类别后，点击正下方的蓝色钩按钮进行保存(或Ctrl+S保存)，当前图像标注保存成功后会有提示。

![image-20221102124129835](docs.assets/image-20221102124129835.png)

然后依次点击左侧的后一个(或键盘右方向键)即可往后继续进行图像分类数据的标注。

最后，标注完成后，可以点击左侧的导出按钮进行数据标注的导出。选择输出方式为json、配置导出路径即可。

![image-20221102130337149](docs.assets/image-20221102130337149.png)

导出后文件在指定目录下生成outputs标注文件目录，并在其中对之前导入的所有图像生成标注文件。(务必对所有导入的图像进行标注后再导出)

![image-20221102124528486](docs.assets/image-20221102124528486.png)

![image-20221102130403835](docs.assets/image-20221102130403835.png)

## 2.2 数据格式转化

PaddleX做为飞桨全流程开发工具，提供了非常多的工具，在这里我们使用paddlex进行数据格式转化。
首先安装paddlex

```bash
pip install paddlex
```

目前所有标注工具生成的标注文件，均为与原图同名的json格式文件，如`1.jpg`在标注完成后，则会在标注文件保存的目录中生成`1.json`文件。转换时参照以下步骤：

1. 将所有的原图文件放在同一个目录下，如`pics`目录  
2. 将所有的标注json文件放在同一个目录下，如`annotations`目录  
3. 使用如下命令进行转换:

```bash
paddlex --data_conversion --source jingling --to ImageNet --pics ./pics --annotations ./annotations --save_dir ./converted_dataset_dir
```

| 参数          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| --source      | 表示数据标注来源，支持`labelme`、`jingling`（分别表示数据来源于LabelMe，精灵标注助手） |
| --to          | 表示数据需要转换成为的格式，支持`ImageNet`（图像分类）、`PascalVOC`（目标检测），`MSCOCO`（实例分割，也可用于目标检测）和`SEG`(语义分割) |
| --pics        | 指定原图所在的目录路径                                       |
| --annotations | 指定标注文件所在的目录路径                                   |

## 2.3 数据划分

在这里，我们依旧使用paddlex进行数据划分
使用paddlex命令即可将数据集随机划分成70%训练集，20%验证集和10%测试集:

```bash
paddlex --split_dataset --format ImageNet --dataset_dir ./converted_dataset_dir --val_value 0.2 --test_value 0.1
```

执行上面命令行，会在./converted_dataset_dir下生成 labels.txt, train_list.txt, val_list.txt, test_list.txt，分别存储训练样本信息，验证样本信息，测试样本信息

至此我们的数据就创作完成了，最终我们的产出形态应如下所示

- 文件结构

```bash
|converted_dataset_dir
|--class1
|  |--image1.jpg
|  |--image2.jpg
|  |--...
|--class2
|  |--image1.jpg
|  |--image2.jpg
|  |--...
...
|
|--labels.txt
|
|--train_list.txt
|
|--val_list.txt
|
|--test_list.txt
```

- 文件夹converted_dataset_dir为数据目录、class1为类别1图像所在目录、class2为类别2图像所在目录。其中converted_dataset_dir命名不是必须的，用户可以自主进行命名。

- 其中labels.txt的内容如下所示(即包含数据集中实际分类名称的文件)

  ```txt
  class1
  class2
  ...
  ```

- 其中train.txt和val.txt的内容如下所示：

```txt
class1/image1.jpg 0
...
class2/image1.jpg 1
...
```

**note**非常重要！！！

我们一般推荐用户将数据集放置在PaddleClas下的dataset文件夹下，下文配置文件的修改也是按照该方式。

# 3 模型训练与验证

## 3.1模型训练参数说明

### 3.1.1 训练前准备

我们可以通过PaddleClas提供的脚本对模型进行训练，在本小节中我们使用HRNet模型与flowers102数据集展示训练过程。 在训练之前，最重要的修改自己的数据情况，确保能够正常训练。

在本项目中，我们使用PaddleClas/ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml进行训练。

我们需要修改PaddleClas/ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml中数据集的路径、模型的分类数（class_num）、模型类别id与实际类别映射文件（flowers102_label_list.txt--包含数据集中实际分类名称的文件，文件名不固定）以及预测后处理输出类别（设置每张图像只输出1个类别），修改为如下内容。

```yaml
# model architecture
Arch:
  name: HRNet_W18_C
  class_num: 102

# data loader for train and eval
DataLoader:
  Train:
    dataset:
      name: ImageNetDataset
      image_root: ./dataset/flowers102/
      cls_label_path: ./dataset/flowers102/train_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - RandCropImage:
            size: 224
        - RandFlipImage:
            flip_code: 1
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''

    sampler:
      name: DistributedBatchSampler
      batch_size: 128
      drop_last: False
      shuffle: True
    loader:
      num_workers: 4
      use_shared_memory: True

  Eval:
    dataset: 
      name: ImageNetDataset
      image_root: ./dataset/flowers102/
      cls_label_path: ./dataset/flowers102/val_list.txt
      transform_ops:
        - DecodeImage:
            to_rgb: True
            channel_first: False
        - ResizeImage:
            resize_short: 256
        - CropImage:
            size: 224
        - NormalizeImage:
            scale: 1.0/255.0
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
            order: ''
    sampler:
      name: DistributedBatchSampler
      batch_size: 64
      drop_last: False
      shuffle: False
    loader:
      num_workers: 4
      use_shared_memory: True
...
Infer:
  infer_imgs: docs/images/inference_deployment/whl_demo.jpg
  ...
  PostProcess:
    name: Topk
    topk: 1
    class_id_map_file: ./dataset/flowers102/flowers102_label_list.txt
```

**note**非常重要！！！

* 关键改动的配置中的路径，这一个涉及相对路径，安照提示一步步来，确保最终能够完成。
* 本次项目中使用到的flowers102数据包含了1020张训练图片、1020张验证图片、0张测试图片，共102种分类，获取数据集的方式为执行以下脚本:

```bash
# linux or mac
cd dataset/
# 如果希望从浏览器中直接下载，可以复制该链接并访问，然后下载解压即可
wget https://paddle-imagenet-models-name.bj.bcebos.com/data/flowers102.zip
# 解压
unzip flowers102.zip
# 文件结构
# |-- dataset
# | | -- flowers102
# | | | -- jpg
# | | | | -- image1.jpg
# | | | | -- ...
# | | | -- flowers102_label_list.txt
# | | | -- ...
# | | | -- train_list.txt
# | | | -- val_list.txt
# 回到PaddleClas目录
cd ../
```

- 以上脚本要保证工作目录已进入到PaddleClas目录(若没有进入，需要通过cd指令进入)

```bash
# 将cd后的参数改成实际目录的路径即可
cd PaddleClas_path
```



### 3.1.2 开始训练

请确保已经完成了PaddleClas的安装工作，并且当前位于PaddleClas目录下，执行以下脚本：

```bash
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0

python tools/train.py \
       -c ./ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml \
       -o Arch.pretrained=True \
       -o Global.epochs=40 \
       -o Global.eval_interval=2 \
       -o Global.save_interval=2 \
       -o Global.print_batch_step=10 \
	   -o Global.use_visualdl=True \
	   -o Global.output_dir=./outputs \
	   -o Global.eval_during_train=True
```

### 3.1.3 训练参数解释

| 参数名                      | 用途                                            | 是否必选项 | 默认值 |
| :-------------------------- | :---------------------------------------------- | :--------- | :----- |
| -c                          | 指定模型的yaml-config文件                       | 是         | -      |
| -o                          | 覆盖配置文件中的指定参数值                      | 否         | -      |
| -o Arch.pretrained=True     | 覆盖配置文件中的模型加载预训练权重状态值        | 否         | -      |
| -o Global.epochs            | 覆盖配置文件中的训练迭代轮次值                  | 否         | -      |
| -o Global.eval_interval     | 覆盖配置文件中的评估间隔轮次值                  | 否         | -      |
| -o Global.save_interval     | 覆盖配置文件中的模型保存间隔轮次值              | 否         | -      |
| -o Global.print_batch_step  | 覆盖配置文件中的每个batch训练日志输出迭代间隔值 | 否         | -      |
| -o Global.use_visualdl      | 覆盖配置文件中的visualdl日志器的状态值          | 否         | -      |
| -o Global.output_dir        | 覆盖配置文件中的模型保存/输出路径               | 否         | -      |
| -o Global.eval_during_train | 覆盖配置文件中的训练时评估模式的状态值          | 否         | -      |
|                             |                                                 |            |        |

### 3.1.3 多卡训练

如果想要使用多卡训练的话，需要将环境变量CUDA_VISIBLE_DEVICES指定为多卡（不指定时默认使用所有的gpu)，并使用paddle.distributed.launch启动训练脚本（windows下由于不支持nccl，无法使用多卡训练）:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 设置4张可用的卡
python -m paddle.distributed.launch tools/train.py \
       -c ./ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml \
       -o Arch.pretrained=True \
       -o Global.epochs=40 \
       -o Global.eval_interval=2 \
       -o Global.save_interval=2 \
       -o Global.print_batch_step=10 \
	   -o Global.use_visualdl=True \
	   -o Global.output_dir=./outputs \
	   -o Global.eval_during_train=True
```

### 3.1.4 恢复训练

```bash
python tools/train.py \
       -c ./ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml \
       -o Global.checkpoints=./outputs/HRNet_W18_C/epoch_14 \
       -o Global.epochs=40 \
       -o Global.eval_interval=2 \
       -o Global.save_interval=2 \
       -o Global.print_batch_step=10 \
	   -o Global.use_visualdl=True \
	   -o Global.output_dir=./outputs \
	   -o Global.eval_during_train=True
```


### 3.1.5 训练可视化

PaddleClas会将训练过程中的数据写入VisualDL文件，并实时的查看训练过程中的日志，记录的数据包括：（当-o Global.use_visualdl=True开启后生效）

1. loss变化趋势
2. 学习率变化趋势
3. 日志记录时间
4. top1变化趋势
5. top5变化趋势
6. 评估精度(acc)变化趋势（当-o Global.eval_during_train=True开启后生效）

使用如下命令启动VisualDL查看日志

```bash
# 下述命令会在127.0.0.1上启动一个服务，支持通过前端web页面查看，可以通过--host这个参数指定实际ip地址
visualdl --logdir outputs/vdl
```

在浏览器输入提示的网址，效果如下：
![image-20221103192426287](docs.assets/vdl.png)


## 3.2 模型验证参数说明

### 3.2.1 评估操作

训练完成后，用户可以使用评估脚本tools/eval.py来评估模型效果。假设训练过程中迭代轮次（epochs）为40，保存模型的间隔为2，即每迭代2个轮次的数据集就会保存1次训练的模型。因此一共会产生20个定期保存的模型，加上保存的最佳模型`best_model`以及最近保存的模型`latest`，一共有22个模型，可以通过`-o Global.pretrained_model`指定期望评估的模型文件。

```bash
python tools/eval.py \
    -c ./ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml \
    -o Global.pretrained_model=./outputs/HRNet_W18_C/best_model
```

- 参数说明如下

| 参数名                     | 用途                       | 是否必选项 | 默认值 |
| :------------------------- | :------------------------- | :--------- | :----- |
| -c                         | 指定模型的yaml-config文件  | 是         | -      |
| -o Global.pretrained_model | 指定模型加载特定的权重文件 | 否         | -      |

**注意** 如果你想提升显存利用率，可以适当的提高 num_workers 的设置，以防GPU工作期间空等。设置评估时的 num_workers 指令如下:

```bash
python tools/eval.py \
    -c ./ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml \
    -o Global.pretrained_model=./outputs/HRNet_W18_C/best_model \
    -o DataLoader.Eval.loader.num_workers=6
```

### 3.2.2 评估方式说明

在图像分类领域中，评估模型质量主要是准确率（acc）来评估的。

- 准确率（Acc）：表示预测正确的样本数占总数据的比例，准确率越高越好
  - Top1 Acc：预测结果中概率最大的所在分类正确，则判定为正确；
  - Top5 Acc：预测结果中概率排名前 5 中有分类正确，则判定为正确；

随着评估脚本的运行，最终打印的评估日志如下：

```bash
[2022/11/03 19:32:26] ppcls INFO: [Eval][Epoch 0][Iter: 0/16]CELoss: 1.05227, loss: 1.05227, top1: 0.75000, top5: 0.92188, batch_cost: 3.20560s, reader_cost: 1.45519, ips: 19.96508 images/sec
[2022/11/03 19:32:28] ppcls INFO: [Eval][Epoch 0][Iter: 10/16]CELoss: 0.40082, loss: 0.40082, top1: 0.89062, top5: 0.97159, batch_cost: 0.21201s, reader_cost: 0.00127, ips: 301.87772 images/sec
[2022/11/03 19:32:29] ppcls INFO: [Eval][Epoch 0][Avg]CELoss: 0.54829, loss: 0.54829, top1: 0.87549, top5: 0.97255
```

## 3.3 模型预测

除了可以分析模型的准确率指标之外，我们还可以对一些具体样本的预测效果，从Bad Case启发进一步优化的思路。

tools/infer.py脚本是专门用来可视化预测案例的，命令格式如下所示：

```bash
python tools/infer.py \
    -c ./ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml \
    -o Global.pretrained_model=./outputs/HRNet_W18_C/best_model \
    -o Infer.infer_imgs=dataset/flowers102/jpg/image_00001.jpg
```

其中-o Infer.infer_imgs可以指定一张图片的路径。

**note**非常重要！！！

* 批量预测一个jpg目录下的所有图像，使用以下指令:

```bash
python tools/infer.py \
    -c ./ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml \
    -o Global.pretrained_model=./outputs/HRNet_W18_C/best_model \
    -o Infer.infer_imgs=dataset/flowers102/jpg/ \
    -o Infer.batch_size=16
```

- 其中，batch_size表示每一次预测批量处理16张图像，并且输出结果中，会将每16个结果放在一个列表中输出：

  ![image-20221103193414237](docs.assets/batch_infer_log.png)

### 3.3.1 输出说明

- 在执行预测时，仅需要原始图像。无论是单张图片预测还是目录预测，都会输出每个样本的预测结果如下：

```bash
[{'class_ids': [76], 'scores': [0.99997], 'file_name': 'dataset/flowers102/jpg/image_00001.jpg', 'label_names': ['passion flower']}]
```

其中:

- class_ids: 表示预测类别的id号
- scores: 预测为该类别的得分
- file_name: 当前预测的文件名称
- label_names: 对应类别id所属的实际类别

# 4 配置文件的说明

正是因为有配置文件的存在，我们才可以使用更便捷的进行消融实验。在本章节中我们选择
```PaddleClas/ppcls/configs/ImageNet/HRNet/HRNet_W18_C.yaml```文件来进行配置文件的详细解读

## 4.1 整体配置文件格式综述

我们将```HRNet_W18_C.yaml```进行拆分解释

* **HRNet**表示模型的名称
* **W18**表示模型的最小分支通道数(基础通道数)——W18对应模型的高分辨率多分支结构中的最小通道数为18
* **C**表示模型输出为分类的聚合模式——将输出的多分支特征进行聚合得到最终的分类特征

**引入配置文件的说明**

当前PaddleClas为了降低配置网络的难度，利用配置文件进行训练、评估等模型的加载工作。要实现一个模型的训练，往往就需要至少1个配置文件才可运行，如，我们现在选择的```HRNet_W18_C.yaml```。

## 4.2 数据路径与数据预处理说明

这一小节主要是说明数据部分，当准备好数据，如何进行配置文件修改，以及该部分的配置文件有什么内容。
如下是截取的是```HRNet_W18_C.yaml```配置。

```yaml
DataLoader: # 数据加载器配置部分
  Train: # 训练数据加载部分
    dataset: # 训练数据集配置
      name: ImageNetDataset # 数据集类型 —— 统一使用ImageNetDataset格式
      image_root: ./dataset/flowers102/ #数据集路径
      cls_label_path: ./dataset/flowers102/train_list.txt #根据该txt寻找验证的数据路径
      transform_ops: #数据预处理/增强的方式
        - DecodeImage: # 读取图像
            to_rgb: True # 图像读取后转为RGB图像格式
            channel_first: False # 转换图像数据为通道在前
        - RandCropImage: #从原始图像中随机裁剪224x224大小
            size: 224
        - RandFlipImage: #将原始图像随机左右翻转
            flip_code: 1
        - NormalizeImage: #将图像归一化
            scale: 1.0/255.0 # 归一化缩放因子
            mean: [0.485, 0.456, 0.406] # 归一化均值
            std: [0.229, 0.224, 0.225] # 归一化方差
            order: '' # 归一化前数据格式，不配置默认为HWC
    sampler: # 数据采样器配置部分
      name: DistributedBatchSampler # 采样器类型——分布式批量数据采样器
      batch_size: 256 # 每次采样的batch size
      drop_last: False # 不丢弃数据集末尾采样不足batch_size大的训练数据
      shuffle: True # 随机打乱训练数据顺序
    loader: # 数据加载其它参数配置
      num_workers: 8 # 加载使用的线程数
      use_shared_memory: True # 加载使用共享内存

  Eval: # 验证数据加载部分
    dataset: # 验证数据集配置
      name: ImageNetDataset # 数据集类型 —— 统一使用ImageNetDataset格式
      image_root: ./dataset/flowers102/ #数据集路径
      cls_label_path: ./dataset/flowers102/val_list.txt #根据该txt寻找验证的数据路径
      transform_ops: #数据预处理/增强的方式
        - DecodeImage: # 读取图像
            to_rgb: True # 图像读取后转为RGB图像格式
            channel_first: False # 转换图像数据为通道在前
        - ResizeImage: # 缩放图像大小为256*256
            resize_short: 256 
        - CropImage: # 裁剪原始图像224*224的中心区域
            size: 224
        - NormalizeImage: #将图像归一化
            scale: 1.0/255.0 # 归一化缩放因子
            mean: [0.485, 0.456, 0.406] # 归一化均值
            std: [0.229, 0.224, 0.225] # 归一化方差
            order: '' # 归一化前数据格式，不配置默认为HWC
    sampler: # 数据采样器配置部分
      name: DistributedBatchSampler # 采样器类型——分布式批量数据采样器
      batch_size: 64 # 每次采样的batch size
      drop_last: False # 不丢弃数据集末尾采样不足batch_size大的训练数据
      shuffle: False # 不随机打乱训练数据顺序
    loader: # 数据加载其它参数配置
      num_workers: 4 # 加载使用的线程数
      use_shared_memory: True # 加载使用共享内存
```

**note**

* 关于如何正确来写image_root以及cls_label_path是非常关键的，可以根据上一章节训练的过程以及数据标注章节进行相对文件夹路径的推演。
* ``class_num``配置需要在HRNet_W18_C.yaml中的以下位置配置:

```yaml
# model architecture
Arch:
  name: HRNet_W18_C
  class_num: 102
```

- 预测时对应的id与实际类别映射的文件需要配置好才能保证预测结果中的类别名称正确，其改动需要在HRNet_W18_C.yaml中的以下位置配置:

```yaml
Infer:
  infer_imgs: docs/images/inference_deployment/whl_demo.jpg
  batch_size: 10
  ...
  PostProcess:
    name: Topk
    topk: 1 # 预测每张图像最可能的类别数，一般设置为1，即表示预测输出当前图像的实际类别
    class_id_map_file: dataset/flowers102/flowers102_label_list.txt # 修改为当前数据集的类别映射文件即可，该文件格式即第二章中数据划分所示的labels.txt
```

## 4.3 模型说明

当我们配置好数据后，下面在看关于模型的选择说明

``` yaml
Arch:
  name: HRNet_W18_C
  class_num: 102
  pretrained: True # yaml可能不包含，但是可以自行添加
```

  **note**

* 我们模型的name是HRNet_W18_C，即表示选用HRNet_W18_C这个模型
  * 类似的，PaddleClas还包含: HRNet_W32_C、HRNet_W48_C、HRNet_W64_C等同类不同配置的模型网络，详细可查看[官网模型](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.5/ppcls/configs/ImageNet/HRNet)
* class_num 配置该模型总的预测类别数
* pretrained 配置是否加载预训练模型辅助训练
* PaddleClas对于不同的模型有不同的yaml配置文件，可以在ppcls/configs/ImageNet或ppcls/configs下查询相应模型

## 4.4 优化器和损失函数说明

当我们配置好数据以及选定模型后，可以再进行优化器以及损失函数的选择(在HRNet_W18_C.yaml)

``` yaml
Optimizer: # 优化器配置部分
  name: Momentum # 优化器类型 —— 动量优化器
  momentum: 0.9 # 动量大小
  lr: # 学习率配置
    name: Piecewise # 学习率策略类型 —— 分段衰减策略
    learning_rate: 0.1 # 基础学习率大小
    decay_epochs: [30, 60, 90] # 分段轮次: 第30轮、第60轮、第90轮发生学习率变化
    values: [0.1, 0.01, 0.001, 0.0001] # 分段学习率变化值: 第30轮前为0.1，第30轮及其后为0.01，第60轮及其后为0.001，第90轮及其后为0.0001
  regularizer: # 正则化配置
    name: 'L2' # 正则化类型 —— L2正则，另外还有L1正则
    coeff: 0.0001 # 正则因子

Loss: # 损失函数配置部分
  Train: # 训练损失函数部分
    - CELoss: # 交叉熵损失
        weight: 1.0 # 权重大小为1.0
  Eval: # 验证损失函数部分
    - CELoss: # 交叉熵损失
        weight: 1.0 # 权重大小为1.0
```

**note**

*  学习率策略类型支持有Momentum, MultiStepDecay, Step等6种
*  优化器类型支持有AdamWDL, AdamW, RMSProp, SGD等6种

## 4.5 其他参数说明

``` yaml
Global: # 公共配置
  checkpoints: null # 检查点 —— 即断训的路径描述(不包括.pdparams后缀)
  pretrained_model: null # 训练好的模型路径描述(不包括.pdparams后缀)
  output_dir: ./output/ # 模型保存路径
  device: gpu # 模型运行/执行设备为GPU
  save_interval: 1 # 保存轮次间隔
  eval_during_train: True # 边训练边评估 —— 按照验证轮次间隔进行评估
  eval_interval: 1 # 验证轮次间隔
  epochs: 120 # 训练总轮次
  print_batch_step: 10 # 日志输出间隔
  use_visualdl: False # 不适用visualdl进行日志记录
  # used for static mode and model export
  image_shape: [3, 224, 224] # 处理的图像大小 —— 导出部署模型时的图像大小
  save_inference_dir: ./inference # 部署模型导出路径
```

``` yaml
Metric: # 评估标准配置
  Train: # 训练评估标准部分
    - TopkAcc: # Topk精确度评估标准
        topk: [1, 5] # Top5精确度的评估标准
  Eval: # 验证评估标准部分
    - TopkAcc: # Topk精确度评估标准
        topk: [1, 5] # Top5精确度的评估标准
```







