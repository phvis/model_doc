# 概述

本文重点介绍如何利用飞桨目标检测套件**PaddleDetection**在小度熊实例分割数据上，使用当前PaddleDetection的Mask RCNN模型进行详细讲解。
Mask RCNN模型是PaddleDetection内置的二阶段实例分割模型，其模型结构如下。



<div align=center><img width="604" alt="image" src="docs.assets/image-20221115190347238.png"></div>

Mask RCNN模型更详细的原理介绍请参考[官方论文](https://arxiv.org/abs/1703.06870)以及[官网说明](https://github.com/PaddlePaddle/PaddleDetection)。

## 文章目录结构

- 1 环境安装
  - 1.1 PaddlePaddle安装
    - 1.1.1 安装对应版本PaddlePaddle
    - 1.1.2 验证安装是否成功
  - 1.2 PaddleDetection安装
    - 1.2.1 下载PaddleDetection代码
    - 1.2.2 安装依赖项目
    - 1.2.3 验证安装是否成功
- 2 数据准备
  - 2.1 数据标注
    - 2.1.1 LabelMe安装
    - 2.1.2 LabelMe的使用
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
    - 3.3.1 准备预测数据
    - 3.3.2 输出文件说明
- 4 配置文件的说明
  - 4.1 整体配置文件格式综述
  - 4.2 数据路径与数据预处理说明
  - 4.3 模型说明
  - 4.4 优化器说明
  - 4.5 其它参数说明



# 1 环境安装

## 1.1 PaddlePaddle安装

### 1.1.1 安装对应版本PaddlePaddle

根据系统和设备的cuda环境，选择对应的安装包，这里默认使用pip在linux设备上进行安装。

<div align=center><img width="604" alt="image" src="https://user-images.githubusercontent.com/48433081/176497642-0abf3de1-86d5-43af-afe8-f97db46b7fd9.png"></div>

在终端执行

```
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

## 1.2 PaddleDetection安装

### 1.2.1 下载PaddleDetection代码

用户可以通过使用github或者gitee的方式进行下载，我们当前版本为PaddleDetection的release v2.5版本。后续在使用时，需要对应版本进行下载。

<div align=center><img width="604" alt="image" src="docs.assets/image-20221102134211058.png"></div>

``` bash
# github下载
git clone https://github.com/PaddlePaddle/PaddleDetection.git
# gitee下载
git clone https://gitee.com/PaddlePaddle/PaddleDetection.git
```

### 1.2.2 安装依赖项目

* 方式一：
  通过直接pip install 安装，可以最高效率的安装依赖

``` bash
pip install paddledet
```

* 方式二：
  下载PaddleDetection代码后，进入PaddleDetection代码文件夹目录下面

``` bash
cd PaddleDetection
pip install -r requirements.txt
```

### 1.2.3 验证安装是否成功

在PaddleDetection目录下执行如下命令，会进行简单的单卡训练和单卡预测。

等待模型下载以及查看执行日志，若没有报错，则验证安装成功。

``` bash
python tools/infer.py -c configs/ppyolo/ppyolo_r50vd_dcn_1x_coco.yml -o use_gpu=true weights=https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams --infer_img=demo/000000014439.jpg
```

# 2 数据准备

## 2.1 数据标注

无论是语义分割，实例分割，还是目标检测，我们都需要充足的训练数据。如果你想使用没有标注的原始数据集做实例分割任务，你必须先为原始图像作出标注。

### 2.1.1 LabelMe安装



用户在采集完用于训练、评估和预测的图片之后，需使用数据标注工具[LabelMe](https://github.com/wkentaro/labelme)完成数据标注。

LabelMe支持在Windows/macOS/Linux三个系统上使用，且三个系统下的标注格式是一样。具体的安装流程请参见[官方安装指南](https://github.com/wkentaro/labelme)。

### 2.1.2 LabelMe的使用

打开终端输入`labelme`会出现LableMe的交互界面，可以先预览`LabelMe`给出的已标注好的图片，再开始标注自定义数据集。

<div align=center><img width="604" alt="image" src="docs.assets/176489586-ac8cd9df-cfa5-418a-824f-2415c0694c0c.png"></div>

<div align="center">
    <p>图1 LableMe交互界面的示意图</p>
 </div>
   * 开始标注

请按照下述步骤标注数据集(标注内容与语义分割基本一致)：

​        (1)   点击`OpenDir`打开待标注图片所在目录，点击`Create Polygons`，沿着目标的边缘画多边形，完成后输入目标的类别。在标注过程中，如果某个点画错了，可以按撤销快捷键可撤销该点。Mac下的撤销快捷键为`command+Z`。

<div align=center><img width="604" alt="image" src="docs.assets/176489848-12e8f9ab-344f-4234-84a3-482f12275911.png"></div>

<div align="center">
    <p>图2 标注单个目标的示意图</p>
 </div>

​		(2)   右击选择`Edit Polygons`可以整体移动多边形的位置，也可以移动某个点的位置；右击选择`Edit Label`可以修改每个目标的类别。请根据自己的需要执行这一步骤，若不需要修改，可跳过。

<div align=center><img width="604" alt="image" src="docs.assets/176490004-b5adeb07-c30a-4b51-ad42-b2b90f65819c.png"></div>

<div align=center><img width="604" alt="image" src="docs.assets/176490269-fb10da8d-9d86-49d7-8de8-bc3789427e8b.png"></div>

<div align="center">
    <p>图3 修改标注的示意图</p>
 </div>

(3)   图片中所有目标的标注都完成后，点击`Save`保存json文件，**请将json文件和图片放在同一个文件夹里**，点击`Next Image`标注下一张图片。

<div align=center><img width="604" alt="image" src="docs.assets/176490440-2a941d27-9ab2-4eb9-b3dd-41821168ce30.png"></div>

<div align="center">
    <p>图4 LableMe产出的真值文件的示意图</p>
 </div>

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
paddlex --data_conversion --source labelme --to MSCOCO --pics ./pics --annotations ./annotations --save_dir ./converted_dataset_dir
```

| 参数          | 说明                                                         |
| ------------- | ------------------------------------------------------------ |
| --source      | 表示数据标注来源，支持`labelme`、`jingling`（分别表示数据来源于LabelMe，精灵标注助手） |
| --to          | 表示数据需要转换成为的格式，支持`ImageNet`（图像分类）、`PascalVOC`（目标检测），`MSCOCO`（实例分割，也可用于目标检测）和`SEG`(语义分割) |
| --pics        | 指定原图所在的目录路径                                       |
| --annotations | 指定标注文件所在的目录路径                                   |

- **note**非常重要！！！

  经过上述转换为MSCOCO后的数据集就是实例分割的数据集了。
  
  若图像与标注文件保存在同一个文件夹中，则只需保持`--pics`以及`--annotations`的路径参数一致即可

```bash
paddlex --data_conversion --source labelme --to MSCOCO --pics ./pics --annotations ./pics --save_dir ./converted_dataset_dir
```

## 2.3 数据划分

在这里，我们依旧使用paddlex进行数据划分
使用paddlex命令即可将数据集随机划分成70%训练集，20%验证集和10%测试集:

(实例分割数据集需要按照COCO格式的数据划分方式)

```bash
paddlex --split_dataset --format COCO --dataset_dir ./converted_dataset_dir --val_value 0.2 --test_value 0.1
```

执行上面命令行，会在`./converted_dataset_dir`下生成`train.json`, val.json`, `test.json，分别存储训练样本信息，验证样本信息，测试样本信息

至此我们的数据就创作完成了，最终我们的产出形态应如下所示

- 文件结构

  ```bash
  custom_dataset
  |
  |--JPEGImages
  |  |--image1.jpg
  |  |--image2.jpg
  |  |--...
  |
  |--...
  |
  |--train.json
  |
  |--val.json
  |
  |--test.json
  ```

- 文件夹命名为custom_dataset不是必须，用户可以自主进行命名。

- 其中`train.json`的内容如下所示：

  ```txt
  root:
  	annotations:
  		0:
  			segmentation: 
  				0:
                       0:658
                       1:542
                       2:670
                       3:555
                       4:698
                       5:563
                       6:720
                       7:554
                       8:737
                       9:557
                       10:738
                       11:539
                       12:738
                       13:498
                       14:734
                       ...
  			iscrowd: 0
  			...
  			bbox: 
  				0:594
                   1:467
                   2:144
                   3:96
               area:13824
               category_id:1
               id:1
          1:
           	...
          ...
      images:
      	...
      categories:
      	...
  ```

- **note**非常重要！！！

我们一般推荐用户将数据集放置在PaddleDetection下的dataset文件夹下，下文配置文件的修改也是按照该方式。

# 3 模型训练与验证

## 3.1模型训练参数说明

### 3.1.1 训练前准备

我们可以通过PaddleDetection提供的脚本对模型进行训练，在本小节中我们使用MASK RCNN模型与小度熊实例分割数据集展示训练过程。 在训练之前，最重要的修改自己的数据情况，确保能够正常训练。

在本项目中，我们使用```configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml```进行训练。

我们发现
`mask_rcnn_r50_fpn_1x_coco.yml`，需要逐层依赖`../datasets/coco_instance.yml`、`_base_/mask_rcnn_r50_fpn.yml`、`_base_/optimizer_1x.yml` 、`_base_/mask_fpn_reader.yml`、`../runtime.yml`。

在这里改动../datasets/coco_instance.yml中文件的路径，修改为如下内容。

```yml
metric: COCO # COCO数据集评价指标
num_classes: 1 # 数据的分类数(不包含类别背景)

TrainDataset: # 训练数据集配置
  !COCODataSet
    image_dir: JPEGImages # 3.图像数据文件夹
    anno_path: train.json # 2.标注记录的json
    dataset_dir: dataset/xiaoduxiong_ins_det # 1.数据集所在根目录
    data_fields: ['image', 'gt_bbox', 'gt_class', 'gt_poly', 'is_crowd']

EvalDataset: # 验证数据集配置
  !COCODataSet
    image_dir: JPEGImages # 3.图像数据文件夹
    anno_path: val.json # 2.标注记录的json
    dataset_dir: dataset/xiaoduxiong_ins_det # 1.数据集所在根目录

TestDataset: # 预测数据集配置
  !ImageFolder
    anno_path: annotations.json # 2.包含数据集全部类别信息的任意json标注文件
    dataset_dir: dataset/xiaoduxiong_ins_det # 1.数据集所在根目录
```

**note**非常重要！！！

* 关键改动的配置中的路径，这一个涉及相对路径，安照提示一步步来，确保最终能够完成。

* 本次项目中使用到的数据[下载链接](https://bj.bcebos.com/paddlex/datasets/xiaoduxiong_ins_det.tar.gz)，本章节将使用小度熊实例分割数据集进行训练，小度熊实例分割数据集是由一些长相不相同的小度熊组成的COCO格式实例分割数据集，包含了14张训练图片、4张验证图片、3张测试图片，包含1个类别: xiaoduxiong.(以下是目录结构)

```bash
|--xiaoduxiong_ins_det
|	|--JPEGImages
|		|--image1.png
|		|--...
|	|--annotations.json
|	|--train.json
|	|--val.json
|	|--test.json
```

### 3.1.2 开始训练

请确保已经完成了PaddleDetection的安装工作，并且当前位于PaddleDetection目录下，执行以下脚本：

```bash
export CUDA_VISIBLE_DEVICES=0 # 设置1张可用的卡
# windows下请执行以下命令
# set CUDA_VISIBLE_DEVICES=0

python tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
    -o save_dir=./output \
    --eval \
    --use_vdl=True \
    --vdl_log_dir=./output/vdl_log_dir/scalar
```

### 3.1.3 训练参数解释

| 主要参数名    | 用途                                                         | 是否必选项 | 默认值             |
| :------------ | :----------------------------------------------------------- | :--------- | :----------------- |
| -c            | 指定训练模型的yaml文件                                       | 是         | 无                 |
| -o            | 修改yaml中的一些训练参数值                                   | 否         | 无                 |
| -r            | 指定模型参数进行继续训练                                     | 否         | 无                 |
| -o save_dir   | 修改yaml中模型保存的路径(不使用该参数，默认保存在output目录下) | 否         | output             |
| --eval        | 指定训练时是否边训练边评估                                   | 否         | -                  |
| --use_vdl     | 指定训练时是否使用visualdl记录数据                           | 否         | False              |
| --vdl_log_dir | 指定visualdl日志文件的保存路径                               | 否         | vdl_log_dir/scalar |
| --slim_config | 指定裁剪/蒸馏等模型减小的配置                                | 否         | 无                 |
| --amp         | 启动混合精度训练                                             | 否         | False              |

### 3.1.3 多卡训练

如果想要使用多卡训练的话，需要将环境变量CUDA_VISIBLE_DEVICES指定为多卡（不指定时默认使用所有的gpu)，并使用paddle.distributed.launch启动训练脚本（windows下由于不支持nccl，无法使用多卡训练）:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3 # 设置4张可用的卡
python tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
    -o save_dir=./output \
    --eval \
    --use_vdl=True \
    --vdl_log_dir=./output/vdl_log_dir/scalar
```

### 3.1.4 恢复训练

```bash
python tools/train.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
	-r ./output/mask_rcnn_r50_fpn_1x_coco/9 \
    -o save_dir=./output \
    --eval \
    --use_vdl=True \
    --vdl_log_dir=./output/vdl_log_dir/scalar
```

### 3.1.5 训练可视化

PaddleDetection会将训练过程中的数据写入VisualDL文件，并实时的查看训练过程中的日志，记录的数据包括：

1. 当前计算各种损失的变化趋势
2. 日志记录时间
3. mAP变化趋势（当打开了`do_eval`开关后生效）

使用如下命令启动VisualDL查看日志

```bash
# 下述命令会在127.0.0.1上启动一个服务，支持通过前端web页面查看，可以通过--host这个参数指定实际ip地址
visualdl --logdir output/
```

在浏览器输入提示的网址，效果如下：

<div align=center><img width="604" alt="image" src="docs.assets/image-20221117225957809.png"></div>

## 3.2 模型验证参数说明

### 3.2.1 评估操作

训练完成后，用户可以使用评估脚本tools/eval.py来评估模型效果。假设训练过程中迭代轮次（epoch）为12，保存模型的间隔为1，即每迭代1次数据集保存1次训练模型。因此一共会产生12个定期保存的模型，加上保存的最佳模型`best_model`，一共有13个模型，可以通过`-o weights`指定期望评估的模型文件。

```bash
python tools/eval.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
	-o use_gpu=true \
	-o weights=./output/mask_rcnn_r50_fpn_1x_coco/best_model
```

如果获取评估后各个类别的PR曲线图像，可通过传入--classwise进行开启。使用示例如下：

```bash
python tools/eval.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
	-o use_gpu=true \
	-o weights=./output/mask_rcnn_r50_fpn_1x_coco/best_model \
	--classwise
```

图像保存在PaddleDetection/segm_pr_curve目录下，以下是其中一个类别的PR曲线图:（该图为实例分割中分割对应的PR曲线图）

<div align=center><img width="604" alt="image" src="docs.assets/image-20221117234035444.png"></div>

- 参数说明如下

| 主要参数名             | 用途                                       | 是否必选项 | 默认值 |
| ---------------------- | ------------------------------------------ | ---------- | ------ |
| -c                     | 指定训练模型的yaml文件                     | 是         | 无     |
| -o                     | 修改yaml中的一些训练参数值                 | 是         | 无     |
| -o use_gpu             | 指定评估时是否采用gpu                      | 否         | false  |
| -o weights             | 指定评估时模型采用的模型参数               | 否         | 无     |
| --output_eval          | 指定一个目录保存评估结果                   | 否         | None   |
| --json_eval            | 指定一个bbox.json/mask.json用于评估        | 否         | False  |
| --slim_config          | 指定裁剪/蒸馏等模型减小的配置              | 否         | None   |
| --bias                 | 对于取得w和h结果加上一个偏置值             | 否         | 无     |
| --classwise            | 指定评估结束后绘制每个类别的AP以及PR曲线图 | 否         | -      |
| --save_prediction_only | 指定评估仅保存评估结果                     | 否         | False  |
| --amp                  | 指定评估时采用混合精度                     | 否         | False  |

**注意** 如果你想提升显存利用率，可以适当的提高 num_workers 的设置，以防GPU工作期间空等。

### 3.2.2 评估方式说明

在实例分割领域中，评估模型质量与目标检测类似也是主要通过1个指标进行判断：mAP(平均精度)。

- IOU: 预测目标与真实目标的交并比

<div align=center><img width="604" alt="image" src="docs.assets/image-20221115165025244.png"></div>

- AP: 单个类别PR曲线下的面积，其中P为精确度，R为召回率。

<div align=center><img width="604" alt="image" src="docs.assets/image-20221115164911523.png"></div>

- TP: 预测目标的IOU>设定的某一个IOU阈值且预测正确时的检测框数量
- FP: 预测目标的IOU<设定的某一个IOU阈值且预测正确时/与单个真实框匹配后多余出来满足阈值的检测框数量
- FN: 没有检测到的真实框的数量
- mAP: 所有类别的AP的平均值
- 在实例分割中，bbox的mAP计算如上所述，而掩码mask部分也采用预测掩码区域和选定掩码区域计算IOU大小后放入公式中计算mAP

随着评估脚本的运行，最终打印的评估日志如下。

```bash
...
[11/17 12:04:10] ppdet.metrics.metrics INFO: The bbox result is saved to bbox.json.
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
[11/17 12:04:10] ppdet.metrics.coco_utils INFO: Start evaluate...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.02s).
Accumulating evaluation results...
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.265
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.732
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.026
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.265
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.064
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.329
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.364
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.364
[11/17 12:04:10] ppdet.metrics.metrics INFO: The mask result is saved to mask.json.
loading annotations into memory...
Done (t=0.00s)
creating index...
index created!
[11/17 12:04:10] ppdet.metrics.coco_utils INFO: Start evaluate...
Loading and preparing results...
DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *segm*
DONE (t=0.05s).
Accumulating evaluation results...
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.544
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.956
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.562
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.129
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.600
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.721
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.721
[11/17 12:04:10] ppdet.engine INFO: Total sample number: 4, averge FPS: 1.639492407880759
```

## 3.3 模型预测

除了分析模型的mAP指标之外，我们还可以查阅一些具体样本的检测样本效果，从Bad Case启发进一步优化的思路。

tools/infer.py脚本是专门用来可视化预测案例的，命令格式如下所示：

```bash
python tools/infer.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
	-o use_gpu=true \
	-o weights=./output/mask_rcnn_r50_fpn_1x_coco/best_model \
	--infer_img ./dataset/xiaoduxiong_ins_det/JPEGImages/Xiaoduxiong113.jpeg \
	--output_dir ./output
```

其中`--infer_img`是一张图片的路径，还可以用`--infer_dir`指定一个包含图片的目录，这时候将对该图片或文件列表或目录内的所有图片进行预测并保存可视化结果图。以下是直接预测的效果图:

<div align=center><img width="604" alt="image" src="docs.assets/image-20221117234912691.png"></div>

由于数据集较少，因此对输出后处理中的nms阈值进行配置，将阈值调小来避免重叠的框(位于\_base\_/mask_rcnn_r50_fpn.yml中，详细可查看4.3节的模型说明):

```yaml
BBoxPostProcess:
  decode:
  nms:
    name: MultiClassNMS
    keep_top_k: 100
    score_threshold: 0.05
    nms_threshold: 0.05 # 0.5 --> 0.05
```

<div align=center><img width="604" alt="image" src="docs.assets/image-20221117234706508.png"></div>

如果需要保存预测结果bbox.json文件，可以使用以下指令:

```bash
python tools/infer.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
	-o use_gpu=true \
	-o weights=./output/mask_rcnn_r50_fpn_1x_coco/best_model \
	--infer_img ./dataset/xiaoduxiong_ins_det/JPEGImages/Xiaoduxiong113.jpeg \
    --output_dir ./output \
    --save_results=True
```

另外如果需要使用切片进行小目标实例分割检测时，可以通过以下指令进行:

```bash
python tools/infer.py -c configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml \
	-o use_gpu=true \
	-o weights=./output/mask_rcnn_r50_fpn_1x_coco/best_model\
	--infer_img ./dataset/xiaoduxiong_ins_det/JPEGImages/Xiaoduxiong113.jpeg \
    --output_dir ./output \
    --save_results=True \
    --slice_infer \
    --slice_size 320 320
```

其中--slice_infer开启切片，--slice_size设置切片大小。

- 参数说明如下

| 主要参数名        | 用途                                                         | 是否必选项 | 默认值            |
| ----------------- | ------------------------------------------------------------ | ---------- | ----------------- |
| -c                | 指定训练模型的yaml文件                                       | 是         | 无                |
| -o                | 修改yaml中的一些训练参数值                                   | 是         | 无                |
| -o use_gpu        | 指定评估时是否采用gpu                                        | 否         | False             |
| -o weights        | 指定评估时模型采用的模型参数                                 | 否         | 无                |
| --visualize       | 指定预测结果要进行可视化                                     | 否         | True              |
| --output_dir      | 指定预测结果保存的目录                                       | 否         | None              |
| --draw_threshold  | 指定预测绘图时的得分阈值                                     | 否         | 0.5               |
| --slim_config     | 指定裁剪/蒸馏等模型减小的配置                                | 否         | None              |
| --use_vdl         | 指定预测时利用visualdl将预测结果(图像)进行记录               | 否         | False             |
| --vdl_log_dir     | 指定visualdl日志文件保存目录                                 | 否         | vdl_log_dir/image |
| --save_results    | 指定预测结果要进行保存                                       | 否         | False             |
| --slice_infer     | 指定评估时采用切片进行预测评估(对于smalldet采用)             | 否         | -                 |
| --slice_size      | 指定评估时的切片大小(以下指令均上一个指令使用时有效)         | 否         | [640,640]         |
| --overlap_ratio   | 指定评估时的切片图像的重叠高度比率(上一个指令使用时有效)     | 否         | [0.25,0.25]       |
| --combine_method  | 指定评估时的切片图像检测结果的整合方法，支持: nms, nmm, concat | 否         | nms               |
| --match_threshold | 指定评估时的切片图像检测结果的整合匹配的阈值，支持: 0-1.0    | 否         | 0.6               |
| --match_metric    | 指定评估时的切片图像检测结果的整合匹配的指标(支持)，支持: iou,ios | 否         | ios               |

**注意** 

- 1.如果直接预测目标重叠度高时可以修改nms阈值来进行一定的改善的。同样地，在训练前/重复训练时可以调整nms阈值参数进行人为干预的模型效果调优。
- 2.Mask RCNN在切片预测下的一般目标的效果不是很好，因此除非必要否则不进行切片预测——同时，切片操作只保留检测框，对于mask掩码的预测效果会丢失。

## 3.3.2 输出文件说明

当你指定输出位置--output_dir ./output后，在默认文件夹output下将生成与预测图像同名的预测结果图像:

```bash
# 预测图像image1.png
|--output
|	|--模型权重输出文件夹
|		|...
|	|...
|	|--bbox.json
|	|--image1.png
```

# 4 配置文件的说明

正是因为有配置文件的存在，我们才可以使用更便捷的进行消融实验。在本章节中我们选择
```configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.yml```文件来进行配置文件的详细解读

## 4.1 整体配置文件格式综述

我们将```mask_rcnn_r50_fpn_1x_coco.yml```进行拆分解释

* **mask_rcnn** 表示模型的名称
* **r50** 表示骨干网络采用resnet50的网络
* **fpn** 表示网络增加FPN的Neck网络
* **1x** 表示模型训练默认的批大小为1
* **coco** 表示模型配置文件默认基于coco数据集格式训练

**配置文件示例说明**

当前PaddleDetection为了降低配置冗余，将配置文件打散。要实现一个模型的训练，往往需要多个配置文件才可运行，如，我们现在选择的```mask_rcnn_r50_fpn_1x_coco.yml```，需要逐层依赖`../datasets/coco_instance.yml`、`_base_/mask_rcnn_r50_fpn.yml`、`_base_/optimizer_1x.yml` 、`_base_/mask_fpn_reader.yml`、`../runtime.yml`。

如果遇到相同的配置项，则直接使用的文件的地位最高，依赖文件越往后地位递减——即主配置文件优先级最高。

由于每个模型对于依赖文件的情况大致相同，因此以yolov3模型的所有配置文件展开示意图为例对本文所实验的模型进行说明:

<div align=center><img width="604" alt="image" src="docs.assets/image-20221115172216292.png"></div>

一个模型的配置文件按功能可以分为:

- **主配置文件入口**: mask_rcnn_r50_fpn_1x_coco.yml
- **定义训练数据路径的配置文件**: ../datasets/coco_instance.yml
- **定义公共参数的配置文件**: ../runtime.yml
- **定义优化器策略的配置文件**: \_base\_/optimizer_1x.yml
- **定义模型和主干网络的配置文件**: \_base\_/mask_rcnn_r50_fpn.yml
- **定义数据预处理方式的配置文件**: \_base\_/mask_fpn_reader.yml

## 4.2 数据路径与数据预处理说明

这一小节主要是说明数据部分，当准备好数据，如何进行配置文件修改，以及该部分的配置文件有什么内容。

**首先是进行数据路径配置**

数据为COCO格式实例分割数据集时，采用```../datasets/coco_instance.yml```配置, 并按照以下修改路径即可。

```yaml
metric: COCO # COCO数据集评价指标
num_classes: 1 # 数据的分类数(不包含类别背景)

TrainDataset: # 训练数据集配置
  !COCODataSet
    image_dir: JPEGImages # 3.图像数据文件夹
    anno_path: train.json # 2.标注记录的json
    dataset_dir: dataset/xiaoduxiong_ins_det # 1.数据集所在根目录
    data_fields: ['image', 'gt_bbox', 'gt_class', 'gt_poly', 'is_crowd']

EvalDataset: # 验证数据集配置
  !COCODataSet
    image_dir: JPEGImages # 3.图像数据文件夹
    anno_path: val.json # 2.标注记录的json
    dataset_dir: dataset/xiaoduxiong_ins_det # 1.数据集所在根目录

TestDataset: # 预测数据集配置
  !ImageFolder
    anno_path: annotations.json # 2.包含数据集全部类别信息的任意json标注文件
    dataset_dir: dataset/xiaoduxiong_ins_det # 1.数据集所在根目录
```

**note**

* 关于如何正确来写数据集的路径是非常关键的，可以根据上一章节训练的过程推演相对文件夹路径。

* ``num_classes``默认不包括背景类别。


**进一步地，可以根据需要配置数据预处理**

在mask_rcnn_r50_fpn_1x_coco.yml中采用的数据预处理配置文件为: \_base\_/mask_fpn_reader.yml

```yaml
worker_num: 2 # 每张GPU reader进程个数
TrainReader: # 训练读取器配置
  sample_transforms: # 采样预处理
  - Decode: {} # 读取图片并解码为图像数据
  - RandomResize: {target_size: [[640, 1333], [672, 1333], [704, 1333], [736, 1333], [768, 1333], [800, 1333]], interp: 2, keep_ratio: True} # 随机缩放
  - RandomFlip: {prob: 0.5} # 随机翻转，翻转概率为0.5
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]} # 图像归一化
  - Permute: {} # 通道提前: 保证为(C,H,W)
  batch_transforms: # 批量预处理
  - PadBatch: {pad_to_stride: 32} # 填充图像至宽高均保证被32整除
  batch_size: 1 # 训练的数据批大小
  shuffle: true # 随机打乱采集的数据顺序
  drop_last: true # 每个轮次数据采集中最后一次批数据采集数量小于batch_size时进行丢弃
  collate_batch: false # 表示reader是否对gt进行组batch的操作，在rcnn系列算法中设置为false，得到的gt格式为list[Tensor]
  use_shared_memory: true # 使用共享内存

EvalReader: # 评估读取器配置
  sample_transforms: # 采样预处理
  - Decode: {} # 读取图片并解码为图像数据
  - Resize: {interp: 2, target_size: [800, 1333], keep_ratio: True} # 图像缩放
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]} # 图像归一化
  - Permute: {} # 通道提前: 保证为(C,H,W)
  batch_transforms: # 批量预处理
  - PadBatch: {pad_to_stride: 32} # 填充图像至宽高均保证被32整除
  batch_size: 1 # 验证的数据批大小
  shuffle: false # 不随机打乱采集的数据顺序
  drop_last: false # 每个轮次数据采集中最后一次批数据采集数量小于batch_size时不进行丢弃

TestReader: # 预测读取器配置
  sample_transforms: # 采样预处理
  - Decode: {} # 读取图片并解码为图像数据
  - Resize: {interp: 2, target_size: [800, 1333], keep_ratio: True} # 图像缩放
  - NormalizeImage: {is_scale: true, mean: [0.485,0.456,0.406], std: [0.229, 0.224,0.225]} # 图像归一化
  - Permute: {} # 通道提前: 保证为(C,H,W)
  batch_transforms: # 批量预处理
  - PadBatch: {pad_to_stride: 32} # 填充图像至宽高均保证被32整除
  batch_size: 1 # 测试的数据批大小
  shuffle: false # 不随机打乱采集的数据顺序
  drop_last: false # 每个轮次数据采集中最后一次批数据采集数量小于batch_size时不进行丢弃
```

**note**

* 让模型评估或预测采用不同的大小，可以修改上述文件的以下位置参数大小来实现
  - 如: EvalReader->sample_transforms->Resize->:target_size: [800, 1333] 改成 target_size: [new_height, new_width]
  - 如: TestReader->sample_transforms->Resize->:target_size: [800, 1333] 改成 target_size: [new_height, new_width]
* PaddleDetection提供了多种数据增强的方式，可以通过访问[数据增强说明](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.5/docs/advanced_tutorials/READER.md#3.%E6%95%B0%E6%8D%AE%E9%A2%84%E5%A4%84%E7%90%86)来进行后续的修改。

## 4.3 模型说明

当我们配置好数据后，下面在看关于模型和主干网络的选择(位于\_base\_/mask_rcnn_r50_fpn.yml中)

```yaml
architecture: MaskRCNN # 模型架构名称
pretrain_weights: https://paddledet.bj.bcebos.com/models/pretrained/ResNet50_cos_pretrained.pdparams # 预训练模型

MaskRCNN: # 架构配置
  backbone: ResNet # 指定骨干网络
  neck: FPN # 指定neck网络的结构
  rpn_head: RPNHead # 指定区域提议的网络结构
  bbox_head: BBoxHead # 指定输出bbox的网络结构
  mask_head: MaskHead # 指定输出mask的网络结构
  # post process
  bbox_post_process: BBoxPostProcess # 指定bbox的后处理
  mask_post_process: MaskPostProcess # 指定mask的后处理

ResNet: # 骨干网络配置
  # index 0 stands for res2
  depth: 50 # 层数
  norm_type: bn # 归一化方式
  freeze_at: 0 # 冻结指定stage的网络参数
  return_idx: [0,1,2,3] # 返回指定的stage特征
  num_stages: 4 # stage数量

FPN: # Neck的FPN网络配置
  out_channel: 256 # 输出通道数

RPNHead: # 区域提议网络结构的配置
  anchor_generator: # anchor生成器的配置
    aspect_ratios: [0.5, 1.0, 2.0] # 生成时采用的多个宽高比
    anchor_sizes: [[32], [64], [128], [256], [512]] # 多个anchor的边小
    strides: [4, 8, 16, 32, 64] # 步长
  rpn_target_assign: # 提议目标分配的配置
    batch_size_per_im: 256
    fg_fraction: 0.5
    negative_overlap: 0.3
    positive_overlap: 0.7
    use_random: True
  train_proposal: # 训练时提议的配置
    min_size: 0.0
    nms_thresh: 0.7
    pre_nms_top_n: 2000
    post_nms_top_n: 1000
    topk_after_collect: True
  test_proposal: # 预测/测试/评估时提议的配置
    min_size: 0.0
    nms_thresh: 0.7
    pre_nms_top_n: 1000
    post_nms_top_n: 1000

BBoxHead: # 输出bbox网络结构的配置
  head: TwoFCHead # head类型
  roi_extractor: # roi区域提取器
    resolution: 7
    sampling_ratio: 0
    aligned: True
  bbox_assigner: BBoxAssigner # 指定bbox分配器

BBoxAssigner: # bbox分配器的配置
  batch_size_per_im: 512
  bg_thresh: 0.5 # 背景阈值
  fg_thresh: 0.5 # 前景阈值
  fg_fraction: 0.25 # 前景比例
  use_random: True# 是否随机采样

TwoFCHead:
  out_channel: 1024 # 输出通道数

BBoxPostProcess: # bbox的后处理配置
  decode: RCNNBox # 指定RCNN模型的box解码器
  nms: # 非极大值抑制配置
    name: MultiClassNMS
    keep_top_k: 100 # 经过NMS抑制后, 最终保留的最大检测框数量。如果设置为 -1 ，则保留全部
    score_threshold: 0.05 # 过滤掉低置信度分数的边界框的阈值
    nms_threshold: 0.5 # 过滤掉其它与置信度最高的边界框的IoU大于该阈值的候选框

MaskHead: # mask掩码的输出头
  head: MaskFeat # head的类型
  roi_extractor: # ROI区域的提取配置
    resolution: 14
    sampling_ratio: 0
    aligned: True
  mask_assigner: MaskAssigner # 指定mask掩码分配器
  share_bbox_feat: False # 共享bbox的特征

MaskFeat: # mask掩码特征模块配置
  num_convs: 4 # 卷积块数量
  out_channel: 256 # 输出通道数

MaskAssigner: # mask掩码分配器配置
  mask_resolution: 28

MaskPostProcess: # mask掩码后处理配置
  binary_thresh: 0.5 # 阈值
```

  **note**

* 我们模型的architecture是MaskRCNN

* 主干网络是 ResNet

* nms 此部分内容是预测与评估的后处理，一般可以根据需要调节threshold参数来优化处理效果。

  * 对于当前实验，由于数据集较小，训练好后可能需要调小nms_threshold来抑制更多的框，以此保持较好的检测效果:

  * ```yaml
    BBoxPostProcess:
      decode:
      nms:
        name: MultiClassNMS
        keep_top_k: 100
        score_threshold: 0.05
        nms_threshold: 0.3 # 0.5 --> 0.3
    ```

## 4.4 优化器说明

当我们配置好数据与模型后，下面再看关于优化器和损失函数的选择

```yaml
epoch: 12 # 训练轮次

LearningRate: # 学习率
  base_lr: 0.01 # 基础学习率大小——单卡GPU需要除以8, 即0.00125
  schedulers: # 学习率策略
  - !PiecewiseDecay # 分段衰减策略
    gamma: 0.1
    milestones: [8, 11] # 分段点
  - !LinearWarmup # 线性预热策略
    start_factor: 0.0
    steps: 1000 # 预热步数——当前数据集较小可以适当改为24，保证前两个轮次预热即可

OptimizerBuilder: # 优化器构建部分
  optimizer: # 优化器
    momentum: 0.9 # 动量大小
    type: Momentum # 优化器类型
  regularizer: # 正则配置
    factor: 0.0001 # 正则大小
    type: L2 # L2正则类型
```

## 4.5 其它参数说明

```yaml
_BASE_: [
  '../datasets/coco_instance.yml',
  '../runtime.yml',
  '_base_/optimizer_1x.yml',
  '_base_/mask_rcnn_r50_fpn.yml',
  '_base_/mask_fpn_reader.yml',
] # 基础依赖的配置文件

# 是否寻找没用上的模型参数
find_unused_parameters: True
# 是否使用ema平均
use_ema: true

...

use_gpu: true # 使用GPU
use_xpu: false # 使用xpu
log_iter: 20 # 日志输出间隔
save_dir: output # 模型文件保存路径
snapshot_epoch: 1 # 快照输出频率(包括模型保存与评估)
print_flops: false # 打印flops

export: # 模型导出的配置
  post_process: True  # 导出时保留后处理
  nms: True           # 后处理中保留nms
  benchmark: False    # 不按照benchmark标准导出
  fuse_conv_bn: False # 不采用fuse_conv_bn
```
