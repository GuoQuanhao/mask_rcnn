# mask_rcnn
tensorflow mask rcnn
@[TOC]
这是一篇拖了快半年的博客(○´･д･)ﾉ
![mask](https://img-blog.csdnimg.cn/20200320143449625.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70#pic_center)
**TensorFlow 是目前应用最广泛的深度学习框架，除了提供 faster rcnn，同样提供 mask rcnn，利用 TensorFlow Models 可以快速搭建自己的 mask rcnn 模型**

# 一、数据文件准备
## 1.数据文件下载
本次博客打算以“人”这个类别为例，所以我们需要大量含有“人”的图片，通过 Python 的爬虫方式，可以快速爬取大量图片
python 爬虫源码文件为 image_gather.py，运行方式为在此源码的同级目录下新建一个 name.txt 文件，里面写入你想要下载的图片名称，我以“美女”为例，然后运行如下命令
```powershell
输入你需要下载的数量，我输入为 20
```
效果如下
![文件](https://img-blog.csdnimg.cn/20200320144907331.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70#pic_center)
## 2.数据文件命名规范
下载好图片文件后，检查有没有不能打开的图片，然后对文件夹与文件名重命名等，美女文件如下
![文件](https://img-blog.csdnimg.cn/20200320150641150.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
正儿八经的数据文件如下
![命名](https://img-blog.csdnimg.cn/20200320145153194.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
关于快速重命名的方法查看这里 N[O.5 Tensorflow在win10下实现object detection](https://blog.csdn.net/qq_39567427/article/details/102712994)
# 二、数据集制作
## 1、数据文件分类
将文件分为两类：train，test
![分类](https://img-blog.csdnimg.cn/20200320150954388.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
## 2、标签框图
打开 labelme
选择 OpenDir 定位到自己的文件夹→Creat Polygon 就可以开始框选
![labelme](https://img-blog.csdnimg.cn/20200320145959518.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
框选完保存为 person 标签，演示我框的比较简单，可以利用鼠标滚轮放大再框选
![保存](https://img-blog.csdnimg.cn/20200320150244168.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
有关labelme的安装与使用，详细在这里 [NO.3 Tensorflow在win10下实现object detection](https://blog.csdn.net/qq_39567427/article/details/102596678)

当全部完成后文件如下，每一张图片都有自己对应的 json 文件，json文件里面存储了标签与你框图时每一个点的坐标
![文件](https://img-blog.csdnimg.cn/20200320151109927.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
创建 labelmap.pbtxt，具体创建的细节参考 [NO.5 Tensorflow在win10下实现object detection](https://blog.csdn.net/qq_39567427/article/details/102712994)

```powershell
item {
  id: 1
  name: 'person'
}
```

## 3、数据集生成
首先需要将 json 文件与图片文件放在不同文件夹下，如下

```powershell
test #里面是test原图片
test_json #里面是test的json文件
train #里面是train原图片
train_json #里面是train的json文件
```
![json](https://img-blog.csdnimg.cn/20200320153025862.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70#pic_center)
然后需要如下三个文件将数据集转为 tfrecord 形式
create_tf_record.py
最后文件列表应如下，raw 为下载的原图片文件，未经任何分类，images 为已进行分类并上标签图像
![文件](https://img-blog.csdnimg.cn/20200320153743444.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70#pic_center)
raw 与 images 内文件列表如下，**请忽略 segmentation 这是后面的选修操作**😀，**train.record 为下面命令行运行生成的 record 文件**
![文件](https://img-blog.csdnimg.cn/20200320154116111.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
运行如下命令，分别对 train，test 进行，你将会得到 train.record，test.record

```powershell
python create_tf_record.py --images_dir=images/train --annotations_json_dir=images/train_json --label_map_path=labelmap.pbtxt --output_path=images/train.record
```
![分类](https://img-blog.csdnimg.cn/20200320152652324.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
# 三、训练与部署
## 1.下载预训练模型
下载地址：[model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
找到 mask_rcnn 的预训练模型，根据自己需要选择一个即可，下载解压即可
![zoo](https://img-blog.csdnimg.cn/20200320154722393.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
## 2.创建mask rcnn config文件
参考 [NO.5 Tensorflow在win10下实现object detection](https://blog.csdn.net/qq_39567427/article/details/102712994) 创建 faster rcnn config类似，你只需要从官方给定的 config 文件选择符合你想训练模型的mask config就行，参数设置方式同样参考，最后以我的为例
mask_rcnn_inception_v2_coco.config
```
你只需要将 [NO.5 Tensorflow在win10下实现object detection](https://blog.csdn.net/qq_39567427/article/details/102712994) 这篇博客的 tfrecord 文件分别对应替换为你的 tfrecord 文件，替换 config 文件等，如果你成功操作了  faster rcnn 的部署，我想这篇博客会很容易实现

```powershell
train.record → train.record
validation.record → test.record
```
模型训练

```python
!python train.py --train_dir training/ --pipeline_config_path mask_rcnn_inception_v2_coco_2018_01_28.config
```
模型冻结

```powershell
!python export_inference_graph.py \
--input_type image_tensor \
--pipeline_config_path mask_rcnn_inception_v2_coco_2018_01_28.config \
--trained_checkpoint_prefix training/model.ckpt-500000 \
--output_directory export/
```

# 选修操作
## 1.segmentation.py
这个文件是配合 labelme_json_to_dataset.exe 一起使用的，创建了一个 optional 文件夹，里面存放了 test 的图片文件以及 json 文件，来说明 segmentation.py 的功能，其功能是对图像进行语义分割，path_file_name 名称根据自己文件夹修改，你可以对 test 与 train 都这样操作
segmentation.py
运行命令

```powershell
python segmentation.py
```
效果如下，在 optional 文件夹有 50 个文件夹，数量等于 test 图片数，每一个文件夹下有 4 个文件
![文件](https://img-blog.csdnimg.cn/2020032016084764.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
## 2.classification.py
作用为对 1 中生成的文件进行分类，我们将 1 中生成的文件夹放在

```powershell
images/segmentation
```
下，里面有 test 的，也有 train 的
![文件](https://img-blog.csdnimg.cn/2020032016142656.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
classification.py
运行命令如下：
label.png 还可以为 label_viz.png 等，修改源码文件夹实现分别对 test，train操作，你甚至可以重写一个 argparse.ArgumentParser() 来变得更人性化

```powershell
python classification.py --classification label.png #修改源码test、train
```
结果如下，会自动创建去掉分类文件后缀名的文件夹
![文件](https://img-blog.csdnimg.cn/20200320162138257.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
label_viz.png 如下
![文件](https://img-blog.csdnimg.cn/20200320162217803.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
# 四、模型调用与实现
模型的训练，冻结都参考博客 [NO.6 Tensorflow在win10下实现object detection](https://blog.csdn.net/qq_39567427/article/details/102800400) 描述很清楚，并且有很清楚的操作方式，具体可以参考 Tensorflow.ipynb，我将会在最后给出我的 github 地址，我的模型是训练的 500000 步，有关调用的代码同样参考上述博客，你只需要修改文件中一小部分路径，模型名称即可，我对视频进行了识别，放两张截图
![video](https://img-blog.csdnimg.cn/20200320163618706.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
![video](https://img-blog.csdnimg.cn/2020032016372679.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzM5NTY3NDI3,size_16,color_FFFFFF,t_70)
视频识别源码
**到此，基于 TensorFlow 的mask rcnn 就全部结束了**

# 五、参考
我的 [blog](https://blog.csdn.net/qq_39567427/article/details/104989739)

