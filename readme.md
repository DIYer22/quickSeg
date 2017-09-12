# deep517：基于Python的图像分割脚手架+模型库

## Introduction
deep517：基于Python的图像分割脚手架+模型库

### 环境

* `Python 2.7`

* 依赖：`pip install -r requirment.txt`

* 系统：建议`ubuntu`


* 环境友好程度:`spyder@Anaconda2` > `Ipython` > `Python`

库中的绝大多数函数都有文档，
在Ipython中 大多数类都可以通过 对象名称后面加一个问号的形式 （即：`object?`） 获取使用说明文档

### 常用缩写：
    * img = image, 表示图片
    * gt = GroundTruth，用于评测的真值
    * re = resoult，模型生成的，需要评测的样本

## 文件结构

```
deep517
│
├── lib  # 存放函数库
│   ├── configManager.py  # 用于存放，处理参数及自动生成各类io函数
│   ├── yllibInterface.py # 引入 yl 文件夹的module
│   └── yl                # 杨磊的python代码库
│       ├── __init__.py
│       ├── tool/   # 常用Python工具module
│       ├── ylimg/  # 关于图片的module
│       ├── ylml/   # 关于机器学习的module
│       └── ylnp.py # 关于numpy的module
│ 
├── nets # 存放网络结构和接口
│   └── res-unet1 # 文件夹名即网络名称
│       ├── trainInterface.py  # 用于train的接口
│       └── trainInterface.py  # 用于predict的接口
│ 
├── projects # 工作空间 以项目名称为文件夹
│   └── projectTemplate # 项目模板 新建项目：复制并重命名文件夹为项目名称
│       └── experment1 # 实验文件夹 新建实验：复制重命名文件夹为实验名称
│           ├── lib.py     # 将 deep517/lib/ 下的所有moudlu导入并加入 sys.path
│           ├── config.py  # 配置训练集，测试集，预测集， 选择net
│           ├── train.py   # 配置训练的参数并训练
│           ├── test.py    # 使用训练好的模型对测试集进行预测
│           ├── val.py     # 用验证集进行细腻度的分析，深入了解模型性能
│           ├── weight/    # 存储训练过程中自动保存的权重
│           ├── test/      # 保存由测试集生成的结果
│           └── val/       # 存储验证时候Evalu产生的缓存数据和分析数据
│
└── readme.md # 此说明文件
```

## 面向网络使用者
1. 将`projects/projectTemplate` 文件夹复制并命名成你的项目名称 如：`COCO`
1. 在`projects/COCO` 中， 将`experment1` 文件夹重命名成你此次实验的名称如：`COCO_Unet`
1. 接下来 打开`projects/COCO/COCO_Unet`

### 关于数据
一般需要将样本分成独立的三部分：训练集(train set)，验证集(val set)和测试集(test set)。
* 训练集用来训练模型
* 验证集用来确定网络结构或者控制模型复杂程度的参数
* 测试集则检验最终选择最优的模型的性能如何

ps: 一个典型的划分是训练集占总样本的50％，而其它各占25％，三部分都是从样本中随机抽取。


### config.py

config.py主要配置数据接口 选择网络文件夹

在 config.py 的 `config BEGIN` 至 `config END` 块内进行配置


|配置名称 | 类型 | 描述 | 样例|
|:---|---|----|---|
|cf.netdir|            str | 要使用的网络所在文件夹的名称| res-unet1 |
|cf.trainGlob|         str | 训练集的图片以glob形式的路径| /data/COCO/train/*.jpg |
|cf.toGtPath|      function| 一个函数 传入图片路径，返回标签所在路径`cf.toGtPath(imgPath)=>gtPath` |lambda path:path.replace('.jpg','.png')|
|cf.val |      str or float|str:验证集的图片以glob形式的路径。float:从训练集中将文件后cf.val的图片划为验证集|val/*.jpg or 0.2|
|cf.toValGtPath | function | 同 cf.toGtPath，默认情况下 和 cf.toGtPath为同一函数|None|
|cf.testGlob|          str | 测试集的图片以glob形式的路径| /data/COCO/train/*.jpg |

### train.py

train.py主要配置train的参数

可使用 `train？` 或 `doc(train)`来查看当前网络的使用说明

在 train.py 的 `config BEGIN` 至 `config END` 块内进行配置

#### 常用参数项：

|参数名称 | 类型 | 描述 | 样例|
|:---|---|----|---|
|batch|int|训练每时刻的batch|4|
|epoch|int|训练多少epoch|30|
|resume|int|从第几个已保存的权重继续训练|0|
|classn|int|分割的结果包含的类别数|2|

#### 其他参数项：

|参数名称 | 类型 | 描述 | 样例|
|:---|---|----|---|
|window| int or (h,w)| 切割小图进行训练时候 切割窗口的大小|512 or (512,1024)



### test.py


### val.py


## 面向网络开发者

要更改网络结构或开发新的网络时，先将 `deep517/nets/netTemplate` 下的 `trainInterface.py` 和 `trainInterface.py` 复制到你的实验文件夹下,参考这两个接口进行修改和开发

开发完成后 只需在nets下新建文件夹 命名为你网络的名称即可 欢迎`git push origin master`提交你的net

### 常用train函数

#### lib.GenSimg
```
Init signature: lib.GenSimg(self, imggts, simgShape, handleImgGt=None, batch=1, cache=None, iters=None, timesPerRead=1, infinity=False)
Docstring:     
随机生成小图片simg及gt 的迭代器，默认使用1Gb内存作为图片缓存
默认生成simg总面积≈所有图像总面积时 即结束
Init docstring:
imggts: zip(jpgs,pngs)
simgShape: simg的shape
handleImgGt: 对输出结果运行handleImgGt(img,gt)处理后再返回
batch: 每次返回的batch个数
cache: 缓存图片数目, 默认缓存1Gb的数目
timesPerRead: 平均每次读的图片使用多少次(不会影响总迭代次数),默认1次
iters: 固定输出小图片的总数目，与batch无关
infinity: 无限迭代
File:           deep517/lib/yl/ylml/ylmlTrain.py
Type:           ABCMeta
```

### 常用predict函数

#### lib.autoSegmentWholeImg
```
Signature: lib.autoSegmentWholeImg(img, simgShape, handleSimg, step=None, weightCore=None)
Docstring:
将img分割到 simgShape 的小图，执行handleSimg(simg),将结果拼接成回img形状的矩阵
img:被执行的图片
simgShape: 小图片的shape
handleSimg: 用于处理小图片的函数 handleSimg(simg)，比如 net.pridict(simg)
step: 切割的步长, 默认为simgShape 可以为int|tuple(steph,stepw)|float
weightCore: 'avg'取平均,'gauss'结果的权重 在重叠部分可以用到
使之越靠经中心的权重越高 默认为直接覆盖
File:      deep517/lib/yl/ylml/ylmlTest.py
Type:      function
```
