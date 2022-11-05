---
title: 常用的模式识别中的图像特征介绍
tags:
  - machine-learning
  - image-processing
  - CV
date: 2018-04-02 17:22:26
---


## [局部二值模式（英文：Local binary patterns，缩写：LBP）](https://blog.csdn.net/u013207865/article/details/49720509)

在最简简化的情况下，局部二值模式特征向量可以通过如下方式计算：

- 将检测窗口切分为区块（cells，例如，每个区块16x16像素）。

- 对区块中的每个像素，与它的八个邻域像素进行比较（左上、左中、左下、右上等）。可以按照顺时针或者逆时针的顺序进行比较。

- 对于中心像素大于某个邻域的，设置为1；否则，设置为0。这就获得了一个8位的二进制数（通常情况下会转换为十进制数字），作为该位置的特征。

  <!-- more -->

- 对每一个区块计算直方图。

- 此时，可以选择将直方图归一化；

- 串联所有区块的直方图，这就得到了当前检测窗口的特征向量。

- Python实现库函数：[请点这里](http://scikit-image.org/docs/stable/api/skimage.feature.html#local-binary-pattern)

{% asset_img 20131025114220937.png 局部二值模式%}

## [方向梯度直方图（英语：Histogram of oriented gradient，简称HOG）](https://www.jianshu.com/p/395f0582c5f7)

* **方向梯度直方图**是应用在[计算机视觉](https://zh.wikipedia.org/wiki/%E8%AE%A1%E7%AE%97%E6%9C%BA%E8%A7%86%E8%A7%89)和[图像处理](https://zh.wikipedia.org/wiki/%E5%9B%BE%E5%83%8F%E5%A4%84%E7%90%86)领域，用于[目标检测](https://zh.wikipedia.org/w/index.php?title=%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B&action=edit&redlink=1)的特征描述器。这项技术是用来计算局部图像梯度的方向信息的统计值。这种方法跟[边缘方向直方图](https://zh.wikipedia.org/w/index.php?title=%E8%BE%B9%E7%BC%98%E6%96%B9%E5%90%91%E7%9B%B4%E6%96%B9%E5%9B%BE&action=edit&redlink=1)（edge orientation histograms）、[尺度不变特征变换](https://zh.wikipedia.org/wiki/%E5%B0%BA%E5%BA%A6%E4%B8%8D%E5%8F%98%E7%89%B9%E5%BE%81%E5%8F%98%E6%8D%A2)（scale-invariant feature transform descriptors）以及[形状上下文方法](https://zh.wikipedia.org/w/index.php?title=%E5%BD%A2%E7%8A%B6%E4%B8%8A%E4%B8%8B%E6%96%87%E6%96%B9%E6%B3%95&action=edit&redlink=1)（ shape contexts）有很多相似之处，但与它们的不同点是：HOG描述器是在一个网格密集的大小统一的细胞单元（dense grid of uniformly spaced cells）上计算，而且为了提高性能，还采用了重叠的局部对比度归一化（overlapping local contrast normalization）技术
* HOG描述器最重要的思想是：在一副图像中，局部目标的表象和形状（appearance and shape）能够被梯度或边缘的方向密度分布很好地描述。具体的实现方法是：首先将图像分成小的连通区域，我们把它叫细胞单元。然后采集细胞单元中各像素点的梯度的或边缘的方向直方图。最后把这些直方图组合起来就可以构成特征描述器。为了提高性能，我们还可以把这些局部直方图在图像的更大的范围内（我们把它叫区间或block）进行对比度归一化（contrast-normalized），所采用的方法是：先计算各直方图在这个区间（block）中的密度，然后根据这个密度对区间中的各个细胞单元做归一化。通过这个归一化后，能对光照变化和阴影获得更好的效果。
* 与其他的特征描述方法相比，HOG描述器有很多优点。首先，由于HOG方法是在图像的局部细胞单元上操作，所以它对图像几何的（geometric）和光学的（photometric）形变都能保持很好的不变性，这两种形变只会出现在更大的空间领域上。其次，作者通过实验发现，在粗的空域抽样（coarse spatial sampling）、精细的方向抽样（fine orientation sampling）以及较强的局部光学归一化（strong local photometric normalization）等条件下，只要行人大体上能够保持直立的姿势，就容许行人有一些细微的肢体动作，这些细微的动作可以被忽略而不影响检测效果。综上所述，HOG方法是特别适合于做图像中的行人检测的。
* Python实现库函数：[请点这里](http://scikit-image.org/docs/stable/api/skimage.feature.html#hog)

{% asset_img v2-890c6f08045598e83c90f2d52b946c17_hd.jpg 梯度直方图%}

{% asset_img v2-f356313f5806fdaaf59ec9196af353b7_hd.jpg 8*8网格直方图%}

{% asset_img v2-802e88923e7e26459250d31086e033ea_hd.jpg visualizing_histogram%}

## [尺度不变特征转换(Scale-invariant feature transform 或 SIFT)](https://blog.csdn.net/zddblog/article/details/7521424)

* 尺度不变特征转换(Scale-invariant feature transform或SIFT)是一种电脑视觉的算法用来侦测与描述影像中的局部性特征，它在空间尺度中寻找极值点，并提取出其位置、尺度、旋转不变量，此算法由 David Lowe在1999年所发表，2004年完善总结。其应用范围包含物体辨识、机器人地图感知与导航、影像缝合、3D模型建立、手势辨识、影像追踪和动作比对。局部影像特征的描述与侦测可以帮助辨识物体，SIFT 特征是基于物体上的一些局部外观的兴趣点而与影像的大小和旋转无关。对于光线、噪声、些微视角改变的容忍度也相当高。基于这些特性，它们是高度显著而且相对容易撷取，在母数庞大的特征数据库中，很容易辨识物体而且鲜有误认。使用 SIFT特征描述对于部分物体遮蔽的侦测率也相当高，甚至只需要3个以上的SIFT物体特征就足以计算出位置与方位。在现今的电脑硬件速度下和小型的特征数据库条件下，辨识速度可接近即时运算。SIFT特征的信息量大，适合在海量数据库中快速准确匹配。

* SIFT算法的特点有：

  1. SIFT特征是图像的局部特征，其对旋转、尺度缩放、亮度变化保持不变性，对视角变化、仿射变换、噪声也保持一定程度的稳定性；


  2. 独特性（Distinctiveness）好，信息量丰富，适用于在海量特征数据库中进行快速、准确的匹配；


  3. 多量性，即使少数的几个物体也可以产生大量的SIFT特征向量；


  4. 高速性，经优化的SIFT匹配算法甚至可以达到实时的要求；

  5. 可扩展性，可以很方便的与其他形式的特征向量进行联合。

* SIFT算法可以解决的问题：

  目标的自身状态、场景所处的环境和成像器材的成像特性等因素影响图像配准/目标识别跟踪的性能。而SIFT算法在一定程度上可解决：

  1. 目标的旋转、缩放、平移（RST）


  2. 图像仿射/投影变换（视点viewpoint）


  3. 光照影响（illumination）


  4. 目标遮挡（occlusion）


  5. 杂物场景（clutter）


  6. 噪声

* Lowe将SIFT算法分解为如下四步：

  1. 尺度空间极值检测：搜索所有尺度上的图像位置。通过高斯微分函数来识别潜在的对于尺度和旋转不变的兴趣点。


  2. 关键点定位：在每个候选的位置上，通过一个拟合精细的模型来确定位置和尺度。关键点的选择依据于它们的稳定程度。


  3. 方向确定：基于图像局部的梯度方向，分配给每个关键点位置一个或多个方向。所有后面的对图像数据的操作都相对于关键点的方向、尺度和位置进行变换，从而提供对于这些变换的不变性。


  4. 关键点描述：在每个关键点周围的邻域内，在选定的尺度上测量图像局部的梯度。这些梯度被变换成一种表示，这种表示允许比较大的局部形状的变形和光照变化。

* Python实现库函数：[请点这里](http://scikit-image.org/docs/stable/api/skimage.feature.html#daisy) （这个实现有近似成分）

## 以上内容的Python实现 Github 地址：[请点这里](https://github.com/miracleyou/cs231n_assignment_HUST/blob/master/cs231n_assignment1_HUST.ipynb)

