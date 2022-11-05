---
title: Pytorch官方模型实现分析
tags:
  - deep-learning
  - pytorch
date: 2020-07-11 13:08:17
---


## Resnet

### 如何应对不同尺寸输入

在网络最后添加一个`AdaptiveAvgPool2d(output_size)`函数，它的作用是无论输入图片形状如何，最终都会转换为给定输出尺寸。而Resnet中，这个`output_size`被设置为了`(1,1)`，即无论输入的图片尺寸为多少，只要其大小足以扛得住网络前面的一系列pooling layers，到最后的输出尺寸大于等于`(1,1)`，其每个Channel就会被这一层压缩成一个点，即最后只会得到一个与Channel数目相等的向量。这个向量被送到了FC层。

<!-- more -->

### 如何应对Channel数目不合适问题

由于兼容了各种大小和尺寸的模型，所以有时难免会出现如Channel数目无法被4整除（BottleNeck Layer要求）的情况。这里，它使用了`1x1 conv`的方法。由于每个block的输出都要加到输入上，如果这个整除不了，结果就是Channel数目不匹配无法做Residual。这里的操作是，如果输出`Channel数*4 != 输入Channel数`，那么就直接让输入先用`1x1 conv`改变维度到`输出Channel数*4`。

```python
if stride != 1 or self.inplanes != planes * block.expansion:
    downsample = nn.Sequential(
        conv1x1(self.inplanes, planes * block.expansion, stride),
        norm_layer(planes * block.expansion),
    )
```

### 如何应对不同总Layer数的Resnet使用不同的Layer Block的问题

在Resnet中，resnet18, 34使用的是双层`3x3 conv`的`Basic Block`，而resnet50, 101, 152则使用的是`1x1conv -> 3x3 conv -> 1x1 conv`的结构。为了获得最大的兼容性，这里官方模型将`Basic Block`和`Bottleneck Block`分别定义为两个class，即子模块，然后对于不同尺寸的resnet分别输入不同的模块。

### 为什么定义了conv3x3和conv1x1两个函数

这两个函数看似画蛇添足多此一举，但实际上在我的理解中，他们避免了一些重复变量的输入，更直观地反映了该层的功能：`1x1`或`3x3`，也潜在地避免了一些错误，并优化了理解。