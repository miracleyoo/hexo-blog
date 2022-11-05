---
title: 全连接层与1x1卷积的关系
tags:
  - machine-learning
date: 2019-02-12 10:24:16
---

## 理解全连接层：

**全连接层可以由卷积核深度为上层特征图深度的卷积运算代替，卷积后channel上每个1x1卷积核都会作为一个输出，可看作全连接层的一个点。**

假设最后一个卷积层的输出为7×7×512，连接此卷积层的全连接层为1×1×4096。

如果将这个全连接层转化为卷积层：
1. 共有4096组滤波器（**通道数4096**）
2. 每个卷积核的大小为**512x7×7**
3. 则输出为**1×1×4096**

由于每个滤波核的大小和上一层的feature map大小一样，保证了转换后的卷积层的运算结果和全连接层是一样的。
若后面再连接一个1×1×4096全连接层。则其对应的转换后的卷积层的参数为：

1. 共有4096组滤波器（**通道数4096**）
2. 每个卷积核的大小为**4096x1x1**
3. 则输出为**1×1×4096**

## 举例说明

参考“ [RCNN系列目标检测详解](https://bbs.dian.org.cn/topic/589/rcnn%E7%B3%BB%E5%88%97%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B%E8%AF%A6%E8%A7%A3)”，在Faster RCNN中，RPN网络的关键部分就是由两个1x1的卷积核完成的，其作用即为全连接。
![Network Structure](006y8mN6ly1g7lwj8tmefj311609y0tp.jpg)