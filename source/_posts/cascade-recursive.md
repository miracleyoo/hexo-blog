---
title: Cascade, Recursive, Residual and Dense 辨析
tags:
  - deep-learning
date: 2020-08-05 17:00:45
---

## Cascade

相当于Progressive Optimization，每个阶段都会输出一个和最终结果形状相同的Matrix，如目标分布的图像、BBox等，然后下一个block的作用则是输入这个Matrix和前面提取的Features，输出Refined后的Matrix，该步骤不断重复。核心是逐步优化。

<!-- more -->

## Recursive

相当于把一块Conv block重复了好多次，每次的权重是共享的。核心作用是节省内存和参数量、节省运算时间。同时它也含有时域特征。

## Residual

保留一条“信息高速公路”，使得前一轮的输出可以直接点加到经过了新一轮的Block卷积过后的结果上。核心作用是解决梯度消失问题，同时在网络的各层保留了不同层级的信息。变形有如Residual in Residual。

![img](v2-862e1c2dcb24f10d264544190ad38142_1440w.jpg)

> ResNet网络的短路连接机制（其中+代表的是元素级相加操作）

## Dense

每个Conv Block的输出会在Channel维上和后面所有Conv Block的输出Concate到一起。注意和Residual结构的区别，前者是直接逐点相加，而Dense则是并到Channel维度上。

![img](v2-2cb01c1c9a217e56c72f4c24096fe3fe_1440w.jpg)

> DenseNet网络的密集连接机制（其中c代表的是channel级连接操作）

![img](v2-0a9db078f505b469973974aee9c27605_1440w.jpg)

> DenseNet的前向过程

![img](v2-c81da515c8fa9796601fde82e4d36f61_1440w.jpg)

> 原图