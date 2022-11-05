---
title: KD-Tree的理解
tags: machine-learning
date: 2018-04-01 23:25:11
---


## 由下图可以大致看出KD-Tree的构造方式：

{% asset_img kd-tree.png KD-Tree 理解图%}

首先问题是隶属于分类问题的。每个sample有若干个属性（axis），如（3，4）就是一个有两个属性的sample。我们按axis=0，1，2，…的方式分别寻找每个维度（属性）的中位数并分别划分开来，就得到了一个树状结构，这样预测一个新的数据点的时候就可以很方便的按照树状结构将其归位到某个分区里去，而不用花费大量的计算资源去计算距离了。