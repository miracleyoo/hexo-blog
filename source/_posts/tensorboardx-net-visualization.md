---
title: Pytorch利用Tensorboardx进行网络结构可视化
tags:
  - pytorch
  - machine-learning
  - visualization
date: 2018-08-18 18:00:16
---


之前提到过利用`python-graphviz`进行自动网络可视化，尽管其较为适合展示，但是Tensorboard生成的网络结构图有着可折叠、便于调试的优点，那Pytorch可以使用这项功能吗？

答案是肯定的，tensorboardx提供了这项支持。

首先提供相关资源：

## 资源与下载

1. tensorboardx的安装：`pip install tensorboardx`

2. tensorboardx画网络图的实例：[实例](https://github.com/lanpa/tensorboardX/blob/master/examples/demo_graph.py)

3. 结果展示：

   ![image-20180818175931766](006tNbRwgy1fuemv60tllj30ba0isn4w.jpg)

## 基础操作

```python
dummy_input = Variable(torch.rand(13, 1, 28, 28))
model = Net1()
with SummaryWriter(comment='Net1') as w:
    w.add_graph(model, (dummy_input, ))
```

## WARNING

如果需要可视化网络结构图，一定要让这一步操作放在把net加载为CUDA模型之前进行！