---
title: Pytorch 自动生成网络结构图
tags:
  - pytorch
  - machine-learning
  - visualization
date: 2018-08-11 00:14:30
---


# Pytorch 自动生成网络结构图

Pytorch没有TensorFlow类似的原生支持，尽管也可以用TensorBoardX进行可视化，但是其生成的网络图不适合直接展示。所以，有没有合适的自动化方案？

首先，可以利用一个开源库，[functional-zoo](https://github.com/szagoruyko/functional-zoo)

## 所需准备：

1. `brew/apt-get/yum install graphviz`
2.  `conda install python-graphviz`
3. `pip install torchviz`

以我自己的网络为例：

```python
import torch
from torch.autograd import Variable
from graphviz import Digraph
from torchviz import make_dot

from models import miracle_lineconv_net
from config import Config
opt = Config()

x = Variable(torch.randn(128,2,41,9))#change 12 to the channel number of network input
model = miracle_lineconv_net.MiracleLineConvNet(opt)
y = model(x)
g = make_dot(y)
g.view()
```

+ 如果依然有安装问题，继续下载 `pip install git+https://github.com/szagoruyko/pytorchviz`

## 官方效果如下图：

![image-20180811001334071](0069RVTdly1fu5opw86r4j30nq0ou416.jpg)