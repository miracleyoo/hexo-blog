---
title: PyTorch中使用指定的GPU
tags:
  - machine-learning
  - pytorch
  - server
date: 2018-05-14 23:41:40
---


[[转]PyTorch中使用指定的GPU](http://www.cnblogs.com/darkknightzh/p/6836568.html)

PyTorch默认使用从0开始的GPU，如果GPU0正在运行程序，需要指定其他GPU。

<!-- more -->

有如下两种方法来指定需要使用的GPU。

1. 类似tensorflow指定GPU的方式，使用CUDA_VISIBLE_DEVICES。

   1.1 直接终端中设定：

   `CUDA_VISIBLE_DEVICES=1 python my_script.py`

​	1.2 python代码中设定：

​	`import os os.environ["CUDA_VISIBLE_DEVICES"] = "2"`

​	见网址：<http://www.cnblogs.com/darkknightzh/p/6591923.html>

2. 使用函数 set_device

   `import torch torch.cuda.set_device(id)`

   该函数见 pytorch-master\torch\cuda\__init__.py。

   不过官方建议使用CUDA_VISIBLE_DEVICES，不建议使用 set_device 函数。