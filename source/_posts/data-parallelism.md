---
title: Pytorch 的 Data Parallelism 多GPU训练
tags:
  - machine-learning
  - pytorch
date: 2018-08-22 21:23:05
---


## 简单步骤

1. 确定Device，看是否有可利用的GPU：`device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`

2. 正常定义并实例化模型和Dataloader

3. 如果检测到的GPU多于一块，将模型并行化：

   <!-- more -->

   ```python
   if torch.cuda.device_count() > 1:
     print("Let's use", torch.cuda.device_count(), "GPUs!")
     model = nn.DataParallel(model)
   ```

4. 将模型部署到相应的设备上：`model.to(device)`

5. 运行模型（循环中将Input数据也加载到相应设备上）：

   ```python
   for data in rand_loader:
       input = data.to(device)
       output = model(input)
       print("Outside: input size", input.size(),
             "output_size", output.size())
   ```

6. 得到结果：

   ```python
   	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
   	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
   Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
   	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
   	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
   Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
   	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
   	In Model: input size torch.Size([15, 5]) output size torch.Size([15, 2])
   Outside: input size torch.Size([30, 5]) output_size torch.Size([30, 2])
   	In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
   	In Model: input size torch.Size([5, 5]) output size torch.Size([5, 2])
   Outside: input size torch.Size([10, 5]) output_size torch.Size([10, 2])
   ```
