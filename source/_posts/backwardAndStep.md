---
title: How are optimizer.step() and loss.backward() related?
tags:
  - machine-learning
  - pytorch
date: 2018-04-11 00:45:42
---


今天和同学讨论的时候发现这个地方知识存在漏洞，赶紧补了一波。问题就是optimizer.step() 和 loss.backward()这两个总是出现在一起的两个pytorch函数各自执行的功能。

## 简单的说，

* loss.backward()根据这一轮的loss计算出了网络中所有需要计算的导数
* optimizer.step()根据你选择的优化器，使用上面👆loss.backward()计算出的各个导数更新了网络中的各个权值

<!-- more -->

## 以下为论坛原文：

`loss.backward()` computes `dloss/dx` for every parameter `x` which has `requires_grad=True`. These are accumulated into `x.grad` for every parameter `x`. In pseudo-code:

```
x.grad += dloss/dx
```

`optimizer.step` updates the value of `x` using the gradient `x.grad`. For example, the SGD optimizer performs:

```
x += -lr * x.grad
```

`optimizer.zero_grad()` clears `x.grad` for every parameter `x` in the optimizer. It’s important to call this before `loss.backward()`, otherwise you’ll accumulate the gradients from multiple passes.

If you have multiple losses (loss1, loss2) you can sum them and then call backwards once:

```
loss3 = loss1 + loss2
loss3.backward()
```