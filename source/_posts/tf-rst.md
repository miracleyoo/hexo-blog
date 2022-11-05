---
title: tensorflow：理解 rank, shape, type
tags:
  - tensorflow
  - machine-learning
date: 2019-02-08 09:12:36
---


tensorflow 使用一种叫 tensor 的数据结构去展示所有的数据，我们可以把 tensor 看成是 n 维的 array 或者 list。在 tensorflow 的各部分图形间流动传递的只能是tensor。

### rank

rank 就是 tensor 的维数。
例如我们所说的标量(Scalar)：
`s = 8` 维数为 0，所以它的 rank 为 0。

例如矢量(Vector)：
`v = [1, 2, 3]`，rank 为 1。

例如矩阵(Matrix)：

```
m = [
  [1, 1, 1],
  [2, 2, 2],
  [3, 3, 3]
] # rank 为 2

```

又例如 rank 为 3 的 tensor：

```
t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]]

```

依次类推……

### shape

tensorflow 用 3 种方式描述一个 tensor 的维数：
rank, shape, 以及 dimension number (维数)
所以 shape 和 rank 的意思的一样的，只是表达的形式不同。

| rank | shape               | dimension |
| ---- | ------------------- | --------- |
| 0    | []                  | 0 维      |
| 1    | [D0]                | 1 维      |
| 2    | [D0, D1]            | 2 维      |
| n    | [D0, D1, ..., Dn-1] | n 维      |

shape 写成只包含整数的 list 或者 tuple 形式，例如 [1, 4, 2]

### data type

tensor 的数据结构除了 维数(dimensionality)，还有 数据类型(data type)。
例如 32位浮点数(32 bits floating point) 等等，可以从下面的链接中查看完整的：
[https://www.tensorflow.org/programmers_guide/dims_types#data_types](https://link.jianshu.com/?t=https://www.tensorflow.org/programmers_guide/dims_types#data_types)