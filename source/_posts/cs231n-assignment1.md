---
title: cs231n-assignment1
tags:
  - machine-learning
  - cs231n
  - CV
date: 2018-04-04 16:49:47
---


# 多分类SVM

* Loss的理解：对于每一个样本，先计算其W.dot(x)后分到每一个类的可能性，假设这个样本为j，一共c个分类，那么就要使用每个分错的可能性减去分到正确类的可能性。如果这个值为负数，则归零，之后把这(c-1)个值求和即为我们要求的Loss。

  $L_i=\sum_{j\neq y_i}max(0,s_j-s_{y_i}+1)$，而$L=\frac{1}{N}\sum^N_{i=1}L_i$ 

  这里的 $s_j-s_{y_i}+1$ 中的1其实是一个margin，表示如果 $s_j$ 只比 $s_{y_i}$ 小一点，就没有必要去惩罚这一项了。

  <!-- more -->

* Loss中的正则项（Regularization）：为了保证模型不要过拟合训练数据。正则项通常有L1正则项和L2正则项，它们的工作原理是把W矩阵中每一项的绝对值/平方加和，乘上一个常数$\lambda$ （正则化强度超参）后也放到Loss中，这样在降低loss的时候就会注意保证W中的数不要过大。

  $L=\frac{1}{N}\sum^N_{i=1}\sum_{j\neq y_i}max(0,f(x_i;W)_{y_i}+1)+\lambda R(W)$ 

* L2 regularization: $R(W)=\sum_k \sum_l W_{k,l}^2$ 

  L1 regularization: $R(W)=\sum_k \sum_l |W_{k,l}|$ 

  Elastic net 弹性网络（L1+L2）: $R(W)=\sum_k\sum_l\beta W^2_{k,l}+|W_{k,l}|$ 

  L1更倾向于选择值较小且较为稀疏的矩阵，而L2倾向于选择值较小且较平均的矩阵

* SVM的“Kernel”的意思是一个映射函数。SVM的核心思想是找一个超平面把正负样本分开，而在很多情况下，并不存在这样一个平面。于是我们用一个函数把这些数据点映射到一个新的容易分开的空间中，这时候找一个平面来分开

* SVM算法既可用于回归问题，比如SVR(Support Vector Regression, 支持向量回归）；也可以用于分类，比如SVC(Support Vector Classification,支持向量分类）

# 多项式Softmax

* Softmax的Loss计算： $L_i=-log(\frac{e^{sy_i}}{\sum_j e^{s_j}})$ 
* Softmax方程本身是：$P(Y=k|X=x_i)=\frac{e^{s_k}}{\sum_j e^{s_j}}$ 
* 由于各种可能的预测结果的softmax方程之和总为1，我们所需要做的只是不断优化target的值，其他不正确的score自然会降低的
* Softmax也可以被称为交叉熵loss，因为信息熵的定义是 $-log(P_i)$ ，其中$P_i$ 是概率

# 优化方式

1. 随机搜索（Random Search）：不断的给W随意的赋值，Loss表现更低的话就把W换成这次赋值得到的W。
2. 顺斜率搜索（Follow the slope）：$\frac{df(x)}{dx}=lim_{h \to \infty} \frac{f(x+h)-f(x)}{h}$
3. 我们把Loss看做是W的函数，而我们要做的就是不断寻找在每个W点上L的梯度的值和方向，并且不断沿着梯度方向下降一段距离，这段距离就是学习率。
4. 随机梯度下降（Stochastic Gradient Descent，SGD）：如果每个样本都算一次梯度，结果往往会有较大的随机性，而如果用一个小的batch作为基本单位，随机性就会减少不少
5. 学习率如果太大容易引起振荡，很难收敛，而太小则速度太慢。往往会在最初步长大一些，后面慢慢收敛。

# 实践部分

- `W.dot(x)`的结果与`np.dot(W,x)`是相同的，都是W和x的矩阵相乘
- 实践中，在涉及到学习优化等和 $\lambda$ 等带有学习速率之类超参数的问题时，常在前面乘上一个0.5，因为平方项求导的时候会出来一个2，0.5$\lambda$ 就比较好。

# 总结

* Ground-truth: 真实标签，X的真实分类等意思，可以略作GT
* 计算机视觉中的Bag of Words里面的words说的是一些底层的pattern，经常性的重复出现的模式，把这些当做“视觉词汇”来进行处理