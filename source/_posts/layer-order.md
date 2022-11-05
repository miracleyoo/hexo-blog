---
title: 主要神经网络layer的合理排布顺序
tags:
  - machine-learning
  - neural-network
date: 2018-08-21 11:47:23
---


### 开篇名义，先把普遍的神经网络层的排布顺序展示出来：



```
Enter: Convlution -> Batch Normalization -> Relu -> Max Pool
Middle: Convlution -> Batch Normalization -> Relu
Middle Complicated: -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->
Tail: Max Pool -> View -> (Dropout) -> Fc
```

<!-- more -->

### 关于 Max Pool 和 Relu 的相对位置

> If you consider the final result, both orders [conv -> relu -> max pooling] and [conv -> max pooling -> relu] will have the same outputs. But if you compare the running time of 2 ways there will be a difference.
>
> To simplify the problem, just ignore layer convolution layer because it is the same in both cases.
>
> Relu layer don’t change the size of the input. Let assume we use max pooling 2x2, so the size of input will be reduce by 2 in height and width when apply max poling layer ( [w, h, d] -> max_pooling_2x2 -> [w/2, h/2, d]).
>
> **In case 1** we using relu -> max pooling the size of data will be:
>
> **image[w, h, d] -> [[relu]]** ->*image[w, h, d]->[[max pooling]]* -> image[w/2, h/2, d]
>
> **In case 2** we using max pooling -> relu the size of data will be:
>
> *image[w, h, d] ->[[max pooling]]* -> **image[w/2, h/2, d]-> [[relu]]** -> image[w/2, h/2, d]
>
> **image[w, h, d] -> [[relu]]** vs **image[w/2, h/2, d]-> [[relu]] :** case 2 save 4 time computational cost than case 1 in layer [[relu]] by using max pooling before relu.
>
> In conclusion, you can save a lot of running time if you put max pooling before the non-linear layers like relu or sigmoid.

### 关于 Max Pool 和 Dropout 的相对位置

> **Edit:** As @Toke Faurby correctly pointed out, the default implementation in tensorflow actually uses an element-wise dropout. What I described earlier applies to a specific variant of dropout in CNNs, called [spatial dropout](https://arxiv.org/pdf/1411.4280.pdf):
>
> In a CNN, each neuron produces one feature map. Since ~~dropout~~ **spatial dropout** works per-neuron, dropping a neuron means that the corresponding feature map is dropped - e.g. each position has the same value (usually 0). So each feature map is either fully dropped or not dropped at all. 
>
> Pooling usually operates separately on each feature map, so it should not make any difference if you apply dropout before or after pooling. At least this is the case for pooling operations like maxpooling or averaging.
>
> **Edit:** However, if you actually use **element-wise** dropout (which seems to be set as default for tensorflow), it actually makes a difference if you apply dropout before or after pooling. However, there is not necessarily a *wrong* way of doing it. Consider the average pooling operation: if you apply dropout before pooling, you effectively scale the resulting neuron activations by `1.0 - dropout_probability`, but most neurons will be non-zero (in general). If you apply dropout after average pooling, you generally end up with a fraction of `(1.0 - dropout_probability)` non-zero "unscaled" neuron activations and a fraction of `dropout_probability` zero neurons. Both seems viable to me, neither is outright wrong.

### 关于 BN 和 Dropout 的相对位置

> In the [Ioffe and Szegedy 2015](https://arxiv.org/pdf/1502.03167.pdf), the authors state that "we would like to ensure that for any parameter values, the network always produces activations with the desired distribution". So the Batch Normalization Layer is actually inserted right after a Conv Layer/Fully Connected Layer, but before feeding into ReLu (or any other kinds of) activation. See [this video](https://www.youtube.com/watch?v=jhUZ800C650&index=5&list=PLLvH2FwAQhnpj1WEB-jHmPuUeQ8mX-XXG) at around time 53 min for more details.
>
> As far as dropout goes, I believe dropout is applied after activation layer. In the [dropout paper](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf) figure 3b, the dropout factor/probability matrix r(l) for hidden layer l is applied to it on y(l), where y(l) is the result after applying activation function f. 
>
> So in summary, the order of using batch normalization and dropout is:
>
> -> CONV/FC -> BatchNorm -> ReLu(or other activation) -> Dropout -> CONV/FC ->

