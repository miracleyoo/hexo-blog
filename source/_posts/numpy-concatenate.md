---
title: Numpy矩阵拼接数据集
tags:
  - python
  - numpy
date: 2018-08-18 20:09:15
---


机器学习中，有时候需要自己生成含有n个channel、size相同的输入数据。这个时候就需要进行numpy的拼接操作了。

### 先看示例代码：

```python
def gen_input(Vn, Vp, nx1, ny1, net_charge):
    net_charge = np.array(net_charge)[np.newaxis, :]
    border_cond = np.zeros((1, int(ny1), int(nx1)))
    border_cond[0, :, 0] = Vn
    border_cond[0, :, -1] = Vp
    model_input = np.concatenate((net_charge, border_cond), axis=0)
    return model_input[np.newaxis, :]
```

上面的net_charge和border_cond即为需要进行拼接操作的两个matrix，分别使用了[np.newaxis,:]和初始化的时候就多生成一维的方法，最后使用np.concatenate进行拼接。

<!-- more -->

### 数组拼接方法一

思路：首先将数组转成列表，然后利用列表的拼接函数append()、extend()等进行拼接处理，最后将列表转成数组。

```python
>>> a_list.extend(b_list)
>>> a_list
[1, 2, 5, 10, 12, 15]
>>> a=np.array(a_list)
>>> a
array([ 1,  2,  5, 10, 12, 15])
```

该方法只适用于简单的一维数组拼接，由于转换过程很耗时间，对于大量数据的拼接一般不建议使用。

### 数组拼接方法二

思路：numpy提供了numpy.append(arr, values, axis=None)函数。对于参数规定，要么一个数组和一个数值；要么两个数组，不能三个及以上数组直接append拼接。append函数返回的始终是一个一维数组。

```python
>>> a=np.arange(5)
>>> a
array([0, 1, 2, 3, 4])
>>> np.append(a,10)
array([ 0,  1,  2,  3,  4, 10])
>>> a
array([0, 1, 2, 3, 4])
>>> b=np.array([11,22,33])
>>> b
array([11, 22, 33])
>>> np.append(a,b)
array([ 0,  1,  2,  3,  4, 11, 22, 33]) 
```

numpy的数组没有动态改变大小的功能，numpy.append()函数每次都会重新分配整个数组，并把原来的数组复制到新数组中。 

### 数组拼接方法三

思路：numpy提供了numpy.concatenate((a1,a2,...), axis=0)函数。能够一次完成多个数组的拼接。其中a1,a2,...是数组类型的参数

```python
>>> a=np.array([1,2,3])
>>> b=np.array([11,22,33])
>>> c=np.array([44,55,66])
>>> np.concatenate((a,b,c),axis=0)  # 默认情况下，axis=0可以不写
array([ 1,  2,  3, 11, 22, 33, 44, 55, 66]) #对于一维数组拼接，axis的值不影响最后的结果
>>> np.concatenate((a,b),axis=1)  #axis=1表示对应行的数组进行拼接
array([[ 1,  2,  3, 11, 21, 31],
       [ 4,  5,  6,  7,  8,  9]])
```

### 拼接方法时间比较

```python
>>> from time import clock as now
>>> a=np.arange(9999)
>>> b=np.arange(9999)
>>> time1=now()
>>> c=np.append(a,b)
>>> time2=now()
>>> print time2-time1
28.2316728446
>>> a=np.arange(9999)
>>> b=np.arange(9999)
>>> time1=now()
>>> c=np.concatenate((a,b),axis=0)
>>> time2=now()
>>> print time2-time1
20.3934997107
```

可知，concatenate()效率更高，适合大规模的数据拼接

### numpy添加新的维度：newaxis

```python
x = np.random.randint(1, 8, size=5)
x
Out[48]: array([4, 6, 6, 6, 5])
x1 = x[np.newaxis, :]
x1
Out[50]: array([[4, 6, 6, 6, 5]])
x2 = x[:, np.newaxis]
x2
Out[52]: 
array([[4],
       [6],
       [6],
       [6],
       [5]])
```

