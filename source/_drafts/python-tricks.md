---
title: Python实用Tricks（一）
tags: [python]
---

# python编程技巧之从字典中提取子集

```python
prices={'ACME':45.23,'APPLE':666,'IBM':343,'HPQ':33,'FB':10}
#选出价格大于 200 的
gt200={key:value for key,value in prices.items() if value > 200}
print(gt200)
print('---------------------')
#提取科技公司的相关信息
tech={'APPLE','IBM','HPQ','FB'}
techDict={ key:value for key,value in prices.items() if key in tech}
print(techDict)123456789
```

运行结果如下：

```python
{'APPLE': 666, 'IBM': 343}
---------------------
{'APPLE': 666, 'FB': 10, 'IBM': 343, 'HPQ': 33}
```

# pytorchのtorch.gather方法

先放官方文档：

![image-20180821094023761](https://ws2.sinaimg.cn/large/006tNbRwly1fuhpaqgl4kj313k0uywkd.jpg)

这个解释可能并不友好，可以看下面这一段代码样例：

```python
>>> import torch
>>> a = torch.Tensor([[1,2],[3,4]])
>>> a
 1  2
 3  4
[torch.FloatTensor of size 2x2]
>>> b = torch.gather(a,1,torch.LongTensor([[0,0],[1,0]]))
>>> b
 1  1
 4  3
[torch.FloatTensor of size 2x2]
>>> b = torch.gather(a,1,torch.LongTensor([[1,0],[1,0]]))
>>> b
 2  1
 4  3
[torch.FloatTensor of size 2x2]
>>> b = torch.gather(a,1,torch.LongTensor([[1,1],[1,0]]))
>>> b
 2  2
 4  3
[torch.FloatTensor of size 2x2]
```

**torch.gather(input, dim, index, out=****None)中的dim表示的就是第几维度，在这个二维例子中，如果dim=0，**

**那么它表示的就是你接下来的操作是对于第一维度进行的，也就是行；如果dim=1,那么它表示的就是你接下来的操作是对于第**

**二维度进行的，也就是列。index的大小和input的大小是一样的，他表示的是你所选择的维度上的操作，比如这个例子中**

```python
a = torch.Tensor([[1,2],[3,4]])  b = torch.gather(a,1,torch.LongTensor([[0,0],[1,0]])) 其中， dim=1，表示的是在第二维度上操作。
```

```python
index = torch.LongTensor([[0,0],[1,0]])，[0,0]就是第一行对应元素的下标，也就是对应的是[1,1]； [1,0]就是第二行对
```

**WARNING: index的类型必须是LongTensor类型的。**

# Pythonのnamedtuple类

namedtuple是继承自tuple的子类。namedtuple创建一个和tuple类似的对象，而且对象拥有可访问的属性。

```python
from collections import namedtuple

# 定义一个namedtuple类型User，并包含name，sex和age属性。
User = namedtuple('User', ['name', 'sex', 'age'])

# 创建一个User对象
user = User(name='kongxx', sex='male', age=21)

# 也可以通过一个list来创建一个User对象，这里注意需要使用"_make"方法
user = User._make(['kongxx', 'male', 21])

print user
# User(name='user1', sex='male', age=21)

# 获取用户的属性
print user.name
print user.sex
print user.age

# 修改对象属性，注意要使用"_replace"方法
user = user._replace(age=22)
print user
# User(name='user1', sex='male', age=21)

# 将User对象转换成字典，注意要使用"_asdict"
print user._asdict()
# OrderedDict([('name', 'kongxx'), ('sex', 'male'), ('age', 22)])
```

		