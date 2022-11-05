---
title: Python 在类的静态函数中调用另一个静态函数
tags:
  - python
date: 2020-03-04 18:28:42
---


有两种常见解决办法：

1. 调用目标函数的`__func__()`方法。
2. 使用`CLASS_NAME.target_func()`方法。这种方法更加干净、Pythonic。

```python
class Klass(object):

    @staticmethod  # use as decorator
    def stat_func():
        return 42

    _ANS = stat_func.__func__()  # call the staticmethod

    def method(self):
        ret = Klass.stat_func()
        return ret
```

