---
title: Python调用非相同文件夹下的MATLAB文件
tags:
  - python
  - matlab
date: 2018-08-23 09:46:47
---


# 问题

Python如何调用非相同文件夹下的MATLAB文件。

# 解决方案

```python
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath('./matlab_source_code')
```

