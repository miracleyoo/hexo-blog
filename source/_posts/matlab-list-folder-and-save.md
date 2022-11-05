---
title: MATLAB中列出文件夹下内容并提取含Pattern文件名存储
tags:
  - matlab
date: 2018-08-27 11:18:42
---


# 问题 

MATLAB中如何将一个文件夹下所有文件列出并储存到一个字符串数组中，最后提取含有特定Pattern的文件名。

# 解决方案

```matlab
filenames = dir;
filenames = {filenames.name};
filenames = string(filenames);
STTF = startsWith(filenames,'.');
filenames = filenames(~STTF);
MTF = contains(filenames,'.mat');
filenames = filenames(MTF);
filenum = length(filenames);
```

