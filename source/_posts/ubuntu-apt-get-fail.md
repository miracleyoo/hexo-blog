---
title: Ubuntu的apt-get找不到软件包解决方案
tags:
  - linux
  - apt-get
date: 2018-08-15 20:14:28
---


有时Ubuntu系统自带的apt-get会让人头疼的什么包都安装失败，提示无法找到相应包。

## 原因：

`apt-get`命令太久未更新，或是服务器自带版本过低，需要手动更新并升级相应的库。

## 解决方案：

```bash
sudo apt-get update
sudo apt-get upgrade
```

