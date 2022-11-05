---
title: 自动生成和安装requirements.txt依赖
tags:
  - machine-learning
  - python
date: 2018-05-14 23:53:20
---


在查看别人的Python项目时，经常会看到一个requirements.txt文件，里面记录了当前程序的所有依赖包及其精确版本号。这个文件有点类似与Rails的Gemfile。其作用是用来在另一台PC上重新构建项目所需要的运行环境依赖。
requirements.txt可以通过pip命令自动生成和安装

### 生成REQUIREMENTS.TXT文件

`pip freeze > requirements.txt`

### 安装REQUIREMENTS.TXT依赖

`pip install -r requirements.txt`

这是发现的一个蛮好用的东西，转载自[链接](http://lazybios.com/2015/06/how-to-use-requirementstxt-file-in-python/)

然而要提醒的一点是，这个自动生成的会把你所有的包都生成进去，如果只有几个特定的包需要特意安装，那么只留下这几个就好，剩下的直接删掉。