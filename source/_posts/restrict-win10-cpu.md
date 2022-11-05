---
title: 限制Win10中应用程序的CPU使用率
tags:
  - windows
date: 2018-08-28 12:18:47
---


# 问题

如何限制Windows下某个特定应用程序的CPU使用率。

# 解决方案

**通过在Win10中为某些应用程序分配更少的内核来限制CPU使用率**

1. 打开任务管理器

![image-20180828121424616](006tNbRwgy1fupx4d1fxdj31b215ub29.jpg)

<!-- more -->

2. 进入`Details`选项卡，找到CPU消耗剧烈的目标程序，右键选择`Set affinity`

![image-20180828121552884](006tNbRwgy1fupx4rpj4jj31b215u7wh.jpg)

3. 点选掉合适量的CPU内核

![image-20180828121647347](006tNbRwgy1fupx797p23j31b01601kx.jpg)