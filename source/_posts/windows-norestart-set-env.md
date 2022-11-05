---
title: Windows不重启电脑将路径加入系统PATH
tags:
  - windows
date: 2018-09-15 16:02:20
---


# 问题

如何在不重启电脑的情况下将某个路径加入到系统路径PATH中。

# 解决方案

假设我们需要添加的路径名称为`C:\ProgramData\Anaconda3`

那么在命令行中执行：

`set PATH=%PATH%C:\ProgramData\Anaconda3;`

完毕后，可以使用`echo %PATH%`查看当前系统路径中是否已经完成添加

# 注意

这种添加仅限于在当前命令行中有效，重启命令行请重新操作。



