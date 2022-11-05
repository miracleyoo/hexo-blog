---
title: Linux下百度网盘命令行客户端
tags:
  - linux
date: 2018-08-28 12:19:04
---


# 问题

往Linux上上传超大文件（如数据集）或从服务器上下载训练好的模型时，用传统的scp或zssh都较慢，如果有中间跳板机，更是只能使用zssh。

# 解决方案

**使用Linux下的百度网盘客户端进行中介。**

**发布页面：[BaiduPCS-Go](https://github.com/iikira/BaiduPCS-Go/releases)**

<!-- more -->

## 优点如下：

1. 速度快，如果你恰好有百度网盘的会员且服务器端网速可以，甚至可以轻松跑到10M/s以上。
2. 支持秒传。如果这个文件曾经被上传过一次百度云，你再次上传时一秒结束战斗。
3. 多平台支持, 支持 Windows, macOS, linux, 移动设备等.
4. 百度帐号多用户支持;
5. 支持搜索文件。太好用了有木有。
6. 通配符匹配网盘路径和 Tab 自动补齐命令和路径, [通配符_百度百科](https://baike.baidu.com/item/%E9%80%9A%E9%85%8D%E7%AC%A6);
7. [下载](https://github.com/iikira/BaiduPCS-Go#%E4%B8%8B%E8%BD%BD%E6%96%87%E4%BB%B6%E7%9B%AE%E5%BD%95)网盘内文件, 支持多个文件或目录下载, 支持断点续传和单文件并行下载;
8. [上传](https://github.com/iikira/BaiduPCS-Go#%E4%B8%8A%E4%BC%A0%E6%96%87%E4%BB%B6%E7%9B%AE%E5%BD%95)本地文件, 支持上传大文件(>2GB), 支持多个文件或目录上传;
9. [离线下载](https://github.com/iikira/BaiduPCS-Go#%E7%A6%BB%E7%BA%BF%E4%B8%8B%E8%BD%BD), 支持http/https/ftp/电驴/磁力链协议.

个人感觉进去了之后就像一个小的独立操作系统，支持其设定的各种命令，而且大多数和linux原生命令重合，如`cd`，`ls`，`search`等。

## 使用方法

1. 从**[BaiduPCS-Go](https://github.com/iikira/BaiduPCS-Go/releases)**这个发布页面用wget等命令下载符合你机器cpu架构的版本。

2. 解压压缩包，进入解压后目录，运行`./BaiduPCS-Go`

3. `ls`查看主目录，`cd`切换到你想下载的位置。

4. 上传：

   ```bash
   BaiduPCS-Go upload <本地文件/目录的路径1> <文件/目录2> <文件/目录3> ... <目标目录>
   BaiduPCS-Go u <本地文件/目录的路径1> <文件/目录2> <文件/目录3> ... <目标目录>
   ```

5. 下载：

   ```bash
   BaiduPCS-Go download <网盘文件或目录的路径1> <文件或目录2> <文件或目录3> ...
   BaiduPCS-Go d <网盘文件或目录的路径1> <文件或目录2> <文件或目录3> ...
   ```
