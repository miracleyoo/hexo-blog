---
title: 通过zssh进行跨跳板机的文件传输
tags:
  - linux
  - machine-learning
  - server
  - tool
date: 2018-08-27 16:54:12
---


# 问题

当使用了跳板机连接远程服务器时，传输较大文件较为麻烦。

# 解决方案

### 使用zssh进行传输

1. 安装软件。Mac端：`brew install zssh`

### 上传本地文件到服务器

1. 用zssh登录到远端服务器：`zssh user1@domain1 -p port`
2. 如果需要进行跳板操作，继续：`ssh user2@domain2 -p port`
3. 切换文件夹到将要接收文件的目录：`cd xx/xx/xx`
4. 进入zssh：按下`ctrl+@`组合键
5. 接下来，你会进入以`zssh >`开头的命令行中，此时你实际上是在本机操作
6. 切换到本机待传文件的位置：`zssh > cd target_folder`
7. 开始传输文件到进入zssh前的服务器目录下：`zssh > sz 123.txt `

### 下载服务器文件到本机

1. 先将本地目录切换到将要接收文件的文件目录下：`cd xxx/xxx/xxx `
2. 用zssh登录到远端服务器：`zssh user1@domain1 -p port`
3. 如果需要进行跳板操作，继续：`ssh user2@domain2 -p port`
4. 切换文件夹到将要发送文件的目录：`cd xx/xx/xx`
5. 在远程机器上,启动sz, 准备发送文件：`sz 123.txt `
6. 看到一堆乱码，此时按住组合键进入zssh：`ctrl+@`
7. 接住对应的文件：`zssh > rz `

# 提示

1. zssh 相当于一个套在后续服务器端操作上的一层壳，你按下`ctrl+@`时就回到了你的本机进行操作。

2. 传输文件分为两步进行，分别是`rz`：接收和`sz`：传输

3. 补充命令：

   ```ba&#39;sh
   zssh > pwd //查看本地机器的目录位置
   zssh > cd  //xxx/xxx/xxx 切换目录
   zssh > ls  //查看当前目录下文件列表
   ```


