---
title: Nohup 命令总结
tags: linux
date: 2018-05-06 21:16:11
---


**用途**：不挂断地运行命令。

**语法**：`nohup Command [ Arg … `][ & ]

**描述**：nohup 命令运行由 Command 参数和任何相关的 Arg 参数指定的命令，忽略所有挂断（SIGHUP）信号。在注销后使用 nohup 命令运行后台中的程序。要运行后台中的 nohup 命令，添加 & （ 表示”and”的符号）到命令的尾部。

**输出被重定向**到myout.file文件：`nohup command > myout.file 2>&1 &`

两个常用的**ftp工具**：ncftpget和ncftpput，可以实现后台的ftp上传和下载，这样就可以利用这些命令在后台上传和下载文件了。

如何杀死nohup进程：

```bash
this is how I try to find the process ID:
ps -ef |grep nohup 
this is the command to kill
kill -9 1787 787
```

查看最新输出文件内容：`tail my.log`

动态追踪最新输出内容：`tail -f my.log`