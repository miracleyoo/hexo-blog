---
title: Linux 查看服务器开放的端口号
tags:
  - linux
  - server
date: 2018-08-27 11:19:11
---


# 问题

如何确定一台Linux主机上有哪些端口正在使用。

# 解决方案

1. 安装**nmap工具检测开放端口**：`apt-get install nmap`
2. 查看本机活跃端口：`nmap 127.0.0.1`

