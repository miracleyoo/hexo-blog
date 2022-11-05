---
title: 改变flask监听的主机地址和端口号
tags:
  - python
  - web
  - flask
date: 2018-08-27 11:19:29
---


# 问题

如何改变flask监听的主机地址和端口号。

# 解决方案

1. 在app.run()中修改配置

1. ```python
   if __name__== '__main__':
       app.run(
           host = '0.0.0.0',
           port = 7777,  
           debug = True 
       )
   ```

2. 运行时指定：

   ```python
   falsk run -h 0.0.0.0 -p 7777
   ```
