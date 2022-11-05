---
title: Jupyter notebook远程访问服务器配置
tags:
  - machine-learning
  - server
  - linux
date: 2018-05-15 00:20:21
---


1.背景

　　一直苦恼于本地机器和服务器上都要配置一些机器学习方面的环境，今天花了点时间研究了下Jupter notebook远程访问服务器，所以记录一下。

　　有些步骤非必须，这里尽量写清楚，读者理解后自行决定如何安装，本文以非root用户安装。

2.注意事项

> 一定要注意检查服务器端的防火墙相应端口有没有打开！！！
>
> 位置往往在：
>
> 安全->防火墙->添加规则->应用更改
>
> VPC网络->点击已经创建的VPC网络->管理配置->添加规则->应用更改

3.安装步骤

1. 登录服务器

2. 检查是否有安装jupyter notebook,终端输入jupyter notebook,如果报错就是无，那么就要用下面命令安装。

   ```
   $sudo pip install pyzmq
   $sudo pip install tornado
   $sudo pip install jinja2
   $sudo pip install jsonschema
   $sudo pip install jupyter
   ```

3. 生成配置文件

jupyter notebook --generate-config

4. 生成密码（后续写配置文件、登录Jupyter notebook需要）

jupyter notebook password

或打开python终端

```
In [1]: from IPython.lib import passwd

In [2]: passwd()
Enter password: 
Verify password: 
Out[2]: 'sha1:0e422dfccef2:84cfbcbb3ef95872fb8e23be3999c123f862d856' 
```

5. 修改默认配置文件

vim ~/.jupyter/jupyter_notebook_config.py 

进行如下修改（这里可以自行配置）：

```
c.NotebookApp.ip='*'
c.NotebookApp.password = u'sha:ce...刚才复制的那个密文'
c.NotebookApp.open_browser = False
c.NotebookApp.port =8888 #随便指定一个端口
c.NotebookApp.allow_root = True
c.IPKernelApp.pylab = 'inline'
```

6. 启动Jupter notebook

jupyter notebook

7. 远程访问

此时应该可以直接从本地浏览器直接访问http://address_of_remote:8888就可以看到jupyter的登陆界面。（特别注意：服务器上的Jupyter notebook不要关）

8. 本地端口转发

ssh转发命令：ssh -L <local host><local port>:<remote host>:<remote port> <SSH hostname> 最好是在vim ~/.zshrc   中设置别名如：

alias aihubjn1='ssh -L localhost:9999:localhost:8989 root@saas.aihub.finalshares.com -p 2880' 

```
第一步：启动ssh端口转发
ssh -L localhost:9999:localhost:8989 root@saas.aihub.finalshares.com -p 2880
第二步：在服务器发起jupyter服务
jupyter notebook --port 8989
第三步：本地开启9999
浏览器访问：http://localhost:9999/
```

这里root为username,登录ssh的账号，saas.aihub.finalshares.com:2880是老师提供的服务器ssh端口，我使用自己的9999端口监听服务器上的8989端口，信息通过ssh转发，在服务器上启动jupyter即可本地登录，爽的飞起。

9. 一点小问题

由于笔者之前本地转过jupter notebook,改下端口号登录

jupyter notebook --no-browser --port=8889

3.参考文献

（1）[如何在云端服务器运行Jupyter Notebook？](https://zhuanlan.zhihu.com/p/20226040) 

（2）[通过SSH远程使用jupyter notebook](http://blog.csdn.net/patrick75/article/details/51473884) 

（3）[远程访问jupyter notebook](http://www.cnblogs.com/yangxiaolan/p/5778305.html)