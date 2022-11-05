---
title: 尝试使用 Hexo 博客
tags: Hexo
date: 2018-03-07 18:07:37
---

本次试验使用Hexo部署一个个人博客系统，总结几点：

1. Hexo是要安装在本地的，服务器端安装的是Nginx服务器
2. Nginx服务器的主目录下面的`/etc/nginx/nginx.conf.default` 和`etc/nginx/nginx.conf`不是一个东西，以后者为主，修改配置也是在后者里面修改的，改前面的文件没用
3. 更换主题时要先在本地进行`hexo clean ; hexo genarate`操作再`hexo deploy`才会生效
4. 可以通过`hexo s —debug`在本地测试运行服务器
5. 可以通过修改本地主目录下的`_config.yml`文件来确保必须先写草稿再进行发布。新建文章的方式为`hexo new first-post`，发布方式为`hexo publish name_of_file`
6. 更换主题后，要先进入主题的主目录下，输入`npm install`命令后会自动安装所有需要的包