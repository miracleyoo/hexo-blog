---
title: Linux Server One-Key Setup
tags:
  - linux
  - server
  - tool
date: 2019-09-20 10:02:32
---

推荐一下新写的Linux系统一键装机脚本(๑´∀｀๑)一行命令获得常用命令行软件、zsh、方便好用的诸多zsh插件（如自动补全、一键解压、目录快速跳转、命令行语法高亮等。详见readme和source code）以及一个配置好的spacevim。当前只支持ubuntu。后面将会支持centos。

你将在任何一个新的Linux系统上一键得到一个稳定可靠的使用体验！当然，脚本高度可定制，以上所有内容都可以简单地增减。有问题或需求欢迎提issue～（注意⚠️在已经使用并配置过的电脑上运行可能会出现zshrc配置重复问题）

[GitHub链接](https://github.com/miracleyoo/initialize-server-script)

<!-- more -->

**下面是英文的使用说明：**

## Overview

This project aims to conveniently setup and deploy a Linux environment which is easy to use and help install many useful packages. It mainly have the ability to deploy zsh with a set of handy plugins, and a spacevim, which is my favorite vim distro. You will not encounter messy installation problems and the script is tested on ubuntu and WSL ubuntu.

Currently, it only support Ubuntu system, but the support of centos is also on the way, maybe also the MacOS version.

It is highly customizable and elegantly wrote, you can folk and customize your own version based on it!

## Usage

### Method 1

1. Make sure you already have curl installed by "sudo apt-get install curl".
2. Use command `curl -fsSL https://raw.githubusercontent.com/miracleyoo/initialize-server-script/master/one-key-linux-setup.sh -o minit.sh && sudo bash minit.sh`
3. You are all set! Here is an awesome new linux!

### Method 2

1. Make sure you already have curl installed by "sudo apt-get install git".
2. Clone this repo using `git clone https://github.com/miracleyoo/initialize-server-script`
3. Switch into this folder and run `./one-key-linux-setup.sh`
4. You are all set!

## Content

1. `apt-get install packags` like git, curl, tmux, vim, and python supports.
2. A `zsh` which has plenty of handy plugins like `oh-my-zsh`, `git`, `zsh-autosuggestions`, `zsh-syntax-highlighting`, `zsh-completions` , `extract`, `z`, `cp`. They are managed with `antigen`, which made it easy and decent to mange your zsh plugins. You can get even more plugins [here](https://github.com/robbyrussell/oh-my-zsh/wiki/Plugins-Overview).
3. A `.zshrc` file which contains some basic but useful functions. You can change it to your own favorite commands and alias.
4. A [spacevim](https://github.com/SpaceVim/SpaceVim), which is a quite good version of vim. It initially installed several famous plugins, with a nice interface. You will find it a really vim distro as you use it. Certainly, you can change to your own version, while I've tested several distro and they all have some kinds of inconvenience, like the line number, extra space, wrong background color and so on.
5. More on the way!