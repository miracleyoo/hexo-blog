---
title: Git Submodule 攻略
tags:
  - git
date: 2020-10-22 19:26:00
---


## TL;DR

```bash
# Add submodule
git submodule add

# Clone a project with submodules
git clone --recursive

# Update when submodeule remote repo changed
git submodule update --remote

# When cloned without recursive
git submodule init
git submodule update

# Push submodule change to its remote origin master
cd <submodule_name>
git add -A .
git commit -m "xxx"
git checkout <detached branch name/number>
git merge master
git push -u origin master
```

<!-- more -->

## 定义

`git submodule`允许用户将一个 Git 仓库作为另一个 Git 仓库的子目录。 它能让你将另一个仓库克隆到自己的项目中，同时还保持提交的独立性。

## 作用

在我这里，它的作用非常明确，即给在各个项目中都会用到的代码段一个公共栖息地，做到“一处改，处处改”。

## 常用命令

### 添加

`git submodule add`

```bash
# 直接clone，会在当前目录生成一个someSubmodule目录存放仓库内容
git submodule add https://github.com/miracleyoo/someSubmodule

# 指定文件目录
git submodule add https://github.com/miracleyoo/someSubmodule  src/submodulePath
```

添加完之后，子模块目录还是空的（似乎新版不会了），此时需要执行：

`git submodule update --init --recursive`

来真正将子模块中的内容clone下来。同时，如果你的主目录在其他机器也有了一份clone，它们也需要执行上面的命令来把远端关于子模块的更改实际应用。

### Clone时子模块初始化

`clone`父仓库的时候加上`--recursive`，会自动初始化并更新仓库中的每一个子模块

```bash
git clone --recursive
```

或：

如果已经正常的`clone`了，那也可以做以下补救：

```bash
git submodule init
git submodule update
```

正常`clone`包含子模块的函数之后，由于.submodule文件的存在`someSubmodule`已经自动生成，但是里面是空的。上面的两条命令分别：

1. 初始化的本地配置文件
2. 从该项目中抓取所有数据并检出到主项目中。

### 更新

```bash
git submodule update --remote
```

Git 将会进入所有子模块，分别抓取并更新，默认更新master分支。

不带`--remote`的`update`只会在本地没有子模块或它是空的的时候才会有效果。

### 推送子模块修改

这里有一个概念，就是主repo中的子模块被拉到本地时默认是一个子模块远程仓库master分支的`detached branch`。这个分支是master的拷贝，但它不会被推送到远端。如果在子模块中做了修改，并且已经`add`，`commit`，那你会发现当你想要`push`的时候会报错：`Updates were rejected because a pushed branch tip is behind its remote`。这便是所谓的`detached branch`的最直接的体现。

解决方法是：在子模块中先`git checkout master`，然后在`git merge <detached branch name/number>`，最后`git push -u origin master`即可。

这里解释一下`<detached branch name/number>`这个东西可以使用`git branch`命令查看。如果你使用的是`zsh`，那么问题就更简单了，直接在命令提示符处就可以找到。

![image-20200704184550817](image-20200704184550817.png)

![image-20200704184656815](image-20200704184656815.png)

## 参考

1. [来说说坑爹的 git submodule](https://juejin.im/post/5d5ca6e06fb9a06b1a568e32)

2. [Git submodule使用指南（一）](https://juejin.im/post/5ca47a84e51d4565372e46e0)

3. [Working with submodules](https://github.blog/author/jaw6/)

4. [Why is my Git Submodule HEAD detached from master?](https://stackoverflow.com/questions/18770545/why-is-my-git-submodule-head-detached-from-master)