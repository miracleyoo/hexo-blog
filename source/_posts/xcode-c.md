---
title: Mac OS下C开发相关事项
tags:
  - xcode
  - c
date: 2018-07-29 18:42:22
---


# Xcode简明教程（Mac OS下C开发环境的搭建）

在 Mac OS X 下学习C语言使用 Xcode。Xcode 是由Apple官方开发的IDE，支持C、C++、Objective-C、Swift等，可以用来开发 Mac OS X 和 iOS 上的应用程序。Xcode最初使用GCC作为编译器，后来由于GCC的不配合，改用LLVM/Clang。

Xcode 的安装非常简单，在 APP Store 上直接下载即可，这里不再赘述。

<!-- more -->

## 在Xcode上运行C语言程序

在 Xcode 上运行C语言程序需要先创建工程，再在工程中添加源代码。

1) 打开 Xcode，选择“Create a new Xcode project”创建一个新工程，如下图所示：

![image-20180729181817554](006tNc79ly1ftrj0gvwv6j314c0ngth9.jpg)

2) 接下来，选择要创建的工程类型，如下图所示：

![image-20180729181855118](006tNc79ly1ftrj13r485j31420ssjuw.jpg)

3) 选择“OS X --> Application --> Command Line Tool”，点击“Next”。Command Line Tool 是“命令行工具”的意思，也就是控制台程序。

![image-20180729181921915](006tNc79ly1ftrj1kdat7j31460t0mzl.jpg)

这里需要填写和工程相关的一些信息：

- Product Name：产品名称，即工程名称。
- Organization Name：组织名称，即公司、个人、协会、团队等的名称。
- Organization Identifier：组织标识符，即有别于其他组织的一个标记，例如身份证号、公司网址、组织机构代码证等。
- Bundle Identifier：程序标识符，即有别于其他程序的一个标记，由 Organization Identifier + Product Name 组成。
- Language：工程所用的编程语言，这里选择C语言。

4) 点击“Next”，保存文件后即可进入当前工程，如下图所示： 

左侧是工程目录，主要包含了工程所用到的文件和资源。单击“main.c”，即可进入代码编辑模式，这里 Xcode 已经为我们创建好了一个“Hello World”小程序。点击上方的“运行”按钮，即可在右下角的选项卡中看到输出结果。



# 修改工作路径使之与当前代码存放目录一致

如下图在Xcode中点击工程名，选择Edit Scheme，之后在Use custom working directory前打钩，并将工作路径修改为**/Users/mac/Desktop/test**。

![image-20180729183848469](006tNc79ly1ftrjltwyphj30ji084jsl.jpg)

![image-20180729183918230](006tNc79ly1ftrjmjg1v2j31di0rowkh.jpg)