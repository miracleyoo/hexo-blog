---
title: Matlab与Python联合编程
tags:
  - matlab
  - python
date: 2018-08-18 20:16:22
---


先交代一下背景：一个机器学习代码的检验需要用到lab之前写好的仿真代码，现在需要将这两份代码结合在一起工作，并且把机器学习完成的任务部分的matlab代码用Python代码代替。

最开始想出来的可能方法大致的有如下三个：

<!-- more -->

## 可能方案

1. 将3500行MATLAB代码直接翻译成Python代码。

   * 优点：最直观，适配性最强，GitHub有相关转换脚本

   - 缺点：耗时大，暗bug难以调试

2. 将Matlab代码拆分成几个大函数，封装好后用Python运行这些文件

   * 优点：大大降低工作量，而且能够保证封装的文件运行得到正确的结果

   - 缺点：调用泊松方程部分知识一个小调用，整个Main文件大体还要改，list等的传参也很成问题

3. 将机器学习部分做成Linux服务器端，在Matlab端只调用接口

   * 优点：快速，可复用

   - 缺点：同样的问题时学习成本较高。有做不出来的可能，耗时较长

## 相关资源

* Matlab代码到Python的编译器：[Small Matlab to Python compiler](https://github.com/victorlei/smop)
* 我尝试转换后的结果（经过手动Debug）：[Diode-Electron-Hole-Particles-Simulation](https://github.com/miracleyoo/Diode-Electron-Hole-Particles-Simulation)
* Matlab官方给定Python中调用MATLAB API：[用于 Python 的 MATLAB API](https://ww2.mathworks.cn/help/matlab/matlab-engine-for-python.html)
* 上述API安装方法：[安装用于 Python 的 MATLAB 引擎 API](https://ww2.mathworks.cn/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
* 通过Python调用用户自定义脚本的方法：[通过 Python 调用用户脚本和函数](https://ww2.mathworks.cn/help/matlab/matlab_external/call-user-script-and-function-from-python.html)
* Python 类型、容器到 MATLAB 的映射：[从 Python 将数据传递到 MATLAB](https://ww2.mathworks.cn/help/matlab/matlab_external/pass-data-to-matlab-from-python.html)
* Python与Matlab混合编程实例博客：[Python与Matlab混合编程](http://zhaoxuhui.top/blog/2017/12/14/Python与Matlab混合编程.html)
* 将所有工作区变量保存到 MAT 文件：[将所有工作区变量保存到 MAT 文件](https://ww2.mathworks.cn/help/matlab/ref/save.html#btox10b-2_1)

## 实验过程

起初我尝试了第一种方法，借助GitHub上开源的MATLAB转Python的东风，只用了几个小时便完成了大致的转换，本来信心满满准备着手测试时，却意外的发现尽管大部分语法都得到了正确的转换，但是还是有许许多多细节问题需要一步步调整。整个调试持续了整整三天，当最终可以运行是却再次发现了两个问题：

1. 运行速度相较于MATLAB原生程序慢了太多。MATLAB上只要0.4秒的程序转成了Python竟然需要上千秒。经过仔细的分析，原因主要有两个：一是MATLAB本身就对矩阵运算进行的大量的优化，而Python想要实现相似功能只能借助第三方库函数。二是自动转换脚本为了保证数组下标等的正确性引入了一个新的类matlab array。每个数组都会被初始化为一个实例，而这本身就是很耗时间的。三是其中涉及了很多类型转换，而数组有很大很多，不少数组都有多达数万个元素，拷贝式的转换会消耗掉巨量的时间。
2. 结果不正确。有部分函数和方法MATLAB和Python的解读不同。这种问题大部分是可以被发现的，但是实际上debug的难度随着代码长度和复杂度而急剧上升，尤其是原来的代码本身就较为dirty，而不是结构化的运算时。另外还有部分不仔细分析调试断点根本无法发现的隐藏式的bug，这些都使得这个庞然大物似的代码极难调试。

之后，我试着只手动重写main程序，其他的模块全部封装化，但结果依旧不理想。由于程序不断地在MATLAB和Python解释环境中切换，这必然涉及到大量的传参，而MATLAB和Python之间的传参甚至比普通的类型转换更加耗时，往往一个循环要消耗数百秒。Python中一个MATLAB的函数调用传参转换花的时间和本身花去的时间之比根据参数数量和大小甚至可以达到了1:200以上。这无疑是一个令人震惊的比例。但是如果希望通过不断调用MATLAB函数并将长达400行左右的main函数通过Python执行，大量的传参就是无法避免的。

于是，经过充分的思考后，我果断放弃了之前五天的成果，选择了下一种方法：只在Python中控制主循环以及几个关键而小的变量，把之前的main长长的代码拆成了数个代码块，打包成一个个可供Python调用的函数，并把函数文件间的沟通“大任”从Python主程序转移到了中间mat文件。每个MATLAB脚本执行之前都会Load之前存下的中间变量并在结尾储存当前所有的中间变量。通过这种方法，Python中每个循环的执行速度也由直接传参的数百秒降低到了可以接受的0.5秒。之后便是把自己的机器学习代码写了一个简单而迅速的接口，替换掉了原有的泊松方程求解部分。预计下种可以出来正式的拟合曲线。

## 简单结论

* Smop自动转换会把数组、矩阵解释为一种自定义类matlab array，来回转换效率很低。
* 如果工程项目真的很小，并且希望考虑执行效率，建议手动转换Python
* 如果工程项目中等（1~3个.m文件），并对效率没有太高要求，可以使用GitHub的smop自动转换后**手动Debug**。
* 如果工程可以拆分为数个函数，并且运行的机器上已经装有MATLAB或是可以并愿意画上十几个G安装MATLAB，可以考虑使用MATLAB官方提供的Python调用MATLAB函数的接口。
* 如果采用Python调用MATLAB函数的解决方案，建议只把Python中必须用到的变量传递给Python主程序，其他部分则考虑采用储存和加载工作区全部变量的方式操作。注意这两种变量不要混着用，即既在储存文件时储存，又传回Python，这样很容易引起混乱。



