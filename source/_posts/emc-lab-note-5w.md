---
title: 密苏里科技大学EMC Lab暑研 第五周
tags:
  - EMC
  - abroad
  - essay
  - matlab
date: 2018-08-18 17:00:00
---


辛苦而刺激的一周。

上周完成了深度学习模型的搭建之后，本周开始了复杂而艰难的测试工作。之所以如此复杂，是因为模型验证只能建立在连续模拟得出的曲线之上。而想要得到足够的验证数据，必须把之前学长所搭建的MATLAB仿真代码和自己基于Python的深度学习代码进行结合。大致的方法有三：

1. 将3500行MATLAB代码直接翻译成Python代码。

   - 优点：最直观，适配性最强，GitHub有相关转换脚本

   - 缺点：耗时大，暗bug难以调试

2. 将Matlab代码拆分成几个大函数，封装好后用Python运行这些文件

   - 优点：大大降低工作量，而且能够保证封装的文件运行得到正确的结果

   - 缺点：调用泊松方程部分知识一个小调用，整个Main文件大体还要改，list等的传参也很成问题

3. 将机器学习部分做成Linux服务器端，在Matlab端只调用接口

   - 优点：快速，可复用

   - 缺点：同样的问题时学习成本较高。有做不出来的可能，耗时较长

起初我尝试了第一种方法，借助GitHub上开源的MATLAB转Python的东风，只用了几个小时便完成了大致的转换，本来信心满满准备着手测试时，却意外的发现尽管大部分语法都得到了正确的转换，但是还是有许许多多细节问题需要一步步调整。整个调试持续了整整三天，当最终可以运行是却再次发现了两个问题：

1. 运行速度相较于MATLAB原生程序慢了太多。MATLAB上只要0.4秒的程序转成了Python竟然需要上千秒。经过仔细的分析，原因主要有两个：一是MATLAB本身就对矩阵运算进行的大量的优化，而Python想要实现相似功能只能借助第三方库函数。二是自动转换脚本为了保证数组下标等的正确性引入了一个新的类matlab array。每个数组都会被初始化为一个实例，而这本身就是很耗时间的。三是其中涉及了很多类型转换，而数组有很大很多，不少数组都有多达数万个元素，拷贝式的转换会消耗掉巨量的时间。
2. 结果不正确。有部分函数和方法MATLAB和Python的解读不同。这种问题大部分是可以被发现的，但是实际上debug的难度随着代码长度和复杂度而急剧上升，尤其是原来的代码本身就较为dirty，而不是结构化的运算时。另外还有部分不仔细分析调试断点根本无法发现的隐藏式的bug，这些都使得这个庞然大物似的代码极难调试。

之后，我试着只手动重写main程序，其他的模块全部封装化，但结果依旧不理想。由于程序不断地在MATLAB和Python解释环境中切换，这必然涉及到大量的传参，而MATLAB和Python之间的传参甚至比普通的类型转换更加耗时，往往一个循环要消耗数百秒。Python中一个MATLAB的函数调用传参转换花的时间和本身花去的时间之比根据参数数量和大小甚至可以达到了1:200以上。这无疑是一个令人震惊的比例。但是如果希望通过不断调用MATLAB函数并将长达400行左右的main函数通过Python执行，大量的传参就是无法避免的。

于是，经过充分的思考后，我果断放弃了之前五天的成果，选择了下一种方法：只在Python中控制主循环以及几个关键而小的变量，把之前的main长长的代码拆成了数个代码块，打包成一个个可供Python调用的函数，并把函数文件间的沟通“大任”从Python主程序转移到了中间mat文件。每个MATLAB脚本执行之前都会Load之前存下的中间变量并在结尾储存当前所有的中间变量。通过这种方法，Python中每个循环的执行速度也由直接传参的数百秒降低到了可以接受的0.5秒。之后便是把自己的机器学习代码写了一个简单而迅速的接口，替换掉了原有的泊松方程求解部分。预计下种可以出来正式的拟合曲线。

说完上面洋溢着满满技术味的本周进展情况，下面就是精彩刺激的“绝地求生”“荒野行动”了。本周五lab组织全员参与了独木舟漂流活动。活动描述很简单：三个人一条小舟，带上充饥的食品，从漂流的起点划到终点。但是到了现场才发现：这个活动和最初我们的印象完全不一样。首先漂流并不是简单的顺水漂下，而是真正用桨把一个铁质的小舟划起来。环境也让人感到十分惊异：和国内经过开发的景点完全不同，河岸两边都是茂密的森林，整个旅程中的大半时间都处在“前无古人后无来者”的状态，紧张刺激的同时又的确充满了危险的气息。其次是水道的状态：急流、险滩、暗礁、巨大树枝、深浅水域、搁浅地带等都给划行造成了巨大的困难。而最关键的三点是：

1. 我们三名来自种子班的同学被分配到了同一条船上，但我们都没有划船的经验
2. 整个水道流域很长，当我们划了4个半小时被告知还有一半的时候真的对绝望有了很好的理解
3. 没人救援。虽然同行的人很多，但是因为几名教授和学长为了增加旅程的乐趣用各种方法翻过路的船，而我们都是新手，为了避免不断被翻船选择了等到所有人走了之后再动身

我们三人在一天中划行了整整7个小时，可谓是相当“勇猛”了。实际上，最快的组4个小时便到达了终点，而我们则因为中途被翻了4次船、等待翻船大佬们先行和由于对划船不熟练造成的转向、撞暗礁和其他各种障碍物、搁浅等原因而延误了很久。不过令人欣慰的是我们在这个过程中在热心老外的指导下成功学会了配合划船，而且由于我们选用了隔水性能很好的保鲜袋，至少保证了一天的食物这两点。

一路上虽然非常辛苦，甚至都多多少少受了些伤，但是泛舟于丛林深处、玩味夏日的蓝天与翠林、感受“蝉噪林逾静，鸟鸣山更幽”的静谧还是让人十分愉快而放松的。在回来的路上，我们还有幸看到了天空中高挂的彩虹，可谓是美妙的一天。![IMG_3058](006tNbRwgy1fuejycdf41j31400u0q7a.jpg)

另外真的多谢早早划完全程还不断担心我们几个的诸位学长老师，当划到最后筋疲力尽时候看到专程从岸边跑回来瞭望我们的教授们时，当最终上岸被学长们热情迎接时，真的感觉非常非常的温暖。谢谢大家！

此外，本周我们还有幸看到了英仙座流星雨，尽管是在我们的公寓后的停车场的简单肉眼观测，但是还是感到了无比的梦幻而美好。漫天星海中的那靓丽的一闪而过，恍若樱花般灿烂而短暂，尽管只消一瞬，但却永驻观者心间。“Wish upon the shooting star”。![IMG_3045](006tNbRwgy1fuejz7guf0j30u01o0ad9.jpg)

本周厨艺依旧在稳步进步，现在已经渐渐能够烧制一些可口的饭菜了，幸福指数又能++了。