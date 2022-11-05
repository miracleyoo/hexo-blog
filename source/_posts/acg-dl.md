---
title: ACG相关AI项目-论文-代码-数据-资源
tags:
  - acg
  - anime
  - deep-learning
  - paper
date: 2020-07-06 11:19:16
---


## 论文

[[Paper]()] [[Code]()] [20XX]

### 自动勾线-线稿

1. Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup [[Paper](http://www.f.waseda.jp/hfs/SimoSerraSIGGRAPH2016.pdf)] [[Code](https://github.com/bobbens/sketch_simplification)] [[Blog](https://medium.com/coinmonks/simplifying-rough-sketches-using-deep-learning-c404459622b9)] [2015]

   - 早稻田大学15年经典论文。

   - 粗糙手稿映射到精描线稿。

   - 使用的是自建数据集，给定精致线稿让画师去照着描粗稿，这样避免了从线稿描到精稿时候添加和大改了很多线条。数据集没有开源......不过似乎作者给了训练好的权重。

     <!-- more -->

   - We found that the standard approach, which we denote as direct dataset construction, of asking artists to draw a rough sketch and then produce a clean version of the sketch ended up with a lot of changes in the figure, i.e., output lines are greatly changed with respect to their input lines, or new lines are added in the output. This results in very noisy training data that does not perform well. In order to avoid this issue, we found that the best approach is the inverse dataset construction approach, that is, given a clean simplified sketch drawing, the artist is asked to make a rough version of that sketch.

   <img src="image-20200705025359787.png" alt="image-20200705025359787" style="zoom:50%;" />

   <img src="image-20200705025440435.png" alt="image-20200705025440435" style="zoom:50%;" />

### 自动线稿上色

1. Scribbler: Controlling Deep Image Synthesis with Sketch and Color [[Paper](http://openaccess.thecvf.com/content_cvpr_2017/papers/Sangkloy_Scribbler_Controlling_Deep_CVPR_2017_paper.pdf)] [[Code]()] [2017]

2. User-Guided Deep Anime Line Art Colorization with Conditional Adversarial Networks [[Paper](https://arxiv.org/pdf/1808.03240.pdf)] [[Code](https://github.com/orashi/AlacGAN)] [2018]

   - 既支持直接从线稿转换为色稿，也支持用户交互画点生成色稿。
   - 生成器和检测器的输入进行了创新。生成器的输入是线稿、用户色点图、还有一个线稿的特征Map。检测器的输入是两对：（真实色稿，线稿特征）与（生成色稿，线稿特征）。这样做的好处是避免了直接让判别器看到原始线稿，从而一定程度上避免了过拟合。
   - 线稿是通过XDoG转换色稿得来。
   - Comment from [reddit](https://www.reddit.com/r/AnimeResearch/comments/962ziu/userguided_deep_anime_line_art_colorization_with/): I don't have it locally but it should be easy to make an 'illustration dataset' simply by using a tag like [`-monochrome`](https://danbooru.donmai.us/posts?utf8=✓&tags=-monochrome+&ms=1) to filter out any line-art. It's easy to make a line-art dataset because that's also a tag: `lineart`. Finally, you don't need to half-ass the pairs dataset with fake pairs using XDoG because you can just use [parent-child relationships](https://danbooru.donmai.us/wiki_pages/21859) to find all sets of related images, and then extract all pairs of monochrome vs non-monochrome, which will usually give you sketch to completed image. Or there are tags just for this, like [`colored`](https://danbooru.donmai.us/posts?utf8=✓&tags=colored+&ms=1), for human colorization of BW images (and you can again use the parent/child relationships to filter more).

   <img src="image-20200705104853930.png" alt="image-20200705104853930" style="zoom:33%;" />

   <img src="image-20200705104922846.png" alt="image-20200705104922846" style="zoom:33%;" />

   <img src="image-20200705105004118.png" alt="image-20200705105004118" style="zoom:33%;" />

3. Line Art Correlation Matching Network for Automatic Animation Colorization

### 照片转动漫

1. CartoonGAN: Generative Adversarial Networks for Photo Cartoonization [[Paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_CartoonGAN_Generative_Adversarial_CVPR_2018_paper.pdf)] [[Code]()] [2018]
   - Unpaired image dataset training.
   - Two loss: In generator, a semantic loss defined as an ℓ1 sparse regularization in the **high-level feature maps** of the VGG network. In discriminator, an edge-promoting adversarial loss for preserving **clear edges**
   - Built a new set of data which is animation images with edge blurred. Traditional GAN discriminator only discriminate photo from carton(or real from false), but this discriminator has three classes: Normal Photo, Real Carton Image, and Blurred edge carton image.

### 动漫图片Embedding

1. illustration2vec (i2v)。[[Paper](https://www.gwern.net/docs/anime/2015-saito.pdf)] [[Code](https://github.com/rezoo/illustration2vec)] [2015]
   - 使用了VGG网络将任意动漫图片压缩为一个长度为4096的Vector。
   - 同样，由于训练时Y为标签，这里可以使用该网络为动漫图片打标。
   - 训练图片来自于Danbooru和Safebooru两个网站。共计使用了1,287,596张图片和它们的metadata。
   - 标签分为四类：`General tags`, `copyright tags`,`character tags`和`rating tags`。第一个是图片本身的特征，如“武器”，“微笑”，第二个是版权方，如“VOCALOID”；第三个是角色名字，如“hatsune miku”最后一个类别表示是18禁、擦边球还是全年龄图片。

### 简笔画转照片

1. Deep Learning for Free-Hand Sketch: A Survey and A Toolbox [[Paper](https://arxiv.org/pdf/2001.02600.pdf)] [[Code](https://github.com/PengBoXiangShang/torchsketch/)] [2020]

   ![image-20200705105458250](image-20200705105458250.png)

   ![image-20200705105522568](image-20200705105522568.png)

### 其他GAN

1. Style GAN [[Paper](https://arxiv.org/pdf/1812.04948.pdf)] [[Code](https://github.com/NVlabs/stylegan)] [2019]

   - NVIDIA出品
   - 它可以连续控制生成图片的各种独立参数！
   - 之前的GAN都是Input使用一个随机噪声，而NVIDIA这篇这是在许多层中间都添加一波噪声。层数越靠后，这些噪声控制的特征就越细节。
   - gwern训练好的二次元StyleGAN模型：[Link](https://drive.google.com/file/d/1z8N_-xZW9AU45rHYGj1_tDHkIkbnMW-R/view)

   <img src="image-20200705105910579.png" alt="image-20200705105910579" style="zoom:33%;" />

   <img src="v2-3fffd4e8d8e16d7da045de536f6d0e95_1440w.jpg" alt="img" style="zoom:33%;" />

   <img src="image-20200705105951584.png" alt="image-20200705105951584" style="zoom:33%;" />

2. [AdversarialNetsPapers 各种GAN的Papers](https://github.com/zhangqianhui/AdversarialNetsPapers)

3. [GANotebooks 各种GAN的Jupyter Notebook教程](https://github.com/tjwei/GANotebooks)

4. [animeGAN A simple PyTorch Implementation of GAN, focusing on anime face drawing.](https://github.com/jayleicn/animeGAN)

## 数据集

### 纯动漫图片

1. Danbooru2019. [[Release](https://www.gwern.net/Danbooru2019)] [[Code](https://github.com/fire-eggs/Danbooru2019)] [2019]

   - Original or 3x512x512
   - ~3TB or 295GB
   - 3.69M or 2828400 images
   - 108M tag instances (of 392k defined tags, ~29/image). 
   - Covering Danbooru from 24 May 2005 through 31 December 2019 (final ID: #3,734,659).
   - Image files & a JSON export of the metadata.

2. Anime Face Dataset [[Link](https://www.kaggle.com/splcher/animefacedataset)][2019]

   - 数据来源：[Getchu](www.getchu.com)
   - 包含图片数目： 63,632
   - 只包含脸部截取图片
   - 大小：395.95MB
   - 每张图片分辨率：90 * 90 ~ 120 * 120

   <img src="test.jpg" alt="anime girls" style="zoom:50%;" />

### 线稿-色稿对

1. Danbooru Sketch Pair 128x [[Link](https://www.kaggle.com/wuhecong/danbooru-sketch-pair-128x)] [2019]
   - 9.58GB
   - 647K images. 323K sketch-color pairs
   - Image size: 3x128x128
   - 有的是插画，有的是漫画。
   - 博客：Sketch to Color Anime: An application of Conditional GAN. [[Link](https://medium.com/@raviranjankr165/sketch-to-color-anime-an-application-of-conditional-gan-e40f59c66281)] [[Code](https://github.com/ravi-1654003/Sketch2Color-conditional-GAN)] [2020]

### 推荐与评价

1. Anime Recommendations Database [[Link](https://www.kaggle.com/CooperUnion/anime-recommendations-database)] [2016]

   - 数据来源于：[Link](https://myanimelist.net/)
   - 大小：107.14MB
   - This data set contains information on user preference data from 73,516 users on 12,294 anime. 

   <img src="image-20200705024244220.png" alt="image-20200705024244220" style="zoom: 33%;" />

## 博客

1. [GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)
2. [GAN — What is Generative Adversary Networks GAN?](https://medium.com/@jonathan_hui/gan-whats-generative-adversarial-networks-and-its-application-f39ed278ef09)
3. [可能是近期最好玩的深度学习模型：CycleGAN的原理与实验详解](https://www.leiphone.com/news/201709/i9qlcvWrpitOacjf.html)
4. [输入各种参数生成动漫人物头像官方博客](https://makegirlsmoe.github.io/main/2017/08/14/news-english.html)
5. [宅男的福音：用GAN自动生成二次元萌妹子](https://www.jiqizhixin.com/articles/2017-08-20-4)
6. [Chainerを使ってコンピュータにイラストを描かせる](https://qiita.com/rezoolab/items/5cc96b6d31153e0c86bc)
7. [旋转吧！换装少女：一种可生成高分辨率全身动画的GAN](https://www.jqr.com/article/000215)
8. [不要怂，就是GAN](https://www.jianshu.com/p/f31d9fc1d677)
9. [GAN — Some cool applications of GANs](https://medium.com/@jonathan_hui/gan-some-cool-applications-of-gans-4c9ecca35900)
10. [眼见已不为实，迄今最真实的GAN：Progressive Growing of GANs](https://zhuanlan.zhihu.com/p/30532830)
11. [通俗理解生成对抗网络GAN](https://zhuanlan.zhihu.com/p/33752313)
12. [从头开始GAN](https://zhuanlan.zhihu.com/p/27012520)
13. [Cycle GAN 作者官网](https://junyanz.github.io/CycleGAN/)
14. [如何从零开始构建深度学习项目？这里有一份详细的教程](https://www.jiqizhixin.com/articles/2018-04-17-5)
15. [带你理解CycleGAN，并用TensorFlow轻松实现](https://zhuanlan.zhihu.com/p/27145954)

## 其他资源

### 文字转语音

1. [Sinsy](http://sinsy.jp/): アップロードされた楽譜(MusicXML)に基づいて任意の歌声を生成するHMM/DNN歌声合成システム，Sinsy（しぃんしぃ）です．
   - 支持日语、中文、英语。
   - 输入特定格式的乐谱，输出相当不错的唱词音频文件。
   - [sinsy-cli](https://pypi.org/project/sinsy-cli/): 使用命令行调用Sinsy进行合成。安装：`pip install sinsy-cli`
   - 介绍博客：[Hands on Sinsy, a free software solution for song vocal synthesis](http://blog.pcedev.com/2016/02/18/hands-sinsy-solution-song-vocal-synthesis-using-open-source-software/)
2. [Vocaloducer](https://www.animenewsnetwork.com/interest/2013-10-22/vocaloducer-automatically-creates-songs-from-lyrics)。

### 动漫人脸检测切割

1. [自动化动漫人物脸部切割保存](https://github.com/nagadomi/lbpcascade_animeface)
   ![result](43184241-ed3f1af8-9022-11e8-8800-468b002c73d9.png)

### 头像生成

1. [输入各种参数生成动漫人物头像](https://make.girls.moe/)
2. [Chainer-CycleGAN 动漫人物头发转银色](https://github.com/Aixile/chainer-cyclegan)

### 图像超分辨率

1. [Waifu2x 动漫图片无损放大](http://waifu2x.udp.jp)

### 自动线稿上色

1. [PaintsChainer](https://github.com/pfnet/PaintsChainer): Paints Chainer is a line drawing colorizer using chainer. Using CNN, you can colorize your sketch semi-automatically .

   - 作者提供了直接搭建网站server 的代码。
   - 这里是搭建好的[站点](http://paintschainer.preferred.tech/)。
   - 该网站也提供草图或照片提取线稿功能。

   ![image](sample.png)

### 照片画风迁移

1. [Repaint your picture in the style of your favorite artist](https://deepart.io/)

   ![image-20200705105719731](ACG相关AI项目-论文-代码-数据-资源/image-20200705105719731.png)