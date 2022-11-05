---
title: What is a pipeline and baseline in machine learning algorithms?
tags:
  - machine-learning
date: 2018-04-07 00:52:19
---


**This article is from Quora, and the original link is [here](https://www.quora.com/What-is-a-pipeline-and-baseline-in-machine-learning-algorithms).**

## 先简单概括一下原文：

* Pipeline指的是一整套流程，从得到数据，加工成能用的数据，通过一套算法进行训练直至最终输出的这个完整过程。
* 而与此相对，baseline指的是一个用作对比的非常simple的方案，通常是以一些市面上流行的基础算法为核心的方案，其作用就是为了让你有一个参照，看看自己到底做到了**多好**

## 下面是[Prasoon Goyal](https://www.quora.com/profile/Prasoon-Goyal)大佬的原答案：

A machine learning algorithm usually takes clean (and often tabular) data, and learns some pattern in the data, to make predictions on new data. However, when ML is used in real-world applications, the raw information that you get from the real-world is often not ready to be fed into the ML algorithm. So you need to preprocess that information to create input data for the ML algorithm. Similarly, the output of the ML algorithm by itself is just some number in software, which will need to be processed to perform some action in the real-world.

Let’s take the example of self-driving cars. Say you have cameras mounted on a car, and you want to predict the optimal steering angle given the images from the camera and the speed of the car. Now once you have collected the data, you will have a set of images from the camera, and you will have the speed of the car at various time points. You will need to align the images with the speed of the car based on the timestamp. Secondly, if you are learning in the supervised setting, you would also have recorded the steering angles of the human drivers while collecting the above data. Again, these need to be aligned with the images and the speed data according to timestamp. What you have done so far is preprocessing — there is no machine learning yet. Now comes the machine learning part — say you train a neural network that is fed these images and corresponding speeds and is trained to predict the optimal steering angle. This is you ML algorithm. Once you’ve trained it, for new data, you perform the preprocessing by aligning camera images and speeds, and feed to the neural network. The neural network outputs a steering angle on a computer screen. Now you need to take this output and actually rewire your car in a way such that the steering wheel rotates based on this output. This part is again not ML.

So this entire framework from converting raw data to data usable by ML algorithm, training an ML algorithm, and finally using the output of the ML algorithm to perform actions in the real-world is the **pipeline**. It is called a pipeline because it is analogous to physical pipelines — just as a liquid passes through one pipe, entering the next, sequentially, our data goes through one stage, entering into the next, sequentially.

------

Baseline, on the other hand, is a totally unrelated concept. Let’s say you want to do part-of-speech tagging — given an English text, tag each word as noun, pronoun, verb, etc. Note that this is non-trivial because a lot of words could belong to several parts-of-speech, based on the context, e.g. *building*.

Now, you train an ML model, and you get an accuracy of 80%. Is that good? You can’t answer that question unless you compare your accuracy to something else. That “something else” is the **baseline**. Sometimes, you pick a simple baseline. So for the example above, a simple baseline could be to just tag each word with its most common part-of-speech. Or you can use existing popular algorithms for that task as the baseline. The choice of the baseline depends on your objective. The goal is to either beat the baseline, if the goal of your work is to improve accuracy, or get results comparable to the baseline while improving some other aspect of the algorithm (like training time, prediction time, memory usage, etc.)