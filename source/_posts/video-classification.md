---
title: Video Classification Investigation Report
tags:
  - machine-learning
  - deep-learning
  - cv
  - video-classification
date: 2019-11-07 13:00:06
---


# Overview

Video classification, or in our case, more specifically, action recognition, are studied for a long time. There are many traditional as well as deep learning based method developed to address this problem, and the latest action recognition result trained on a large dataset Kinetics can even reach 98% accuracy. Considering the fact that the action we need to classify is not too much, giving enough data and using the pre-trained model on Kinetics, the result can be quite promising. 

# Tough Points in Video Classification

1. The huge computational cost
2. How to capture long context and make decision comprehensively
3. How to design the classification structure which contain spatiotemporal information 
4. How to deal with a smaller dataset

# Approaches overview

## The core idea 

1. Try to build a workflow which can combine both spatial information and temporal information. 
2. Try to focus on both frame itself and the motion near each frame.
3. Try to make decision based on the whole video rather than only parts of it.
4. Try to decrease the computational cost and remove the long pre-process.

## Two basic methods

### [Single Stream Network](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf)

![image-20191028125241454](image-20191028125241454.png)

There are four ways of fusion, which means combine the information from each frame together to derive the final answer. They are:

1.  Single frame uses single architecture that fuses information from all frames at the last stage.
2. Late fusion uses two nets with shared parameters, spaced 15 frames apart, and also combines predictions at the end.
3. Early fusion combines in the first layer by convolving over 10 frames. 
4. Slow fusion involves fusing at multiple stages, a balance between early and late fusion.

### [Two Stream Networks](https://arxiv.org/pdf/1406.2199.pdf)

![image-20191028125608889](image-20191028125608889.png)

Video can naturally be decomposed into spatial and temporal components. 

1. The spatial part, in the form of individual frame appearance, carries information about scenes and objects depicted in the video. 
2. The temporal part, in the form of motion across the frames, conveys the movement of the observer (the camera) and the objects. In fact, the essence of "motion" is [optical flow](https://en.wikipedia.org/wiki/Optical_flow).

# Improvement of methods

Firstly I'd like to show a graph which shows an overview of all previous action classification architectures drawn in the paper [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750).  

![image-20191028130233266](image-20191028130233266.png)

To summarize, there are these kinds of improved methods:

1. [LRCN](https://arxiv.org/abs/1411.4389): Long-term Recurrent Convolutional Networks for Visual Recognition and Description 

    ![2 stream architecture](GenericLRCN_high.png) 

   Send each frame to a CNN at first and then uses the features extracted as the input of LSTM.

2. [C3D](https://arxiv.org/pdf/1412.0767): Learning Spatiotemporal Features with 3D Convolutional Networks 

    ![SegNet Architecture](c3d_high-1572285955778.png) 

   The first time using 3D Conv to process frames. 

3. [Conv3D & Attention](https://arxiv.org/abs/1502.08029): Describing Videos by Exploiting Temporal Structure 

    ![Attention Mechanism](Larochelle_paper_high.png)

   Add a attention mask before send the CNN-extracted feature into LSTM. 

4. [TwoStreamFusion](https://arxiv.org/abs/1604.06573): Convolutional Two-Stream Network Fusion for Video Action Recognition 

    ![SegNet Architecture](fusion_strategies_high.png)

   Fuse two stream in a smarter way and get a better result. 

5. [TSN](https://arxiv.org/abs/1608.00859) :Temporal Segment Networks: Towards Good Practices for Deep Action Recognition 

    ![SegNet Architecture](tsn_high.png)

   Select video snippets not completely randomly, but divide the video into k equal-length parts and choose a snippets randomly from each division. 

6. [ActionVlad](https://arxiv.org/pdf/1704.02895.pdf):ActionVLAD: Learning spatio-temporal aggregation for action classification

    ![SegNet Architecture](actionvlad-1572285243641.png)

    In this work, the most notable contribution by the authors is the usage of learnable feature aggregation (VLAD) as compared to normal aggregation using maxpool or avgpool.  

7. [HiddenTwoStream](https://arxiv.org/abs/1704.00389):Hidden Two-Stream Convolutional Networks for Action Recognition 

   ![image-20191028134438690](image-20191028134438690.png)

   It uses a "MotionNet" to take the place of optical flow.

8. [I3D](https://arxiv.org/abs/1705.07750): Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset 

   Mainly used pretrained network by ImageNet and Kinetics dataset. Also, it use different 3D network for images and optical flows.

9. [T3D](https://arxiv.org/abs/1711.08200): Temporal 3D ConvNets: New Architecture and Transfer Learning for Video Classification 

    ![SegNet Architecture](ttl_layer_high.png)

   Transfer a 2-D DenseNet to a 3D one. 

# Result comparation

![image-20191028133828231](image-20191028133828231.png)

![image-20191028133226300](image-20191028133226300.png)

# Current Thought

As we can see from the analysis above, the I3D is the most computational efficient and accurate method. Also, the pre-trained model of I3D is provided by the author, so we can also take advantage of it. Now I think we should collect enough data of the corresponding action. Moreover, I noticed that there are many new method on Temporal Action Proposals, Temporal Action Localization and Dense-Captioning Events in Videos appearing this year in the competition ActivityNet, I may research into it to get better result later.

# Datasets

* [UCF101]( https://www.crcv.ucf.edu/data/UCF101.php )
* [Sports-1M]( https://cs.stanford.edu/people/karpathy/deepvideo/ )
* [Kinetics]( https://deepmind.com/research/open-source/kinetics )
*  [ActivityNet Version 1.3 dataset](http://activity-net.org/download.html) 

# Codes

* [I3D models transfered from Tensorflow to PyTorch]( https://github.com/hassony2/kinetics_i3d_pytorch )
* [I3D models trained on Kinetics](https://github.com/deepmind/kinetics-i3d )
* [Video Classification Using 3D ResNet]( https://github.com/kenshohara/video-classification-3d-cnn-pytorch )

# Reference

* [Deep Learning for Videos: A 2018 Guide to Action Recognition]( http://blog.qure.ai/notes/deep-learning-for-videos-action-recognition-review )
* [Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset](https://arxiv.org/abs/1705.07750) 
* [Awesome Action Recognition]( https://github.com/jinwchoi/awesome-action-recognition )
* [ActivityNet]( http://activity-net.org/index.html )
* [Determining optical flow]( https://www.sciencedirect.com/science/article/abs/pii/0004370281900242 )
