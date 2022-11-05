---
title: Paper Reading： "W!NCE： Unobtrusive Sensing of Upper Facial Action Units with EOG-based Eyewear"
tags:
  - paper
  - HCI
  - mobile-health
  - machine-learning
  - deep-learning
date: 2019-10-20 19:07:48
---


## One Line Summary

W!NCW developes a two-stage processing pipeline which can do continuously and unobtrusively sensing of upper facial action units with high fidelity. Because it doesn't use camera so it also eliminate the privacy concerns.

## Terms

1. Electrooculography(EOG, 眼球电图检查): A technique for measuring the corneo-retinal standing potential that exists between the front and the back of the human eye. The resulting signal is called the electrooculogram. Primary applications are in ophthalmological diagnosis and in recording eye movements.
2. Motion artifacts removal pipeline: Mainly used to remove noise across multiple EOF channels and many different head movement patterns. It is based on neural network.
3.  

## Points

1. There are already standard Facial Action Coding System(FACS) along with camera based methods which can be applied to check facial expressions, but their positioning is awkward and they may bring in privacy problems.
2. The hardware is not a lab-product, rather, it is based on commercially available comfortable daily eyeware device J!NS MEME.
3. EOG metrics is useful for recognizing different types of activities such as reading and writing. since each activity has its own unique eye movement pattern.
4. The EOG sensors are placed on the nose and the IMU sensor is embedded in the temples of the eyeglass.
5. W!NCE takes the body motion into consideration, while existing work work in motion artifact removal from physiological signals couldn't do so.
6. The lower face action is harder to be detected because the signal are generated  in distant muscles, so it will be damped when reaches the sensor.
7. J!NS MEME employs stainless steel eletrodes which belong to the stiff material dry eletrodes. It has a lower price and a good electrical performance and lower possiblility of skin irritation compared to gel-based ones.
8. Some actions of heads will cast a similar influence on EOG sensors(like nod and lower eyebrows), while the IMU signals will be quite different, which can help confirm the real action. 
9. We have to consider the signal variation across individuals, since the face shape, the shape of nose-bridge, the fit of the glasses behind the ear, the variation in the way individuals use upper facial muscles influence the signals captured greatly.
10. Personalizing with transfer learning is utilized to address the problem above. The device will take some labled data from user when they use it for the first time, and only re-train the last layer(full-connection layer).
11. The CNN model will not always be in the working state. In fact, the model process will only be triggered when substantial EOG activities are detected after the motion artifact removal stage. Also, the motion artifact removal model will only run when significant variation is observed in the raw EOG signal.

## Question

1. What if user sweet on there nose? Will it affect the accuracy of EOG sensor?
2. This eye-glass based design will be easy to accommodate for those who always wearing a glass, but to the others who don't have the habit, it might be difficult.
3. For the CNN and motion artifact removal trigger, for the emotions or movement which only generate minor signal, like the lower face action, will it be detected?  
4. How to know whether the prediction is right or not? User may express multiple emotion and movement at the same time. Same question when dataset is collected.
5. Why people will need, or need to buy this product? Will a normal person have the requisition to know their facial action and emotion all day? If so, what can the data collected derive?


## Images

![image-20191020171322278](image-20191020171322278.png)

![image-20191020172902493](image-20191020172902493.png)

![image-20191020190355816](image-20191020190355816.png)

![image-20191020173940080](image-20191020173940080.png)

![image-20191020183735447](image-20191020183735447.png)








