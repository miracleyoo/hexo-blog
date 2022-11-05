---
title: Paper Reading ： "NOSE： A Novel Odor Sensing Engine for Ambient Monitoring of the Frying Cooking Method in Kitchen Environments"
tags:
  - paper
  - machine-learning
  - HCI
  - mobile-health
date: 2019-10-24 19:40:02
---

# One Line Summary

NOSE: A device which utilize order sensing component and machine learning to detect which kind of cooking method and which kind of foods, oils are used when you are cooking. It can be used to periodically reports to users about their cooking habits.

# Terms

1.  MOS Gas Sensor: A sensor which is sensitive to specific target analytes and attempts to replicate the human olfactory system by detecting various types of odors.

# Points

1. In the real-world environment, we cannot get the data with a start and end well defined. Here the author exploited a two-level classification approach.

![image-20191024193802569](image-20191024193802569.png)

# Images

![image-20191021152129125](image-20191021152129125.png)

![image-20191021154348447](image-20191021154348447.png)

# Questions

1. We can use multiple dimensional information to detect what is going on. For example, if we add sound detect devices or infrared sensor and use their signals to do analysis at the same time, the accuracy may get dramatically improved. If camera can also be applied, even more detailed information can be obtained.
2. Regarding the privacy issue, perhaps we can consider using a embedded auto-clip algorithm which make it possible to only output a limited region which only contains the main region of the Target-of-Interest.

![image-20191024193841882](image-20191024193841882.png)

3. We can even combine different sensor and use one as the trigger of another. For example, camera will only work when the odor sensor feels that there is someone cooking or when infrared sensor feels that someone is approaching or the microphone hears the noise of cooking.  