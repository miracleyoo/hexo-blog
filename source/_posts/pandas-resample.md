---
title: Pandas Resample
tags:
  - python
  - machine-learning
date: 2020-03-24 21:26:56
---


Pandas原生支持`resample`功能，前提是目标DataFrame需要有一个index的column。假设我们现在在对一个取样率为30Hz的DataFrame做操作，并想将它变resample为16Hz。

首先我们要建立一个`timestamp`的列，这个名字随意，然后它是以秒为单位的该帧的时间，如3.25，14.33。然后我们将其转换为datatime格式，单位为s。

之后便是直接resample，resample中的`rule`，即第一个参数，指明了resample后两帧之间的时间间隔，即周期。如果我们是16Hz，那这个周期为62.5ms。

`resample`方法的格式是：

```python
DataFrame.resample(rule, how=None, axis=0, fill_method=None, closed=None, label=None, convention='start',kind=None, loffset=None, limit=None, base=0)
```

## 示例代码

```python
df.index=pd.to_datetime(df['timestamp'],unit='s')
df=df.resample('62.5L').mean()
df=df.reset_index(drop=True)
del df['timestamp']
```

## Pandas时间缩写

```
B         business day frequency
C         custom business day frequency (experimental)
D         calendar day frequency
W         weekly frequency
M         month end frequency
SM        semi-month end frequency (15th and end of month)
BM        business month end frequency
CBM       custom business month end frequency
MS        month start frequency
SMS       semi-month start frequency (1st and 15th)
BMS       business month start frequency
CBMS      custom business month start frequency
Q         quarter end frequency
BQ        business quarter endfrequency
QS        quarter start frequency
BQS       business quarter start frequency
A         year end frequency
BA, BY    business year end frequency
AS, YS    year start frequency
BAS, BYS  business year start frequency
BH        business hour frequency
H         hourly frequency
T, min    minutely frequency
S         secondly frequency
L, ms     milliseconds
U, us     microseconds
N         nanoseconds
```

