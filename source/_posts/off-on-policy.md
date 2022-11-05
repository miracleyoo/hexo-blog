---
title: Off-Policy & On-Policy
tags:
  - reinforcement-learning
  - machine-learning
  - deep-learning
date: 2019-01-22 16:13:57
---

On-Policy 与 Off-Policy的本质区别在于：更新Q值时所使用的方法是沿用既定的策略（on-policy）还是使用新策略（off-policy）。

![](006tKfTcgy1g0f8y7lartj30ib0kf768.jpg)

<!-- more -->

![](006tKfTcgy1g0f8yhlj6cj30ja0cv75a.jpg)

Sarsa更新Q值的时候对下一步的估计采用的是Q本身的$Q(S’,A’)$，而Q-Learning更新的时候对下一步的估计部分则采用的是直接取采取动作a后环境中各个a’对应的Q的最大值，即其在选择Action的时候使用的是e-greedy算法，而更新Q值的时候则采用了直接取最大值的greedy算法。

## 反映到结果上

**On-policy**：必须本人在场, 并且一定是本人边玩边学习。
**Off-policy**：可以选择自己玩, 也可以选择看着别人玩, 通过看别人玩来学习别人的行为准则。