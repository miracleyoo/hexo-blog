---
title: Pytorch 学习笔记 --入门
tags:
  - pytorch
  - machine-learning
  - deep-learning
date: 2018-03-19 10:32:23
---


# Pytorch 学习笔记 --入门

- 搭建一个模型的步骤：

  1. import需要的模块

     ```python
     import torch
     import torchvision
     import torchvision.transforms as transforms
     ```

  2. Load需要的数据。数据分为trainset和testset，著名的一些数据集如Imagenet, CIFAR10, MNIST等可以直接在torchvision.dataset中Load。trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,                                          shuffl =True, num_workers=2) 语句的作用是把读进来的数据分好batch，做好shuffle。num_workers表示使用的进程数。

     ```python
     transform = transforms.Compose(
         [transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

     trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
     trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                               shuffle=True, num_workers=2)

     testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
     testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                              shuffle=False, num_workers=2)

     classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
     ```

  3. 定义一个卷积神经网络

     ```python
     # autograd.Variable的作用是在于对于一个给定的变量，当定义了forward函数后自动生成backward函数，便于后面计算backward函数的梯度。同理，F的作用是对于一个给定的函数，...
     from torch.autograd import Variable
     import torch.nn as nn # 引用神经网络的各个层时要导入的模块
     import torch.nn.functional as F 

     class Net(nn.Module):
         '''
         这个类当实例化并给喂进来数据后会自动执行从input到output的过程
         '''
         def __init__(self): 
             # 这里的init要写的东西是之后你的网络里面会用到的所有的层
             super(Net, self).__init__()
             self.conv1 = nn.Conv2d(3, 6, 5)
             self.pool = nn.MaxPool2d(2, 2)
             self.conv2 = nn.Conv2d(6, 16, 5)
             self.fc1 = nn.Linear(16 * 5 * 5, 120)
             self.fc2 = nn.Linear(120, 84)
             self.fc3 = nn.Linear(84, 10)

         def forward(self, x):
             # 这里是前向函数，要顺序从raw数据一层一层写下来，后向函数会自动定义
             x = self.pool(F.relu(self.conv1(x)))
             x = self.pool(F.relu(self.conv2(x)))
             x = x.view(-1, 16 * 5 * 5)
             x = F.relu(self.fc1(x))
             x = F.relu(self.fc2(x))
             x = self.fc3(x)
             return x
         net = Net() # 初始化一个实例神经网络
     ```


  4. 定义一个Loss函数和优化器
  4. 定义一个Loss函数和优化器

     ```python
     import torch.optim as optim

     criterion = nn.CrossEntropyLoss() # criterion定义的是Loss函数
     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # optimizer定义的是优化方式。它作用于梯度。
     ```

  5. 用训练集训练网络

     ```python
     for epoch in range(2):  # epoch表示把相同的训练集重复训练的次数

         running_loss = 0.0 # 这个是用来计算每个步长(这里的2000)内loss函数结果的平均值
         for i, data in enumerate(trainloader, 0):
             # get the inputs
             inputs, labels = data

             # wrap them in Variable
             inputs, labels = Variable(inputs), Variable(labels)

             # zero the parameter gradients
             optimizer.zero_grad()

             # forward + backward + optimize
             outputs = net(inputs)
             loss = criterion(outputs, labels)
             loss.backward()
             optimizer.step()

             # print statistics
             running_loss += loss.data[0]
             if i % 2000 == 1999:    # print every 2000 mini-batches
                 print('[%d, %5d] loss: %.3f' %
                       (epoch + 1, i + 1, running_loss / 2000))
                 running_loss = 0.0

     print('Finished Training')
     ```

  6. 在测试集上测试网络

     ```python
     # 整体准确率测试
     correct = 0
     total = 0
     for data in testloader:
         images, labels = data
         outputs = net(Variable(images))
         # 这里的1表示的axis，0表示每一列的max；1表示每一行的max，返回第一个参数是行的最大值，第二个参数是该最大值的位置
         _, predicted = torch.max(outputs.data, 1) 
         total += labels.size(0)
         correct += (predicted == labels).sum()

     print('Accuracy of the network on the 10000 test images: %d %%' % (
         100 * correct / total))
     ```

     ```python
     # 分类准确率测试
     class_correct = list(0. for i in range(10))
     class_total = list(0. for i in range(10))
     for data in testloader:
         images, labels = data
         outputs = net(Variable(images))
         _, predicted = torch.max(outputs.data, 1)
         c = (predicted == labels).squeeze()
         for i in range(4):
             label = labels[i]
             class_correct[label] += c[i]
             class_total[label] += 1

     for i in range(10):
         print('Accuracy of %5s : %2d %%' % (
             classes[i], 100 * class_correct[i] / class_total[i]))
     ```

- 在GPU上训练：网络和变量（input和label）在定义后需要调用cuda()方法。如：

  ```python
  net.cuda() # 使用GPU
  inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
  ```

- 多GPU并行计算：nn.DataParallel。使用GPU，并在多GPU时并行计算的代码：

  ```python
  model = Model(input_size, output_size) 
  if torch.cuda.device_count() > 1: # 如果有多个GPU就并行
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = nn.DataParallel(model)

  if torch.cuda.is_available(): # 如果有GPU就用GPU
     model.cuda()
  ```

  ```python
  # 把数据放到并行的GPU上并输出每块GPU上分配到的数据
  for data in rand_loader:
      if torch.cuda.is_available():
          input_var = Variable(data.cuda())
      else:
          input_var = Variable(data)

      output = model(input_var)
      print("Outside: input size", input_var.size(),
            "output_size", output.size())
  ```

  ​