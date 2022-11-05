---
title: Linux(Ubuntu)装机与配置笔记
tags:
  - linux
  - ssh
  - net-disk
  - deep-learning
  - machine-learning
  - cuda
  - python
date: 2020-04-11 18:11:47
---




## 硬盘相关

1. **df命令**
   `df`：检查linux服务器的文件系统的磁盘空间占用情况。**它只会显示已经挂载的磁盘信息！**

   `df -h`, 即` --human-readble`：以1024的倍数的方式显示大小。(e.g., 1023M)

   `df -T`：查看所有磁盘的文件系统类型(type)

2. **`fdisk`命令**

   `fdisk`：强大的磁盘监视和操作工具。

   `fdisk -l`会显示**所有的**磁盘和分区！不论有没有挂载，都会被列出来。

3. **mount命令**

   `mount`：挂载一个文件系统

   `mount -t ntfs <source> <target>`：以ntfs文件系统的形式从源目录挂载到目标目录。t表示types类型

   `mount -a`：挂载 fstab 中的所有文件系统。a表示all

4. **blkid命令**

   `sudo blkid`：获取各个分区的UUID和分区类型TYPE

5. 物理磁盘与磁盘分区：一个物理磁盘在`fdisk -l`中的显示往往类似于`/dev/sda`，`/dev/sdb`，`/dev/nvme0n1`。一般情况下是不带数字的，sda sdb是最常见的命名。而分区命名则是如：`/dev/sda1`，`/dev/sdb2`之类在物理磁盘的后面带上数字表示分区编号。

   但有些如双系统中，可能会出现最后一个例子中展示的命名，这种磁盘的分区则是以`p[x]`结尾，如`/dev/nvme0n1p1`，`/dev/nvme0n1p9`。

6. Linux开机后不会自动挂载Windows文件格式NTFS的磁盘。

7. `sudo chmod -R 777 <Folder_Name>` 可以取消一个文件夹的全部访问权限。

8. `chmod`命令对ext3/4文件系统，即Linux格式的文件系统才有效，对其他文件系统，如vfat(Fat32)，NTFS都是无效的。

9. /etc/fstab` 文件是掌管硬盘自动挂载配置的文件，包含自动挂载分区过程的必要信息。每一条记录格式如下：

   `[Device] [Mount Point] [File System Type] [Options] [Dump] [Pass]`

   如：

   `UUID=B45A01D55A019570 /data ntfs defaults 0 2`

   其中：

   `[Options]` ：`defaults`表示用默认的`rw, suid, dev, exec, auto, nouser, async`等选项（不同内核和文件系统不同）进行挂载，这些选项的含义：`rw` 可读写；`suid` 执行程序时遵守`uuid`；`dev` 解释字符或禁止特殊设备；`exec` 允许执行二进制文件；`auto` 可以`-a`方式加载；`nouser` 禁止普通用户挂载此文件系统；`async` 所有I/O异步完成。

   `[Dump]` ：是否开启分区备份，0表示关闭

   `[Pass]`：系统启动时检查分区错误的顺序，root为1，其他为2，0为不检查。

10. 在`fstab`文件中添加记录前一定要先尝试用mount命令手动挂载。

### 参考

1. [Ubuntu18.04 开机自动挂载其他硬盘](https://blog.csdn.net/qxqxqzzz/article/details/89790688)
2. [Linux查看与挂载新磁盘](https://blog.csdn.net/ybdesire/article/details/79145180)

## CUDA的安装

1. 检查自己的GPU是否是CUDA-capable，在终端中输入`lspci | grep -I NVIDIA` ，会显示自己的NVIDIA GPU版本信息，去CUDA的官网查看自己的GPU版本是否在CUDA的支持列表中。

2. 检查自己的Linux版本是否支持 CUDA（Ubuntu 稳定支持版没问题）。

3. 检查其他问题。这里就不详述了，正常情况下一般OK，这里主要要检查是否安装了`gcc`，是否安装了`kernel header`和 `package development`。如果害怕出现问题可以参考[官网](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)执行这几步检测。

4. 于[CUDA官网](https://developer.nvidia.com/cuda-downloads)下载与系统对应的CUDA版本。最后一个选项选择`runfile`，因为其所需步骤最少，也因此最不容易出问题。所有选项完成后，你会看到如下两行命令：

   ```bash
   wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
   sudo sh cuda_10.2.89_440.33.01_linux.run
   ```

   先不要执行第二条`sudo`开头的指令，只使用`wget`下载。

5. 如果之前有安装过其他版本的CUDA并希望将其卸载，使用`sudo nvidia-uninstall`卸载。如果该命令不在系统路径中，则使用`sudo /usr/bin/nvidia-uninstall`（位置可能变化）卸载。如果还是没有，或是之前的驱动已经损坏，则：

   ```bash
   sudo apt-get remove --purge nvidia*
   sudo chmod +x NVIDIA-Linux-x86_64-410.93.run
   sudo ./NVIDIA-Linux-x86_64-410.93.run --uninstall
   ```

6. 屏蔽`nouveau`驱动。

### Nouveau是什么

> #### Nouveau: Accelerated Open Source driver for nVidia cards
>
> The **nouveau** project aims to build high-quality, free/libre software drivers for [nVidia cards](https://nouveau.freedesktop.org/wiki/CodeNames/). “Nouveau” [*nuvo*] is the French word for “new”. Nouveau is composed of a Linux kernel KMS driver (nouveau), Gallium3D drivers in Mesa, and the Xorg DDX (xf86-video-nouveau). The kernel components have also been ported to [NetBSD](https://nouveau.freedesktop.org/wiki/NetBSD/).

简单说，nouveau是Linux系统默认的给NVIDIA卡预装的一个图形加速驱动，而这个驱动会与CUDA产生部分冲突，所以在安装CUDA之前需要将其禁用，否则会出现卡在开机登录界面无法进入图形界面（仍然可以ssh访问），黑屏，鼠标键盘输入被禁用等问题中的一个或多个（亲身经历）。

继续安装教程：

6. 刚才说到要屏蔽`nouveau`，那么怎么知道你有没有装它呢？
   使用`lsmod | grep nouveau`命令，如果没有输出，就可以判定你没有运行`nouveau`，可以直接进入下一步，否则：

   1. Create a file at `/etc/modprobe.d/blacklist-nouveau.conf` with the following contents:

      ```bash
      blacklist nouveau
      options nouveau modeset=0
      ```

   2. Regenerate the kernel initramfs:

      ```bash
      sudo update-initramfs -u
      ```

   3. Restart.
   4. Run `lsmod | grep nouveau` again. If there is no output, then you succeed.

7. 此后建议进入一个非图形界面安装，这里可以在重启后使用`ssh`接入，也可以在重启后按`alt+ctrl+f1`，进入**text mode**，登录账户。

8. 输入 `sudo service lightdm stop` 关闭图形化界面。

9. 执行刚才官网中给出的第二条命令：`sudo sh cuda_10.2.89_440.33.01_linux.run`。注意这里的版本会不断有变化。注意这里有一个点，即你是否要同时安装OpenGL，如果你是双显，且主显是非NVIDIA的GPU需要选择no，否则yes。同理，如果准备选no，也可以一开始就加上参数`--no-opengl-files`。 另外，如果不能直接执行，使用`sudo chmod a+x cuda_xx.xx.xx_linux.run`为其赋权。

10. 安装成功后，会提示你将cuda的几个路径添加到系统路径中，这里重复一下，

    ```bash
    export PATH=/usr/local/cuda-10.2/bin:/usr/local/cuda-10.2/NsightCompute-2019.1${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64\
                             ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    ```

11. 使用`nvcc -V`检测是否安装成功。当然也可以同时测试`nvidia-smi`。这里可能会报错并提示需要apt安装一个包，按提示来。
12. 完成。

### 参考

1. [NVIDIA CUDA下载官网](https://developer.nvidia.com/cuda-downloads)
2. [NVIDIA 官方安装指南（英文）](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
3. [NVIDIA 官方安装指南中前置检查部分](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#pre-installation-actions)
4. [How To Install CUDA 10 (together with 9.2) on Ubuntu 18.04 with support for NVIDIA 20XX Turing GPUs](https://www.pugetsystems.com/labs/hpc/How-To-Install-CUDA-10-together-with-9-2-on-Ubuntu-18-04-with-support-for-NVIDIA-20XX-Turing-GPUs-1236/)
5. [Ubuntu 安装 cuda 时卡在登录界面（login loop)的解决方案之一](https://blog.csdn.net/lipi37/article/details/90407099)
6. [ubuntu安装cuda循环登录](https://blog.csdn.net/wkk15903468980/article/details/56489704)
7. [Ubuntu安装和卸载CUDA和CUDNN](https://blog.csdn.net/qq_33200967/article/details/80689543)
8. [Linux安装CUDA的正确姿势](https://blog.csdn.net/wf19930209/article/details/81879514)

## CUDA 与 CUDNN 的联系

1. 要先装CUDA再装CUDNN。
2. 前者是平台，后者是基于平台的深度学习加速器。加速可以应用于几乎全部深度学习平台。还是要安的。
3. 一般深度学习使用安装runtime版本即可。
4. [CUDNN官方下载](https://developer.nvidia.com/rdp/cudnn-download)，[CUDNN官方安装步骤](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

## 修复Ubuntu中“检测到系统程序错误”的问题

### 问题描述

每次开机时都会有“**Ubuntu xx.xx 在启动时检测到系统程序错误** ”弹窗出现。即使点击报告下次还会继续出现。

### 问题来源

之前的某个时刻某个程序崩溃了，而Ubuntu想让你决定要不要把这个问题报告给开发者，这样他们就能够修复这个问题。

### 解决办法

1. `sudo rm /var/crash/* ` ：删除这些错误报告。但是如果又有一个程序崩溃了，你就会再次看到“检测到系统程序错误”的错误。你可以再次删除这些报告文件，或者选择禁用Apport来彻底地摆脱这个错误弹窗。如果你这样做，系统中任何程序崩溃时，系统都不会再通知你。但这未必一件坏事，除非你愿意填写错误报告。如果你不想填写错误报告，那么这些错误通知存不存在都不会有什么区别。
2. `sudo vim /etc/default/apport` 永久屏蔽这些报错。

### 参考

1. [如何修复ubuntu中检测到系统程序错误的问题](https://blog.csdn.net/hywerr/article/details/72582082)
2. [How To Fix System Program Problem Detected In Ubuntu](https://itsfoss.com/how-to-fix-system-program-problem-detected-ubuntu/)

## 安装Python3.6版本的Anaconda

由于之前使用的一些开源库和软件对3.7的支持性尚还有问题，而Anaconda默认Python版本为3.6， 所以有必要把Anaconda降级为3.6版本。

安装方法：

1. 到Anaconda官网下载并安装最新3.7版本。

2. 世界线开始分歧，你可以选择保留3.7版本的Anaconda，并创建一个虚拟环境，或是直接替换Python版本。

   1. 对前者，
      若只要一个python环境不要packages，

      ```bash
      conda create --name ana36 python=3.6
      source activate ana36
      ```

      反之，如果要安装一个新的Anaconda，包含默认的所有packages，

      ```bash
      conda create -n ana36 anaconda python=3.6
      source activate ana36
      ```

      

   2. 对后者，

      ```bash
      conda install python=3.6
      ```

      

## 添加Vim拷贝至系统剪贴板快捷键支持

(from: [link](http://vim.wikia.com/wiki/Mac_OS_X_clipboard_sharing))

Having trouble copying selected text from Vim (not MacVim)? Since using `"+y` or '"*y' in Vim on a Mac doesn't actually copy the selected text to the system clipboard, you might find it beneficial to do the following:

1. Open your `~/.vimrc` file
2. add `vmap '' :w !pbcopy`
3. Save it and `source` the file

现在，你就可以在 visual mode， 即在Esc命令模式后按下v键后的选择模式中，选好需要拷贝区域后，连击两次`'` ，即使用 `''`来拷贝所选区域。



## 在Mac/Linux上使用ssh挂载远程网络硬盘

TL;DR：

1. 安装sshfs: `sudo apt-get install sshfs`
2. 直接在`~/.zshrc`中添加以下行：（当然，需要更改文件夹名称，以及挂载后的命名）

### 连接本地Linux Server

```bash
function connect_misaka () {
    if [ ! -d "/Volumes/misaka-home" ]
    then
        mkdir /Volumes/misaka-home
        sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 misaka:/home/miracle /Volumes/misaka-home/ -ovolname=mk-home
    fi
    if [ ! -d "/Volumes/misaka-storage" ]
    then
        mkdir /Volumes/misaka-storage
        sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 misaka:/data /Volumes/misaka-storage/ -ovolname=mk-2T
    fi
}
```

### 连接Gypsum

```bash
function connect_gypsum () {
    if [ ! -d "/Volumes/gypsum/" ]
    then
        mkdir /Volumes/gypsum
        sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 gypsum:/home/zhongyangzha /Volumes/gypsum/ -ovolname=gp-home
    fi
    if [ ! -d "/Volumes/gypsum-scratch/" ]
    then
        mkdir /Volumes/gypsum-scratch/
        sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 gypsum:/mnt/nfs/scratch1/zhongyangzha/ /Volumes/gypsum-scratch/ -ovolname=gp-scratch
    fi
    if [ ! -d "/Volumes/gypsum-work/" ]
    then
        mkdir /Volumes/gypsum-work
        sshfs -o allow_other,default_permissions,IdentityFile=~/.ssh/id_rsa,reconnect,ServerAliveInterval=15,ServerAliveCountMax=3 gypsum:/mnt/nfs/work1/trahman/zhongyangzha /Volumes/gypsum-work/ -ovolname=gp-work
    fi
}
```

### 参数解释

1. `ovolname`：挂载上网络硬盘之后硬盘的命名
2. `IdentityFile`：如果已经设置了免密登录，用这个参数指明ssh私钥位置即可，不需要输入密码。
3. `<source> <target>`：网络硬盘源位置\<username@ip.address:/the/source/path> 与本机目标挂载位置
4. `reconnect,ServerAliveInterval=15,ServerAliveCountMax=3`：多次断线重连，可以再断开网络连接、服务器重启等问题发生后再次自动连接。