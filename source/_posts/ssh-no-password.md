---
title: 优雅地使用ssh免密登录
tags:
  - ssh
  - linux
date: 2018-08-27 16:54:00
---


# 问题

如何优雅地使用ssh登录，并不需要每次输入密码。

<!-- more -->

# 解决方案

## 步骤一：使用*Alias*或*config*记住Server信息

### 添加Alias

如果你正在使用默认的bash，别名*alias*信息会被保存在`~/.bashrc`文件中；如果你使用了zsh，那么你需要在`~/.zshrc`文件中修改。

添加别名操作只需在该文件的任意位置插入以下命令（不过建议加载最前或最后方便管理和查找）：

`alias new_name="ssh user@domain -p port_num"`

并在保存并退出后激活该文件：`source ~/.bashrc` 或 `source ~/.zshrc`

### 添加~/.ssh/config配置

打开该文件，并添加如下代码段：

**有IdentityFile：**

```bash
Host miracle
        HostName `xxx.xxx.xxx.xxx`
        User `username`
        Port `2880`
        IdentityFile `~/xxx/xxx/cert.pem`
        UseKeychain yes
        AddKeysToAgent yes
```

**无IdentityFile：**

```bash
Host yoo
        HostName `xxx.xxx.xxx.xxx`
        User `username`
        Port `2880`
        UseKeychain yes
        AddKeysToAgent yes
```

其中，IdentityFile、UseKeychain、AddKeysToAgent等项不是必须的。

登录方法：`ssh miracle`，其中此处的miracle是你给你的Host起的名字。

**优点**：方便，更易于管理，并且使用scp等时也会更方便，比如：`scp *.mat miracle:~ `



## 步骤二：免密登录

1. 如果你没有使用过rsa密钥对，先生成：`ssh-keygen` ，选项可以一路Enter
2. 切换到 *~/.ssh/* 文件夹下将公钥发送到服务器上的某文件夹里：`scp -P 2880 ~/.ssh/id_rsa.pub root@xxx.xxx.xxx.xxx:~`
3. 登录服务器，把PC端的公钥添加至ssh信任列表末尾：`cat id_rsa.pub >> ~/.ssh/authorized_keys`
4. 如果服务器上没有生成rsa密钥对，按（1）中操作在服务器上重复
5. 如果做了这些后依然需要你输入一个*passphrase*：`Enter passphrase for /Users/miracle/.ssh/id_rsa:`，这个passphase是指你生成ssh密钥时输入的一个字符串，如果你一直按的Enter，这里也直接Enter就好，但是如果不幸你当时输入了一个字符串，那么你需要进行步骤（6）：
6. Add Identity Using Keychain：`ssh-add -K ~/.ssh/id_rsa`，之后它会让你输入上面提到的这个字符串，做完这一步之后，你将会永远不用再在登录这台服务器时输入密码

