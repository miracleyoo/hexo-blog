---
title: MATLAB实用Tricks
tags:
  - matlab
date: 2018-09-15 16:02:35
---


# permute

重新排列 N 维数组的维度

## 语法

`B = permute(A,order)`

## 说明

`B = permute(A,order)` 重新排列 `A` 的维度使其按向量 `order` 所指定的顺序排列。`B` 含有与 `A` 相同的值，但访问任意特定元素所需的下标的顺序已按 `order` 所指定的顺序重新排列。`order` 的所有元素都必须是唯一的正整数实数值。

<!-- more -->

## 示例

全部折叠

### 置换数组维度 

创建一个 3×4×5 数组，并置换它，以便将第一个和第三个维度交换。

```matlab
A = rand(3,4,5);
B = permute(A,[3 2 1]);
size(B)
ans = 

     5     4     3
```



## 通过 Python 调用用户脚本和函数

此示例显示如何通过 Python® 来调用 MATLAB® 脚本，以计算三角形的面积。

在您的当前文件夹中名为 `triarea.m` 的文件中创建一个 MATLAB 脚本。

```matlab
b = 5;
h = 3;
a = 0.5*(b.* h)
```

保存该文件后，启动 Python 并调用该脚本。

```matlab
import matlab.engine
eng = matlab.engine.start_matlab()
eng.triarea(nargout=0)

a =

    7.5000
```

指定 `nargout=0`。尽管脚本会打印输出，但它不会向 Python 返回任何输出参数。

将脚本转换为函数并通过引擎调用该函数。要编辑文件，请打开 MATLAB 编辑器。

```matlab
eng.edit('triarea',nargout=0)
```

删除三个语句。然后添加一条函数声明并保存文件。

```matlab
function a = triarea(b,h)
a = 0.5*(b.* h);
```

通过引擎调用新的 `triarea` 函数。

```matlab
ret = eng.triarea(1.0,5.0)
print(ret)
2.5
```

`triarea` 函数仅返回一个输出参数，因此无需指定 `nargout`。

# 调用其他文件夹中的方法

有时我们希望把matlab的函数文件单独放到一个文件夹，或是其他涉及将.m文件分散放置的情况，这时候要添加路径。

```matlab
addpath Datasets
addpath /Users/miracle/Desktop/MST Project/project code/auto-scan-point
```

这样就可以调用添加的路径下的m文件了。

# 列出某文件夹内容的方法

```matlab
dir
dir name
listing = dir(name)
```

# 查找与指定名称匹配的文件

列出包含词语 `my` 且扩展名为 `.m` 的所有文件。

创建文件夹 `myfolder`，其包含文件 `myfile1.m`、`myfile2.m` 和 `myfile3.txt`。
    
```matlab
mkdir myfolder 
movefile myfile1.m myfolder
movefile myfile2.m myfolder
movefile myfile3.txt myfolder 
```

列出 `myfolder` 中符合条件的文件。
```matlab
cd myfolder 
dir *my*.m 

>>> myfile1.m myfile2.m 
```
# 在子文件夹中查找文件

列出当前文件夹中和当前文件夹的所有子文件夹中的所有文件。

创建文件夹 `myfolder1`，其中包含以下文件和文件夹：
    
    文件结构
    myfile1.m 
    myfolder2 
    	myfile2.m 
    	myfolder3.m 
    		myfile3.m
    
    mkdir myfolder1 mkdir myfolder1/myfolder2 mkdir myfolder1/myfolder2/myfolder3 movefile myfile1.m myfolder1 movefile myfile2.m myfolder1/myfolder2 movefile myfile3.m myfolder1/myfolder2/myfolder3 

列出 `myfolder1` 中和 `myfolder1` 的子文件夹中扩展名为 `.m` 的所有文件。
    
    cd myfolder1 dir **/*.m 
    
    Files Found in Current Folder: myfile1.m Files Found in: myfolder2 myfile2.m Files Found in: myfolder2/myfolder3 myfile3.m 