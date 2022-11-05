---
title: MATLAB的数组元胞结构解析
tags:
  - matlab
date: 2018-08-27 11:19:00
---


# 问题

MATLAB的数组、元胞、字符串数组分别有什么特点？其与Python等语言的数组有什么较大区别？存储和转换时有什么需要注意？

# 解答

1. MATLAB的元胞其实和Python中的数组更加相近，二者都可以储存混合大小和类型的变量，如果在元胞中储存数组，它们没必要维度相同。而MATLAB的数组必须保证每一维度的元素数目相同。

2. MATLAB数组各个维度的存储顺序和Python正好相反。一个在Python中shape为[2,3,4,5]的数组到了MATLAB中会变成[5,4,3,2]。这种数据的储存方法是和[Fortran](https://zh.wikipedia.org/zh-hans/Fortran)相似的，採用列優先（Column first）。Python中的几个数组通过[A, B]的方式合并时会在最左边（第一维）增加一个“2”（因为这里合并的是两个数组），而MATLAB的cat方法合并两个数组则会在最右边（最后一维）增加一个“2”。

3. 如果是普通数组拿来存储字符串，则必须要保证每个字符串长度相同。但是MATLAB提供了字符串数组这种数据格式。

   ## 创建对象

   您可以通过用双引号括起一段文本来创建字符串。从 R2017a 开始引入双引号。
       
       str = "Hello, world" 
       
       str = "Hello, world" 

   创建字符串数组的一种方法是使用方括号将字符串串联成数组，就像将数字串联成数值数组一样。
       
       str = ["Mercury","Gemini","Apollo"; "Skylab","Skylab B","ISS"] 
       
       str = 2x3 string array "Mercury" "Gemini" "Apollo" "Skylab" "Skylab B" "ISS" 

   也可以按如下所述，使用 `string` 函数将不同数据类型的变量转换成字符串数组。

   ### 语法

   `str = string(A)`

   `str = string(D)`

   `str = string(D,fmt)`

   `str = string(D,fmt,locale)`

4. 将元胞转换成普通List：

   ## cell2mat

   将元胞数组转换为基础数据类型的普通数组

   ## 语法

   ```
   A = cell2mat(C)
   ```

5. 将元胞数组转为字符串数组：

   直接`string(C)`

6. 储存的时候，当待储存的变量小于2G时可以使用scipy读取，而大于2G时会储存为[v 7.3 mat file in python](https://stackoverflow.com/questions/17316880/reading-v-7-3-mat-file-in-python)，这时如果想用Python读取.mat数据，传统的scipy则不再支持，这时需要使用h5py模块读取。但是请注意：scipy在读取的时候回自动把MATLAB数据格式转化成Python格式的顺序，即可以维持原来的顺序不变，而用h5py模块时则会将MATLAB的数组顺序完全颠倒过来。此时，需要使用`np.transpose`函数。

   ```python
   # ! v 7.3 .mat file
   import scipy.io
   mat = scipy.io.loadmat('test.mat')
   
   # v 7.3 .mat file
   import h5py
   with h5py.File('test.mat', 'r') as f:
       f.keys()
       
   # transpose
   X = np.transpose(X, (3, 2, 1, 0))
   ```

7. 当然，也可以在MATLAB中存储大数据时先提前把顺序倒好：

   ```matlab
   X = permute(X, [4,3,2,1]);
   ```