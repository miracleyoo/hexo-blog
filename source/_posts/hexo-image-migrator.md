---
title: 用Python抢救你的Hexo博客图床链接到本地
tags:
  - blog
  - python
  - tool
date: 2019-10-02 23:47:49
---

## 问题背景

由于近期各大免费图床纷纷加入了防盗链机制（如新浪）并停止对个人博客用的图床链接进行访问授权，博客上的图片出现了大面积的无法显示（如本博客），严重影响了博客的浏览体验。然而现在直接使用文中链接尚还可以将图片下载到本地，但这也并无法得到任何官方保障，所以当务之急是把所有图床照片下载到本地，用hexo原生的图片插入格式进行插入。

而在免费图床渐渐不再可用的现在，当务之急其实已经不是再次更换图床，而是把这些图片抢救到本地，并直接将原图部署到服务器上；或是自己搭建图床。为了节省时间和成本，我这里采用了直接将原图部署到服务器上的操作。

## 这个问题可以拆解为以下几点：

1. 在_post文件夹中建立与markdown文件同名文件夹用于存放图片。
2. 遍历文件夹中文件并用正则匹配的方式匹配得到待替换的链接。
3. 下载所有图片文件并存储到相应位置。
4. 将原文件中的`![name](link)`替换为可在网页上显示的语句。

于是我为了方便使用python写了一个脚本，使得上面这几步可以自动完成。下面贴上主要代码：

```python
def main():
    names=os.listdir(root)
    files=[i for i in names if i.endswith('.md') and not os.path.isdir(os.path.join(root, i)) and not i.startswith('.')]
    file_paths = [os.path.join(root, i) for i in files]
    dirs=[i for i in names if os.path.isdir(os.path.join(root, i)) and not i.startswith('.')]
    dir_paths = [os.path.join(root, i) for i in dirs]
    print(files)
    for file_iter in files:
        name_temp = os.path.splitext(os.path.split(file_iter)[-1])[0]
        if name_temp not in dirs:
            dir_temp = os.path.join(root, name_temp)
            os.mkdir(dir_temp)
        download(os.path.join(root,file_iter))

# 对每个文件中的链接分别进行下载和替换链接处理
def download(file_path):
    print("==> Now dealing with file:", file_path)
    dir_name = os.path.splitext(os.path.split(file_path)[-1])[0]
    # filename = "test"
    name = file_path.split(u"/")
    filename = name[-1]
    with codecs.open(file_path, encoding="UTF-8") as f:
        text = f.read()
    # regex
    result = re.findall('!\[(.*)\]\((.*)\)', text)

    for i, content in enumerate(result):
        image_quote = content[0]
        image_url = content[1]
        try:
            # download img
            img_data = requests.get(image_url).content
            # img name spell
            image_name = image_url.strip("/").split("/")[-1]
            image_path = os.path.join(root, dir_name, image_name)
            print("==>", image_path, '~~~', image_url)
            # write to file
            with open(image_path, 'wb') as handler:
                handler.write(img_data)

            text=text.replace("!["+image_quote+"]("+image_url+")", "!["+image_quote+"]("+image_name+')')
        except:
            continue
    with codecs.open(file_path, mode="w+", encoding="UTF-8") as f:
        f.write(text)
```

如有需求，推荐查看更加详细的使用说明和注意事项。项目在[Github](https://link.zhihu.com/?target=https%3A//github.com/miracleyoo/hexo-migrator)上，并附有step-by-step的说明，即使没有编程基础也可以轻易上手。

如果有帮助到你，欢迎Star支持一下hhh~ :-)