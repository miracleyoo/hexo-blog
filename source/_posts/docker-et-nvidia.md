---
title: Docker与Nivida-Docker的用法与注意事项
tags:
  - deep-learning
  - docker
  - nvidia
  - gpu
date: 2020-04-26 17:56:52
---


## Docker Images and Containers

- 清除所有已经停止的container：`docker container prune -f` 。其中 `-f` 表示不弹出确认提示。也可使用`docker rm $(docker ps -a -q)`来清理。其中，`docker rm`代表删除container，而`docker rmi`则是删除image。

- 如果你需要实例化一个只用一次的container，那么使用`docker run --rm`参数，结束后会自动删除。

- `docker ps <-a>` 可以列出正在运行的/全部的container。其效果和`docker container ls <-a>`相同。而若想列出全部images，则要使用`docker images`。

- docker images中的环境变量有四个来源：

  1. Dockerfile中通过`ENV`指令添加的环境变量，如`ENV PATH /opt/conda/bin:$PATH`
  2. Dockerfile中通过修改`/root/.bashrc`文件使用`export`命令添加到bash中的环境变量，如`export PATH=/OPT/conda/bin:$PATH`命令。
  3. 在通过image实例化container时添加`-e`或`--env`参数来添加到环境中的变量。这个方法有局限性，它不能完成对已有变量的“添加”操作，只能新建一个新的环境变量，如`--env NEW_VAR=/opt/conda/bin` 
  4. 在`docker run`末端的container内命令的前面添加一句`export`引导的命令，如：`docker run -it -v $(PWD):/app debian:jessie bash -c 'export PATH=$PATH:/opt/conda/bin; bash'`。它的缺点是较为复杂。

  其中，如果能找到源Dockerfile，最好的方法是通过修改Dockerfile然后重新build得到一个自己的版本。其次是方法三，实在不行使用方法四。如果先进入bash再运行命令可以正常运行，而直接使用`docker run`出现了环境变量相关的失败提示，很可能是由于Dockerfile写的时候使用的是在`/root/.bashrc`中添加环境变量的方法所致。

- 如果需要一个container长期在后台待机候命，那可以使用`-d`或`--detach`选项建立一个一直待机的docker进程。使用方法：

  1. `docker run -itd --name NAME xxx/xxx:xx /bin/bash`
  2. `docker exec -it NAME your-command `

- 启动时如果需要对本地文件夹和Docker内部文件夹做映射，则使用`docker run -v <LOCAL_FOLDER>:<DOCKER_FOLDER>` 。 该参数可以复数次出现，如：

  ```bash
  docker run \
      -v $AUDIO_IN:/input \
      -v $AUDIO_OUT:/output \
      -v $MODEL_DIRECTORY:/model \
      -e MODEL_PATH=/model \
      researchdeezer/spleeter \
      separate -i /input/audio_1.mp3 /input/audio_2.mp3 -o /output
  ```

- docker run所有的参数都应该写在镜像名字`xxx/xxx:xx`前面，写在其后面的统统会被视作在docker container中运行的命令或命令参数。

- 如果你有了一个在后台持续运行的container，且你想弄一个交互性bash，此时你仍需要加上`-it`参数，同样是在`docker exec`后，container名字前加需要的参数，restart和start同理。

- 如果你对作者的Docker Image不满意，需要修改，此时有两种方法：

  1. 找到Dockerfile并修改，`docker build`，`docker push`
  2. 使用bash进入一个实例化的Image，在里面做一通操作，出来后使用`docker commit -m <YOUR_MESSAGE> -a <AUTHOR_NAME> <CONTAINER_ID> <DOCKERHUB_USERNAME/NEW_IMAGE_NAME:TAG>`提交更改使其保存为一个新的镜像，最后使用`docker push`推送新的镜像到docker hub。如果命名有误或忘记添加docker hub username作为前缀，那么可以使用`docker tag <existing-image> <hub-user>/<repo-name>[:<tag>]`改名。

  注意，使用`docker push`需要有docker hub账号，并在push前使用`docker login`操作登录。

- 一个辨析：`docker commit`针对的是一个正在运行的container，使其固化为一个image；而`docker push`推送的则是一个image到docker hub。前者是本地操作，后者是上传操作。

- 一个区别：`docker start`是启动一个已经停止的container，而`docker restart`则是先stop一个container再start。如果一个container已经停止了，那么二者等效。

## Dockerfile

- Docker Hub中并不直接提供Dockerfile，但可以通过查看image的“标签”页面看每个image的docker建立操作。但由于Docker build的时候使用git会很方便，所以很多作者会在其Github上发布这些Dockerfile，往往可以查看介绍页面找到链接。
- Dockerfile中设置进入点命令：`ENTRYPOINT ["spleeter"]`。
1. 这里”spleeter“是一个bin可执行文件。它的效果是：本来需要用户在`docker run`时输入`docker run xxx/xxx:xx spleeter separate`， 现在就只用输入`docker run xxx/xxx:xx separate`了，即run的时候帮你先打了一个命令标记但没给你按回车。
  2.  如果你发现自己在运行一个docker image时候提示了某个你没有输入的命令的相关问题，如`xxx don't have a parameter yyy, please input aaa, bbb, or ccc`，很有可能是Dockerfile中设定了进入点。
  3. 如果作者在Dockerfile中设定了进入点，但你需要进入docker进行调试或检查时，可以使用`docker run -it --entrypoint bash`来切换入点，进入一个bash命令行中调试。 
- `docker build`针对的是一个url或是一个本地的文件夹。如果是本地的文件夹，文件夹内需要含有一个以`Dockerfile`为名的文件，如果需要导入某些文件到Docker Image中，则这些文件需要在正确的位置。
  1. 如果dockerfile的名字不是`Dockerfile`，则使用`-f/--file <DOCKERFILE_NAME>`来指定名称。
  2. 如果需要指定输出image的名字和tag，则使用`-t/--tag`标签，以`name:tag`命名。
  3. 如果build的时候忘记了命名image，则输出的image没有名字和tag，只有一个随机序号。此时如果要重命名，可以用`docker tag <SERIAL_NUMBER> <NAME:TAG>` 命令。默认tag为latest。
  4. Docker Build示例：
     - 本地文件夹：`docker build -f <NAME:TAG> <TARGET DICTIONARY> `
     - 本地文件：`docker build - < <Dockerfile_Path>`
     - URL：`docker build https://github.com/<USERNAME>/<REPONAME>.git#<BRUNCH>:<SUBFOLDERNAME>`

## Nivdia Docker

- 先放[链接](https://github.com/NVIDIA/nvidia-docker)。这里是Nvidia Docker的Github仓库。

- 再说作用。若想在Docker中运行GPU程序，则普通的Docker是做不到的，程序无法默认在Docker中使用GPU计算资源；另一方面，如果本地已经安装了某个版本的CUDA，但目标程序需要依赖另一个版本，这也是非常麻烦的。而Nvidia Docker的出现则很好解决了这个问题。它相当于在Docker的下面塞进了一层CUDA层，介于Container和OS之间。

  ![Nvidia Docker 原理图](docker-et-nvidia/5b208976-b632-11e5-8406-38d379ec46aa.png)

- 然后是安装。

  1. 作为前置条件，需要本机上安装有Nvidia Driver，不强制要求CUDA。（不过既然都安到Driver了，不如把本机CUDA也装了）官方教程[链接](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu-installation)。当然，请安装Docker。

  2. 执行以下代码：

     ```bash
     # Add the package repositories
     distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
     curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
     curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
     
     sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
     sudo systemctl restart docker
     ```

     代码可能会随着Nvidia Docker的的升级而发生变化，最好参阅本章第一条的链接。

- 试运行：

  ```bash
  #### Test nvidia-smi with the latest official CUDA image
  docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
  
  # Start a GPU enabled container on two GPUs
  docker run --gpus 2 nvidia/cuda:10.0-base nvidia-smi
  
  # Starting a GPU enabled container on specific GPUs
  docker run --gpus '"device=1,2"' nvidia/cuda:10.0-base nvidia-smi
  docker run --gpus '"device=UUID-ABCDEF,1"' nvidia/cuda:10.0-base nvidia-smi
  
  # Specifying a capability (graphics, compute, ...) for my container
  # Note this is rarely if ever used this way
  docker run --gpus all,capabilities=utility nvidia/cuda:10.0-base nvidia-smi
  ```

  其标志性特点就是一个参数`--gpus <PARAMETERS>` 一般使用`--gpus all`即可，其他部分和普通docker一模一样。