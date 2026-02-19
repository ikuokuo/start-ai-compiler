# Triton 环境

- OS: Ubuntu 24.04

分阶段准备环境：

Step 1）Conda 虚拟环境：玩一下 Triton Python API，了解算子开发、追踪编译流程
Step 2）Docker 编译环境：从源码构建 Triton MLIR 环境，探索编译技术栈、自定义硬件 Dialect

## Conda 虚拟环境

安装 Miniconda，

```bash
# 安装对应版本，如 Linux x86_64
#  https://www.anaconda.com/docs/getting-started/miniconda/install
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# 重开终端，初始化
source ~/miniconda3/bin/activate
conda init --all
```

准备虚拟环境，

```bash
# 创建环境
conda create -n triton python=3.12
conda activate triton

# 安装 PyTorch（CUDA 版本不高于 nvidia-smi 显示的）
#  https://pytorch.org/get-started/locally
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 安装 Triton（Python 版本）
#  https://triton-lang.org/main/getting-started/installation.html
pip3 install triton

# 安装其他依赖
pip3 install matplotlib pandas

# 验证
$ python - <<-EOF
import platform
import torch
import triton
print(f" Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.version.cuda} en={torch.cuda.is_available()}")
print(f" Triton: {triton.__version__}")
EOF
 Python: 3.12.12
PyTorch: 2.10.0+cu130
   CUDA: 13.0 en=True
 Triton: 3.6.0
```

## Docker 编译环境

安装 Docker，

- [Install Docker Engine](https://docs.docker.com/engine/install/)
  - [Docker 加速](https://yyixx.com/docs/op/docker/docker_speed)
- [Install Docker Compose](https://docs.docker.com/compose/install/)

```bash
$ docker -v
Docker version 29.2.1, build a5c7197
$ docker compose version
Docker Compose version v5.0.2
```

<!--
sudo docker run hello-world

sudo docker ps -a
sudo docker container prune
sudo docker exec -it <> bash

sudo docker images
sudo docker image prune
sudo docker image prune -a
-->
