# Triton 环境

- OS: Ubuntu 24.04

分阶段准备环境：

- Step 1）Conda 虚拟环境：玩一下 Triton Python API，了解算子开发、追踪编译流程
- Step 2）Docker 编译环境：从源码构建 Triton MLIR 环境，探索编译技术栈、自定义硬件 Dialect

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

```bash
# docker group
#  https://docs.docker.com/engine/install/linux-postinstall/
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

<!--
docker run hello-world

docker ps -a
docker container prune
docker exec -it <> bash

docker images
docker image prune
docker image prune -a
-->

安装 NVIDIA Container Toolkit，

- [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

```bash
sudo systemctl restart docker
docker run --rm --runtime=nvidia --gpus all ubuntu:22.04 nvidia-smi
```

准备编译环境，

```bash
docker pull ubuntu:22.04

docker run -it \
-v /home/john/Codes/Triton:/source \
-w /source \
--runtime=nvidia \
--gpus all \
--name triton \
ubuntu:22.04 \
/bin/bash

#docker start triton
#docker exec -it triton bash

apt update -y
apt install -y python3 python3-dev python3-pip
apt install -y git clang

python3 --version
pip3 --version
git --version

# ---

cd /source/
git clone https://github.com/triton-lang/triton.git
cat triton/cmake/llvm-hash.txt

#cd /source/
#git clone https://github.com/llvm/llvm-project.git
#cd llvm-project
#git checkout 979132a02d146ec79e2f046e31877516d7f32d20

cd /source/; mkdir llvm-project; cd llvm-project
git init
git remote add origin https://github.com/llvm/llvm-project.git
git fetch --depth 1 origin 979132a02d146ec79e2f046e31877516d7f32d20
git checkout FETCH_HEAD

# ---

#apt install nvidia-cuda-toolkit
pip3 install --upgrade cmake ninja sccache lit
pip3 install -r /source/llvm-project/mlir/python/requirements.txt

export SCCACHE_DIR="/source/sccache"
export SCCACHE_CACHE_SIZE="2G"

cd /source/llvm-project/
mkdir build; cd build
cmake -GNinja -Bbuild \
-DCMAKE_INSTALL_PREFIX="/opt/llvm" \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_C_COMPILER=clang \
-DCMAKE_CXX_COMPILER=clang++ \
-DCMAKE_ASM_COMPILER=clang \
-DCMAKE_C_COMPILER_LAUNCHER=sccache \
-DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
-DCMAKE_CXX_FLAGS="-Wno-everything" \
-DCMAKE_LINKER=lld \
-DPython3_EXECUTABLE="/usr/bin/python3" \
-DPython_EXECUTABLE="/usr/bin/python3" \
-DLLVM_BUILD_UTILS=ON \
-DLLVM_BUILD_TOOLS=ON \
-DLLVM_ENABLE_ASSERTIONS=ON \
-DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
-DLLVM_ENABLE_PROJECTS="mlir;lld" \
-DLLVM_INSTALL_UTILS=ON \
-DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
-DLLVM_ENABLE_ZSTD=OFF \
/source/llvm-project/llvm
ninja -C build install -j 2

export LLVM_DIR=/opt/llvm

cd /source/triton/
pip3 install -r python/requirements.txt

# setup.py
#  build_args += ['-j' + max_jobs]
LLVM_INCLUDE_DIRS=$LLVM_DIR/include \
LLVM_LIBRARY_DIR=$LLVM_DIR/lib \
LLVM_SYSPATH=$LLVM_DIR \
pip3 install -e .

# ---

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip3 install matplotlib pandas

$ python3 - <<-EOF
import platform
import torch
import triton
print(f" Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"   CUDA: {torch.version.cuda} en={torch.cuda.is_available()}")
print(f" Triton: {triton.__version__}")
EOF
 Python: 3.10.12
PyTorch: 2.10.0+cu130
   CUDA: 13.0 en=True
 Triton: 3.6.0
```

<!--
apt clean
rm -rf /var/lib/apt/lists/*
-->
