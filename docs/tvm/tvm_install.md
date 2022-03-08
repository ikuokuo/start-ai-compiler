# TVM 安装

- OS: Ubuntu 20
- Ref: [Install TVM from Source](https://tvm.apache.org/docs/install/from_source.html)

## 获取源码

```bash
git clone --recursive https://github.com/apache/tvm tvm
cd tvm
```

<!--
git submodule update --init
-->

## 前提要求

- GCC >= 5
- CMake >= 3.5
- LLVM >= 4.0
- Python 3.7.X+ / 3.8.X+, 暂不支持 3.9.X+
- CUDA >= 8.0, 如果要用的话

可以选择创建 Python 虚拟环境：

```bash
# Anaconda Python
#  https://www.anaconda.com/products/individual#Downloads
conda create -n tvm python=3.8 -y
conda activate tvm
```

## 准备依赖

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-dev python3-setuptools
sudo apt-get install -y gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
```

## 准备 LLVM

```bash
# LLVM Download Page
#  http://releases.llvm.org/download.html

# 下载预编译文件
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-13.0.0/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz
tar -xf clang+llvm-*.tar.xz

# 软链到 /usr/local/llvm
sudo ln -s ~/clang+llvm-13.0.0-x86_64-linux-gnu-ubuntu-20.04 /usr/local/llvm

# 配置进 ~/.bashrc
#  cp ~/.bashrc ~/.bashrc.bak
cat <<-EOF >> ~/.bashrc
# llvm
export LLVM_HOME=/usr/local/llvm
export PATH=\$LLVM_HOME/bin:\$PATH
EOF

# 检查版本
llvm-config --version

# 或，以 apt 安装
#  https://apt.llvm.org/
```

## 编译配置

```bash
mkdir build
cp cmake/config.cmake build
```

`build/config.cmake`:

- 启用 LLVM for CPU codegen
  - `set(USE_LLVM ON)`
- 启用 CUDA (other backends: OpenCL, RCOM, METAL, VULKAN, …)
  - `set(USE_CUDA ON)`
- 启用 TensorRT codegen or runtime
  - `set(USE_TENSORRT_CODEGEN ON)`
  - `set(USE_TENSORRT_RUNTIME /usr/local/TensorRT)`
- 启用调试
  - `set(USE_GRAPH_EXECUTOR ON)`
  - `set(USE_PROFILER ON)`
- 启用 IR 调试
  - `set(USE_RELAY_DEBUG ON)`
  - `echo 'export TVM_LOG_DEBUG="ir/transform.cc=1;relay/ir/transform.cc=1"' >> ~/.bashrc`

## 编译动态库

```bash
cd build
cmake ..
make -j`nproc`
```

## 安装 Python 包

```bash
cat <<-EOF >> ~/.bashrc
export TVM_HOME=/path/to/tvm
export PYTHONPATH=\$TVM_HOME/python:\${PYTHONPATH}
EOF

# 或，setup.py 安装
#  cd python; python setup.py install; cd ..

# 配置镜像
#  ~/.config/pip/pip.conf
pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
pip config set install.trusted-host mirrors.aliyun.com

# 必要依赖
pip install numpy decorator attrs
# 若用 RPC Tracker
pip install tornado
# 若用 auto-tuning module
pip3 install tornado psutil xgboost cloudpickle

# 测试
python - <<-EOF
import tvm
print(tvm.__version__)
EOF
```
