<!-- markdownlint-disable MD033 -->
# Start AI Compiler

- [文章](#文章)
- [视频](#视频)
- [书籍](#书籍)
- [工程](#工程)
- [论文](#论文)
- [笔记](#笔记)

## 文章

- [2022-02 / 深度学习编译器整理 / 王钧](https://zhuanlan.zhihu.com/p/382015459)
- [2021-09 / AI与传统编译器 / 吴建明](https://zhuanlan.zhihu.com/p/412578327)
- [2021-08 / AI框架中图层IR的分析 / 金雪峰](https://zhuanlan.zhihu.com/p/263420069)
- [2021-07 / tvm or mlir ？ / dxq](https://zhuanlan.zhihu.com/p/388452164)
- [2021-04 / 深度学习编译技术的一些思考（五） / 蓝色](https://zhuanlan.zhihu.com/p/366089471)
- [2019-05 / 深度学习编译技术的现状和未来 / 陈天奇](https://zhuanlan.zhihu.com/p/65452090)

## 视频

- [2022-01 / 聊一聊我最近读的编译器后端论文 / 小乖他爹​](https://www.zhihu.com/zvideo/1469216846351790080)
- [2021-12 / TVMCon - 2021 Sessions](https://youtube.com/playlist?list=PL_4zDggB-DBpynCEnC9hV-1euZrP3xDRK)
- [2021-04 / AI 编译器和硬件研讨会 / 李沐](https://mlsys.org/virtual/2021/symposium/1643)

## 书籍

- [2015-07 / 计算机组成与设计 - 硬件/软件接口（第5版）](https://book.douban.com/subject/26604008/)
- [2015-05 / 并行算法设计与性能优化](https://book.douban.com/subject/26413096/)
- [2014-03 / 算法心得：高效算法的奥秘（第2版）](https://book.douban.com/subject/25837031/)
- [2012-12 / 编译器设计（第2版）](https://book.douban.com/subject/20436488/)
- [2012-01 / 计算机体系结构 - 量化研究方法（第5版）](https://book.douban.com/subject/20452387/) / [视频](https://scs.hosted.panopto.com/Panopto/Pages/Viewer.aspx?id=d8c83d3a-8074-4afe-ae3b-693e2250999a)
- [2008-12 / 编译原理 - 原理、技术与工具](https://book.douban.com/subject/3296317/)
- [2005-01 / 高级编译器与实现](https://book.douban.com/subject/1400374/)

## 工程

### 框架

- [IREE / Google](https://google.github.io/iree/): Intermediate Representation Execution Environment. An MLIR-based end-to-end compiler and runtime.
- [MLIR / LLVM](https://mlir.llvm.org/): Multi-Level IR Compiler Framework. A novel approach to building reusable and extensible compiler infrastructure.
- [TC / Facebook](https://facebookresearch.github.io/TensorComprehensions/): Tensor Comprehensions. A fully-functional C++ library to automatically synthesize high-performance machine learning kernels.
- [Tiramisu / MIT](https://www.csail.mit.edu/research/tiramisu-compiler): Tiramisu. A polyhedral compiler for expressing image processing, DNN, and linear/tensor algebra applications.
- [TVM / Apache](https://tvm.apache.org/): Tensor Virtual Machine. An End to End Machine Learning Compiler Framework for CPUs, GPUs and accelerators.
- [XLA / Google](https://www.tensorflow.org/xla): Accelerated Linear Algebra. A domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.

### 语言

- [Halide](https://halide-lang.org/): a language for fast, portable computation on images and tensors

### 整理

- [merrymercy/awesome-tensor-compilers](https://github.com/merrymercy/awesome-tensor-compilers)
- [zwang4/awesome-machine-learning-in-compilers](https://github.com/zwang4/awesome-machine-learning-in-compilers)
- [BirenResearch/AIChip_Paper_List](https://github.com/BirenResearch/AIChip_Paper_List)

## 论文

- [2020 / MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)
- [2020 / The Deep Learning Compiler: A Comprehensive Survey](https://arxiv.org/abs/2002.03794)
- [2019 / An In-depth Comparison of Compilers for DeepNeural Networks on Hardware](https://ieeexplore.ieee.org/document/8782480)
- [2018 / TVM: An Automated End-to-End Optimizing Compiler for Deep Learning](https://www.usenix.org/conference/osdi18/presentation/chen)

## 笔记

- [术语](docs/term.md)
- [架构](docs/arch.md)
- TVM
  - [TVM 安装](docs/tvm/tvm_install.md) - Ubuntu20
  - [TVMC 调优模型](docs/tvm/tvmc_tune.md) - 命令行
  - [AutoTVM 调优模型](docs/tvm/autotvm_tune.md) - Python 接口
