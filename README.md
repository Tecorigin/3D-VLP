# Context-aware Alignment and Mutual Masking for 3D-Language Pre-training

基于此开源模型（三维视觉-语言预训练模型）做cuda到sdaa的自定义算子迁移，并支持此模型在sdaa上运行
## 1. 模型概述
- 仓库链接：[3D-VLP](https://github.com/leolyj/3D-VLP)
- 论文链接：[Paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Jin_Context-Aware_Alignment_and_Mutual_Masking_for_3D-Language_Pre-Training_CVPR_2023_paper.pdf)
- 详细信息参考readme_en.md

## 2. 快速开始
使用本模型执行训练的主要流程如下：
1. 基础环境安装：介绍训练前需要完成的基础环境检查和安装。
2. 获取数据集：介绍如何获取训练所需的数据集。
3. 构建环境：介绍如何构建模型运行所需要的环境。
4. 启动训练：介绍如何运行训练。

### 2.1 基础环境安装

请参考readme_en.md，完成训练前的基础环境检查和安装。

teco软件栈的版本如下：

--------------+--------------------------------------------
 Host IP      | 20.21.22.13
 PyTorch      | 2.7.1+cpu-cxx11-abi
 Torch-SDAA   | 20250827.8.35.dev0+gitc4736f1
--------------+--------------------------------------------
 SDAA Driver  | 2.1.1 (N/A)
 SDAA Runtime | 2.1.1b0 (/opt/tecoai/lib64/libsdaart.so)
 SDPTI        | 1.4.0b0 (/opt/tecoai/lib64/libsdpti.so)
 TecoDNN      | 2.3.0a0 (/opt/tecoai/lib64/libtecodnn.so)
 TecoBLAS     | 2.2.0a0 (/opt/tecoai/lib64/libtecoblas.so)
 CustomDNN    | 2.2.0 (/opt/tecoai/lib64/libtecocustom.so)
 TecoRAND     | 2.0.1 (/opt/tecoai/lib64/libtecorand.so)
 TCCL         | 1.24.0b0 (/opt/tecoai/lib64/libtccl.so)
--------------+--------------------------------------------

### 2.2 准备数据集
数据集下载与预处理见详情readme_en.md


### 2.3 构建环境

所使用的环境下已经包含PyTorch框架虚拟环境。
1. 执行以下命令，启动虚拟环境。
    ```
    conda activate torch_env
    ```
2. 安装python依赖。
    ```
    pip install -r requirements.txt
    ```
3. 编译自定义算子。
    ```
    cd 3D-VLP/lib/pointnet2_sdaa/
    export PYTHONPATH="/softwares/3D-VLP/lib/pointnet2_sdaa/build/lib.linux-x86_64-cpython-310:$PYTHONPATH"
    (设置环境变量，请根据实际路径修改上述命令)
    source /opt/tecoai/setvars.sh
    python setup.py build_ext
    ```
### 2.4 启动训练

1. 在构建好的环境中，进入项目所在目录。
    ```
    cd 3D-VLP
    ```

2. 运行训练。
    - 以3D 视觉问答为例：

    ```
    export TORCH_SDAA_AUTOLOAD=cuda_migrate  #自动迁移环境变量
    python scripts/train_scripts/train_ft_qa.py --tag finetune_ground_cap --use_normal --epoch 20 --lr 0.0001
   ```

### 2.5 备注

#### Installing problem
- Ninja is required to load C++ extensions
```
pip install Ninja
```

* 如果遇缺少算子支持，请更新torch_sdaa包

