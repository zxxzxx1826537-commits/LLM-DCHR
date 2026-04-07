# 蛋白质超图神经网络自动化调参指南

本文档介绍了如何使用自动化机器学习工具对蛋白质超图神经网络进行超参数优化。

## 文件说明

1. `train.py`: 原始训练脚本，保持原有功能
2. `automl_train.py`: 完整的自动化超参数优化脚本
3. `quick_automl.py`: 快速测试版自动化优化脚本
4. `HypergraphProteinRegressionModel.py`: 支持动态配置回归头的模型

## 功能特性

### 1. 自动化超参数调优
支持自动优化以下参数：
- 学习率 (`lr`)
- 权重衰减 (`wd`)
- Dropout率 (`dropout`)
- 隐藏层维度 (`hid_feats`)
- 网络融合参数 (`lambda_param`)
- 训练轮数 (`epochs`)

### 2. 回归头结构自动优化
支持自动优化回归头的：
- 层数 (`n_regressor_layers`)
- 每层维度 (`regressor_layer_X`)

## 使用方法

### 快速测试（推荐首次使用）

```bash
python quick_automl.py
```

这将执行一个快速的超参数优化过程，使用较小的数据集和较少的试验次数。

### 完整优化

```bash
python automl_train.py
```

这将执行完整的超参数优化过程，可能需要较长时间。

## 自定义配置

### 修改搜索空间
在 `quick_automl.py` 或 `automl_train.py` 中的 `objective` 函数里修改参数搜索空间：

```python
# 示例：修改学习率搜索范围
args.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)  # 更宽的范围
```

### 修改试验次数
在脚本末尾修改试验次数：

```python
# 修改为执行更多试验
study = run_quick_automl(n_trials=50)
```

## 输出结果

优化完成后会生成以下文件：
1. `best_hyperparameters.txt` 或 `quick_best_hyperparameters.txt`：包含最佳超参数
2. `best_hypergraph_protein_model.pth`：在交叉验证中表现最好的模型权重

## 注意事项

1. **计算资源**：完整优化可能需要大量时间和计算资源
2. **GPU支持**：如果系统中有GPU，代码会自动使用
3. **内存管理**：大数据集可能会占用较多内存，请根据硬件配置适当调整数据规模
4. **早停机制**：内置早停机制防止过拟合

## 故障排除

如果遇到内存不足错误，可以尝试：
1. 减少试验次数
2. 使用快速测试脚本
3. 减少交叉验证折数
4. 减少蛋白质数据量

提供所有所需文件，复现请修改对应的路径

环境支持：
# packages in environment at D:\anaconda\envs\dgl_new:
#
# Name                       Version             Build                    Channel
_openmp_mutex                4.5                 7_kmp_llvm               conda-forge
alembic                      1.16.4              pyhd8ed1ab_0             conda-forge
annotated-types              0.6.0               py39haa95532_1
biopython                    1.85                py39h071892a_0
blas                         1.0                 mkl
bottleneck                   1.4.2               py39hc99e966_0
brotlicffi                   1.0.9.2             py39h885b0b7_2
bzip2                        1.0.8               h2bbff1b_6
ca-certificates              2026.2.25           h4c7d964_0               conda-forge
cairo                        1.18.4              he9e932c_0
certifi                      2025.10.5           py39haa95532_0
cffi                         2.0.0               py39h02ab6af_0
charset-normalizer           3.4.4               py39haa95532_0
colorama                     0.4.6               py39haa95532_0
colorlog                     6.9.0               pyh7428d3b_1             conda-forge
contourpy                    1.2.1               py39h214f63a_1
cuda-cccl                    13.0.85             h5221881_1               nvidia
cuda-cccl_win-64             13.0.85             hc667259_1               nvidia
cuda-cudart                  12.4.127            0                        nvidia
cuda-cudart-dev              12.4.127            0                        nvidia
cuda-cupti                   12.4.127            0                        nvidia
cuda-libraries               12.4.1              0                        nvidia
cuda-libraries-dev           12.4.1              0                        nvidia
cuda-nvrtc                   12.4.127            0                        nvidia
cuda-nvrtc-dev               12.4.127            0                        nvidia
cuda-nvtx                    12.4.127            0                        nvidia
cuda-opencl                  13.0.85             h17533db_0               nvidia
cuda-opencl-dev              13.0.85             hde976d8_0               nvidia
cuda-profiler-api            13.0.85             h5221881_0               nvidia
cuda-runtime                 12.4.1              0                        nvidia
cuda-version                 13.0                3                        nvidia
cycler                       0.11.0              pyhd3eb1b0_0
dgl                          1.1.2.cu118         py39_0                   dglteam/label/cu118
expat                        2.7.1               h8ddb27b_0
filelock                     3.17.0              py39haa95532_0
fontconfig                   2.15.0              hd211d86_0
fonttools                    4.60.1              py39h02ab6af_0
freetype                     2.13.3              h0620614_0
giflib                       5.2.2               h7edc060_0
gmp                          6.3.0               h537511b_0
gmpy2                        2.2.1               py39h827c3e9_0
graphite2                    1.3.14              hd77b12b_1
greenlet                     3.2.4               py39hdb6649d_0           conda-forge
harfbuzz                     10.2.0              he2f9f60_1
icc_rt                       2022.1.0            h6049295_2
icu                          73.1                h6c2663c_0
idna                         3.11                py39haa95532_0
importlib-metadata           8.7.0               pyhe01879c_1             conda-forge
importlib_resources          6.5.2               py39haa95532_0
intel-openmp                 2023.1.0            h59b6b97_46320
jinja2                       3.1.6               py39haa95532_0
joblib                       1.5.2               py39haa95532_0
jpeg                         9f                  ha349fce_0
khronos-opencl-icd-loader    2025.07.22          h79b28c9_0
kiwisolver                   1.4.4               py39hd77b12b_0
lcms2                        2.16                hb4a4139_0
lerc                         3.0                 hd77b12b_0
levenshtein                  0.27.1              py39h885b0b7_0
libblas                      3.9.0               1_h8933c1f_netlib        conda-forge
libcblas                     3.9.0               13_hc41557d_netlib       conda-forge
libcublas                    12.4.5.8            0                        nvidia
libcublas-dev                12.4.5.8            0                        nvidia
libcufft                     11.2.1.3            0                        nvidia
libcufft-dev                 11.2.1.3            0                        nvidia
libcurand                    10.4.0.35           h86a452a_1               nvidia
libcurand-dev                10.4.0.35           hde976d8_1               nvidia
libcusolver                  11.6.1.9            0                        nvidia
libcusolver-dev              11.6.1.9            0                        nvidia
libcusparse                  12.3.1.170          0                        nvidia
libcusparse-dev              12.3.1.170          0                        nvidia
libdeflate                   1.17                h2bbff1b_1
libffi                       3.4.4               hd77b12b_1
libglib                      2.86.3              h9bccc14_0
libhwloc                     2.12.1              default_hfa10c62_1000
libiconv                     1.18                hc89ec93_0
libjpeg-turbo                2.0.0               h196d8e1_0
libkrb5                      1.22.1              hb237eb7_0
liblapack                    3.9.0               13_h018ca30_netlib       conda-forge
libnpp                       12.2.5.30           0                        nvidia
libnpp-dev                   12.2.5.30           0                        nvidia
libnvfatbin                  13.0.85             h17533db_0               nvidia
libnvfatbin-dev              13.0.85             h17533db_0               nvidia
libnvjitlink                 12.4.127            0                        nvidia
libnvjitlink-dev             12.4.127            0                        nvidia
libnvjpeg                    12.3.1.117          0                        nvidia
libnvjpeg-dev                12.3.1.117          0                        nvidia
libpng                       1.6.50              h46444df_0
libpq                        17.6                h64815fc_1
libtiff                      4.5.1               hd77b12b_0
libuv                        1.52.0              hd0d7782_0
libwebp                      1.3.2               hbc33d0d_0
libwebp-base                 1.3.2               h3d04722_1
libxml2                      2.13.9              h6201b9f_0
libzlib                      1.3.1               h02ab6af_0
llvm-openmp                  21.1.8              h0d817ff_0
lz4-c                        1.9.4               h2bbff1b_1
m2w64-gcc-libgfortran        5.3.0               6                        conda-forge
m2w64-gcc-libs               5.3.0               7                        conda-forge
m2w64-gcc-libs-core          5.3.0               7                        conda-forge
m2w64-gmp                    6.1.0               2                        conda-forge
m2w64-libwinpthread-git      5.0.0.4634.697f757  2                        conda-forge
mako                         1.3.10              pyhd8ed1ab_0             conda-forge
markupsafe                   3.0.2               py39h827c3e9_0
matplotlib                   3.9.2               py39haa95532_1
matplotlib-base              3.9.2               py39he19b0ae_1
mkl                          2023.1.0            h6b88ed4_46358
mkl-service                  2.4.0               py39h827c3e9_2
mkl_fft                      1.3.11              py39h827c3e9_0
mkl_random                   1.2.8               py39hc64d2fc_0
mpc                          1.3.1               h827c3e9_0
mpfr                         4.2.1               h56c3642_0
mpmath                       1.3.0               py39haa95532_0
msys2-conda-epoch            20160418            1                        conda-forge
mysql-common                 9.3.0               h0b12ad4_5
mysql-libs                   9.3.0               hcb0c519_5
networkx                     3.2.1               py39haa95532_0
numexpr                      2.10.1              py39h4cd664f_0
numpy                        1.26.4              py39h055cbcc_0
numpy-base                   1.26.4              py39h65a83cf_0
opencl-headers               2025.07.22          h885b0b7_0
openjpeg                     2.5.2               hae555c5_0
openssl                      3.6.1               hf411b9b_1               conda-forge
optuna                       4.6.0               pyhd8ed1ab_0             conda-forge
packaging                    25.0                py39haa95532_1
pandas                       2.3.3               py39ha5e6156_0
pcre2                        10.46               h5740b90_0
pillow                       11.1.0              py39h096bfcc_0
pip                          25.2                pyhc872135_1
pixman                       0.46.4              h4043f72_0
platformdirs                 4.3.7               py39haa95532_0
pooch                        1.8.2               py39haa95532_0
psutil                       7.0.0               py39h02ab6af_1
pycparser                    2.23                py39haa95532_0
pydantic                     2.12.2              py39haa95532_0
pydantic-core                2.41.4              py39h114bc41_0
pyparsing                    3.2.0               py39haa95532_0
pyqt                         6.9.1               py39h12ec796_0
pyqt6-sip                    13.10.2             py39h630b2a1_0
pysocks                      1.7.1               py39haa95532_1
python                       3.9.24              h716150d_1
python-dateutil              2.9.0post0          py39haa95532_2
python-tzdata                2025.2              pyhd3eb1b0_0
python_abi                   3.9                 2_cp39                   conda-forge
pytorch                      2.5.1               py3.9_cuda12.4_cudnn9_0  pytorch
pytorch-cuda                 12.4                h3fd98bf_7               pytorch
pytorch-mutex                1.0                 cuda                     pytorch
pytz                         2025.2              py39haa95532_0
pyyaml                       6.0.2               py39h827c3e9_0
qtbase                       6.9.2               h06bae2a_4
qtdeclarative                6.9.2               h88b4c33_1
qtsvg                        6.9.2               h30ace32_1
qttools                      6.9.2               h7e7b719_1
qtwebchannel                 6.9.2               heb02b0b_1
qtwebsockets                 6.9.2               heb02b0b_1
rapidfuzz                    3.12.1              py39h5da7b33_0
requests                     2.32.5              py39haa95532_0
scikit-learn                 1.5.1               py39hc64d2fc_0
scipy                        1.13.1              py39h1a10956_0           conda-forge
setuptools                   80.9.0              py39haa95532_0
sip                          6.12.0              py39h706e071_0
six                          1.17.0              py39haa95532_0
sqlalchemy                   2.0.43              py39h0802e32_0           conda-forge
sqlite                       3.50.2              hda9a48d_1
sympy                        1.14.0              py39haa95532_0
tbb                          2021.13.0           hd094cb3_4               conda-forge
threadpoolctl                3.5.0               py39h9909e9c_0
tk                           8.6.15              hf199647_0
tomli                        2.2.1               pyhe01879c_2             conda-forge
torchaudio                   2.5.1               pypi_0                   pypi
torchvision                  0.20.1              pypi_0                   pypi
tornado                      6.5.1               py39h827c3e9_0
tqdm                         4.67.1              py39h9909e9c_0
typing-extensions            4.15.0              py39haa95532_0
typing-inspection            0.4.2               py39haa95532_0
typing_extensions            4.15.0              py39haa95532_0
tzdata                       2025b               h04d1e81_0
ucrt                         10.0.22621.0        haa95532_0
urllib3                      2.5.0               py39haa95532_0
vc                           14.3                h2df5915_10
vc14_runtime                 14.44.35208         h4927774_10
vs2015_runtime               14.44.35208         ha6b5a95_10
wheel                        0.45.1              py39haa95532_0
win_inet_pton                1.1.0               py39haa95532_1
xz                           5.6.4               h4754444_1
yaml                         0.2.5               he774522_0
zipp                         3.23.0              pyhd8ed1ab_0             conda-forge
zlib                         1.3.1               h02ab6af_0
zstd                         1.5.7               h56299aa_0