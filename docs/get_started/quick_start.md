# 昇腾（NPU）快速使用

本文档介绍 Slime main@2026-10-22 的 NPU 环境配置、镜像构建、本地安装和训练入口。

## 硬件支持说明

**slime-ascend** 本版本的 Dockerfile 和默认安装配置面向华为昇腾 **Atlas 800T A3**，使用 Linux ARM64、CANN 9.1.0 和 Python 3.12。训练侧使用 MegatronAdaptor 和 TransformerEngineNPU。

GLM-5.3 Flash 单独使用 CANN 9.2.0 源码安装配置，支持为 A3 或 A5 构建安装包。八卡示例面向 A5。

## 组件版本

下表为 A3 默认安装和镜像配置。GLM-5.3 Flash 使用 torch 2.10.0、torch_npu 2.10.0.post7、torchvision 0.25.0 和 triton-ascend 3.2.2。Slime、SGLang、sgl-kernel-npu、Megatron、FLA-NPU 与 ops-transformer 的固定提交见[模型源码版本表](../../models/glm5.3flash/README.md#源码版本)。

| 组件 | A3 版本 |
| ---- | ------- |
| 基础镜像 | `quay.io/ascend/cann:9.1.0-a3-ubuntu22.04-py3.12`（`linux/arm64`） |
| Python | 3.12 |
| CANN | 9.1.0 |
| torch | 2.10.0 |
| torch_npu | 2.10.0.post6 |
| torchvision | 0.25.0 |
| Megatron-LM | `4b4acac9a1d28ea6829c8d4f566d75698a21249d` |
| Megatron-Bridge | `35b4ebfc486fb15dcc0273ceea804c3606be948a` |
| MegatronAdaptor | `eb0e5043438dd753099432bc3b3d8e5916ed1844` |
| TransformerEngineNPU | `2b8529f94c3fd10497bf772bdc5b448b7ed4659a` |
| mbridge | `89eb10887887bc74853f89a4de258c0702932a1c` |
| triton-ascend | 3.2.1 |
| SGLang | `v0.5.21`（`e00930c5489053f26d86b179cee0d087f846acbb`） |
| sgl-kernel-npu | 社区 `2026.9.0.post11` A3 ARM64 预编译包 |
| torch-memory-saver | 从 `9765e27532b8f854717eba1d3fb9a0bb97a0c887` 重建已打补丁的 wheel |
| DeepEP | 社区 `2026.9.0.post11` A3 ARM64 预编译包 |
| sglang-router | 0.3.2 |
| memfabric-hybrid | 1.2.1 |
| memfabric-zbal | 1.1.3 |
| transformers | 5.12.1 |
| nvidia-modelopt | 0.46.1 |
| Slime | `main@2026-10-22`（`18868a79b0b083a093798a94ed0f8cb4f71b9c14`） |

CANN 9.1.0 / Python 3.12 / A3 ARM64 使用 [sgl-kernel-npu 预编译包](https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.9.0.post11/sgl-kernel-npu-2026.9.0.post11-torch2.10.0-py312-cann9.1.0-a3-aarch64.zip)，安装时校验 SHA256。补丁和版本基线见 [MANIFEST.toml](../../patches/MANIFEST.toml)。本组合尚未完成 ARM64 整镜构建验证。

## 使用指南

模型和训练配置（训推分离、colocate、全异步）见 [特性与模型支持](../supported_models.md)。

## 环境配置

### 方式一：Docker

使用 [A3 Dockerfile](../../docker/Dockerfile.a3.ubuntu22.04.cann910.latest) 构建镜像。

#### 镜像构建

```bash
# 进入项目根目录
cd {slime-ascend-root-path}

# 构建 A3 镜像
docker build --platform linux/arm64 \
  -f docker/Dockerfile.a3.ubuntu22.04.cann910.latest \
  -t slime-ascend:9.1.0-a3-ubuntu22.04-py3.12-main-20261022 .
```

#### 启动容器

```bash
# 启动 A3 容器
docker run -it --rm \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/add-ons:/usr/local/Ascend/add-ons \
  -v "$(pwd)":/workspace/slime-ascend \
  -w /workspace/slime-ascend \
  slime-ascend:9.1.0-a3-ubuntu22.04-py3.12-main-20261022
```

构建上下文须包含仓库 Git 元数据和 Slime submodule。Dockerfile 逐步安装 SGLang、PyTorch、NPU 内核、mbridge、Megatron-Bridge、Megatron-LM、MegatronAdaptor、TransformerEngineNPU 和 Triton-Ascend，随后回放已登记补丁。

### 方式二：本地安装脚本

我们提供了[一键安装脚本](../../scripts/quick_install.sh)，适用于 A3 ARM64、CANN 9.1.0 和 Python 3.12：

```bash
conda create -n slime-ascend python=3.12
conda activate slime-ascend
# 检出官方仓库的 dev 分支并进入项目根目录
git clone -b dev --recurse-submodules https://gitcode.com/Ascend/slime-ascend.git
cd slime-ascend
# 加载 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
# 运行安装脚本
bash scripts/quick_install.sh
```

**安装脚本会自动完成以下工作**：

1. 检查 A3 ARM64、CANN 9.1.0、Python 3.12 和补丁清单
2. 安装 PyTorch、torch_npu、SGLang 和 NPU 内核包
3. 重新构建已打补丁的 torch-memory-saver wheel
4. 安装 mbridge、Megatron-Bridge、Megatron-LM、MegatronAdaptor 和 TransformerEngineNPU
5. 固定 Triton-Ascend，回放 Slime 补丁并安装项目

### GLM-5.3 Flash 源码安装（CANN 9.2.0）

在 aarch64 节点激活 Python 3.12 环境并加载 CANN 9.2.0 Toolkit 和 NNAL/ATB 环境，然后在 `slime-ascend` 根目录执行以下命令。

```bash
export SLIME_ASCEND_DEPS_ROOT="$PWD/../glm53-deps"
bash scripts/quick_install.sh --glm5.3-flash a5  # A3 使用 a3
```

A5 容器安装使用专用 [CANN 9.2.0 Dockerfile](../../docker/Dockerfile.a5.glm53.cann920)，所需基础开发镜像与构建命令见[模型指南](../../models/glm5.3flash/README.md#a5-docker-镜像)。依赖目录用于存放源码和构建产物。

### 补丁构建

`build.sh` 根据 [MANIFEST.toml](../../patches/MANIFEST.toml) 检查补丁文件和 SHA256，然后在 Slime main@2026-10-22 子模块上按顺序应用补丁：

```bash
bash build.sh --dry-run
bash build.sh
```

`build.sh` 会清理并重置 `slime/` 子模块的工作区；运行前请保存该目录内自己的修改。

## 训练脚本与参数概览

完成环境配置后，可参照仓库中的 `scripts/run-*.sh` 启动训练。运行前使用 `python --version` 和 `npu-smi info` 检查环境。

## 常见问题排查

补丁冲突时，核对 [MANIFEST.toml](../../patches/MANIFEST.toml) 的基线与实际文件。Slime 构建失败时会打印失败的补丁路径并中止该次 `git am`。第三方仓库若仍处于 `git am` 冲突状态，可在对应仓库执行 `git am --show-current-patch=diff` 查看补丁。
