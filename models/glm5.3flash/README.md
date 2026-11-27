# GLM-5.3 Flash on Ascend

本配置用于 CANN 9.2.0、Linux aarch64 和 Python 3.12。仓库根目录的 `build.sh` 按清单将 Slime 补丁应用到子模块，并将 `operators/<算子组>/` 中的模块安装到 `slime_plugins.models.glm5_next`。打包后的 Slime 可独立导入这些模块。上游 CUDA 配置见 [Slime 模型文档](https://github.com/THUDM/slime/blob/18868a79b0b083a093798a94ed0f8cb4f71b9c14/slime_plugins/models/glm5_next/README.md)。

## 源码版本

Slime 的上游提交由 `upstream_commit` 指定。依赖仓库的地址和版本由 `[glm53.sources]` 指定，其中 `patch` 引用对应补丁的 `base_commit`。完整 SHA 和应用顺序见[补丁清单](https://gitcode.com/Ascend/slime-ascend/blob/dev/patches/MANIFEST.toml)。

| 组件 | 本配置的改动 |
| --- | --- |
| Slime | 在 dev 通用补丁后应用 `features/glm5.3-flash.patch` |
| Megatron-LM | 在 dev 的两份 Megatron 补丁后应用优化器补丁 |
| SGLang | GLM5-next 图像、MoE 激活、通信和图捕获 |
| sgl-kernel-npu | TorchMemorySaver、图捕获和 DeepEP 构建 |
| ops-transformer | 在 custom wheel 暂存目录中调整 builder 导入 |
| flash-linear-attention-npu | 按目标 SoC 从源码构建 KDA、Conv1d 所需的 wheel |
| MegatronAdaptor | 使用上游源码 |
| TransformerEngineNPU | 使用上游源码 |

## 安装

在 `slime-ascend` 仓库的 `dev` 分支根目录下，激活 Python 虚拟环境并加载目标节点的 CANN 9.2 Toolkit、配套算子包及 NNAL/ATB 环境。源码构建需要 Git（含可用的提交者身份供 `git am` 使用）、C/C++ 编译器、CMake 和 Rust 工具链。选择可写的空依赖目录。安装脚本会下载源码、构建 SoC 对应的算子和 Python 包。

```bash
export SLIME_ASCEND_DEPS_ROOT="$PWD/../glm53-deps"
bash scripts/quick_install.sh --glm5.3-flash a5
export FLA_NPU_DISABLE_PTH=1 TORCH_DEVICE_BACKEND_AUTOLOAD=1
python_site=$(python -c 'import sysconfig; print(sysconfig.get_path("platlib"))')
source "$SLIME_ASCEND_DEPS_ROOT/opp/vendors/custom_transformer/bin/set_env.bash"
source "$python_site/deep_ep/vendors/hwcomputing/bin/set_env.bash"
source "$python_site/fla_npu/opp/vendors/fla_npu_transformer/bin/set_env.bash"
export MEGATRON_PATH="$SLIME_ASCEND_DEPS_ROOT/Megatron-LM"
```

A3 构建使用参数 `a3`。每个运行节点都要使用对应 SoC 的同一套环境，并在启动 Ray 前加载 CANN、自定义 vendor 和已安装的 FLA、DeepEP 环境。

### A5 Docker 镜像

基础开发镜像需包含 Linux ARM64、CANN 9.2.0 Toolkit、950 算子包、NNAL/ATB、Python 3.12 及上述编译工具链。将 `CANN_BASE_IMAGE` 设为可拉取的镜像引用（建议指定 digest），在 `slime-ascend` 根目录构建。Dockerfile 调用与裸机相同的 A5 安装入口。

```bash
docker build --platform linux/arm64 \
  --build-arg CANN_BASE_IMAGE="${CANN_BASE_IMAGE:?请指定 A5 CANN 9.2.0 开发镜像}" \
  -f docker/Dockerfile.a5.glm53.cann920 -t slime-ascend:glm53-cann920-a5 .
```

基础镜像的 Toolkit 或 ATB 环境脚本在其他位置时，分别通过 `--build-arg CANN_ENV=...`、`--build-arg ATB_ENV=...` 指定。运行时按宿主机的 Ascend 容器运行时配置挂载八卡、驱动、模型和数据。进入容器后加载 Toolkit 和 ATB，再按安装一节设置 custom_transformer、DeepEP 和 FLA 的 vendor 环境。

## 算子与 wheel

稀疏注意力使用 CANN SparseFlashAttention。A3 使用合并的 KV 梯度，A5 分别计算 K 和 V 的梯度并求和。训练端的 KPool 使用 `sgl-kernel-npu` 的索引和缓存接口，与 SGLang 推理端一致。CP 下，KDA 和 Conv1d 在头数可整除时通过 all-to-all 交换 token 与头分片。

训练自定义包包含 mHC pre/post 及其反向、SGLang KPool 依赖的 PoolKeyIndexer。SDK 提供其余基础 API。安装入口从固定源码构建 vendor 和原始 Python wheel，再由 `scripts/stage_glm53_ops_wheel.py` 处理 wheel 中的 `op_builder/builder.py` 并重新生成 `RECORD`。包名、算子入口及原生二进制与上游构建产物一致。

## 训练组件测试

在 `slime/` 子模块目录运行稀疏索引、视觉权重映射、MoE 分块和 KDA 的 CPU 测试。

```bash
python -m pytest -q tests/test_glm5_next_sparse_index_compaction.py tests/test_glm5_next_vision_merger.py \
    tests/test_glm5_next_moe_chunks.py tests/test_glm5_next_kda_reference.py
```

在 A3 或 A5 上运行 KDA 和稀疏注意力的前反向数值测试。

```bash
GLM53_TEST_NPU=1 python -m pytest -q tests/test_glm5_next_kda_reference.py tests/test_glm5_next_sfa.py
```

## 八卡 A5 示例

强化学习训练依赖配套的 SGLang 和 sgl-kernel-npu 补丁，用于路由回放、推理侧 KPool 和图捕获。

示例使用单机八卡，训练并行配置为 TP1、PP4、CP2、EP2，推理并行配置为 TP8。在各节点启动 Ray 前，加载 custom_transformer、DeepEP 和 FLA 的环境，并设置 `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1` 和 `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0`。脚本向当前 Ray 集群提交作业，各工作节点须能通过相同路径访问模型和数据。Geo3K 默认上下文 8192、回答上限 4096，每次 rollout 执行一步，共三步。

```bash
export HF_CHECKPOINT=/path/to/GLM-5.3-Flash
export PROMPT_DATA=/path/to/geo3k/train.jsonl
export MULTIMODAL_KEYS='{"image":"images"}'
export CHAT_TEMPLATE_KWARGS="$(cat /path/to/geo3k/chat_template_kwargs.json)"
bash scripts/run-glm5.3-flash-npu.sh
```

DAPO-Math 与 Geo3K 共用模型权重。文本示例默认使用 16384 上下文、8192 回答上限和 8192 个 token 的单卡打包预算，MoE 分块为 512/1024。指定 `ROLLOUT_MAX_RESPONSE_LEN=4096` 可缩短回答。两个示例都使用 `deepscaler` 奖励函数，可通过 `RM_TYPE` 覆盖。

```bash
export PROMPT_DATA=/path/to/dapo-math/train.jsonl
unset MULTIMODAL_KEYS CHAT_TEMPLATE_KWARGS
bash scripts/run-glm5.3-flash-npu.sh --dapo-math
```
