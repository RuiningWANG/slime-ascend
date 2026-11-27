# 开发指南

## 参数说明

slime 的参数分为三类：

1. **megatron 参数**：slime 会读取 `PYTHONPATH` 中 megatron 的所有参数，可通过 `--tensor-model-parallel-size 2` 等方式配置；
2. **sglang 参数**：支持环境中安装的 sglang 的所有参数，需以 `--sglang-` 起始传入，例如 `--sglang-mem-fraction-static`；
3. **slime 自身的参数**：见 [slime/utils/arguments.py](../slime/slime/utils/arguments.py)。

训练、保存与评估的具体指令和参数设定请直接参考 slime 主仓文档
[快速开始指南](https://github.com/THUDM/slime/blob/main/docs/zh/get_started/quick_start.md)。

## 补丁维护

Slime 与第三方仓库的改动以补丁维护。GLM-5.3 Flash 的算子适配文件保存在 `models/glm5.3flash/operators/`，由构建脚本安装到子模块。
补丁分类原则、添加/更新补丁的标准流程、上游同步策略，**以 [patches/README.md](../patches/README.md) 为准**；
逐 patch 的用途说明与上游跟踪信息见 [patches/MANIFEST.toml](../patches/MANIFEST.toml)。

新增补丁时，在补丁说明中填写完整的 `Base-commit`，并在 `patches/MANIFEST.toml` 的 `[[patches]]` 条目登记 `target`、`category`、`file`、`base_commit` 和 `sha256`。同一基线的补丁依照清单顺序应用。

```bash
python scripts/check_patch_manifest.py
bash build.sh --dry-run
```

## 调试

调试技巧请参考 slime [debug 指南](https://github.com/THUDM/slime/blob/main/docs/zh/developer_guide/debug.md)。
