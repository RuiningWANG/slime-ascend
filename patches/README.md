# slime-ascend 补丁目录

本目录包含 slime-ascend 相对于上游 THUDM/slime 的所有改动，按性质分类。

## 目录结构

```
patches/
├── npu-compat/       # NPU 硬件差异性适配（永久性）
├── upstream-fixes/   # 上游 bug 的临时 workaround（预期上游修复后删除）
├── features/         # Slime 模型功能补丁
├── third_party/      # 外部依赖的固定版本补丁
└── MANIFEST.toml     # 补丁清单和元数据
```

## 补丁分类

### npu-compat/ - NPU 硬件兼容性适配

**特点**：
- 永久性维护
- 与上游版本无关的硬件差异
- 主要涉及设备标识、环境变量、资源分配

**示例**：
- Ray 资源分配从 `num-gpus` 到 NPU 资源
- 环境变量从 `CUDA_VISIBLE_DEVICES` 到 `ASCEND_RT_VISIBLE_DEVICES`
- 设备同步从 `torch.cuda` 到 `torch.npu`

### upstream-fixes/ - 上游临时修复

**特点**：
- 临时性维护
- 预期上游修复后删除
- 包含模型支持扩展、bug 修复等

**示例**：
- `model_provider.patch` - 扩展模型支持

**维护策略**：
- 每次同步上游时检查是否已修复
- 已修复的补丁从此目录删除
- 在 `MANIFEST.toml` 中跟踪上游 issue

### features/ 和 third_party/

GLM-5.3 Flash 分别维护 Slime、Megatron-LM、SGLang 和 sgl-kernel-npu 的补丁。Slime 补丁位于 [features/](features/glm5.3-flash.patch)，依赖仓库的补丁位于各自的 `third_party/<仓库>/`。每份模型补丁的 `Base-commit` 对应 [MANIFEST.toml](MANIFEST.toml) 的 `base_commit`。

CANN 9.2 安装脚本按目标仓库与实际检出的 commit 共同选择依赖补丁。Megatron 在同一基线的 dev 补丁之后应用优化器补丁。SGLang 与 sgl-kernel-npu 选择各自 9.2 配置的补丁系列。ops-transformer 使用指定版本的上游源码并在 wheel 暂存时调整 Python 导入。各组件的完整 SHA 和源码地址见[补丁清单](MANIFEST.toml)，打包入口见[模型指南](../models/glm5.3flash/README.md)。

## 应用补丁

在仓库根目录执行以下命令，将 Slime 补丁应用到子模块。CANN 9.2 第三方补丁由 GLM-5.3 安装入口应用。

```bash
# 构建流程
./build.sh
```

构建脚本先校验 MANIFEST 中所有补丁文件及 SHA256；缺失或不匹配时在重置 submodule 前失败。

构建脚本会：
1. Checkout `slime` submodule 到指定 commit
2. 按 `MANIFEST.toml` 顺序使用 `git am` 依次应用 `npu-compat`、`upstream-fixes`、`features` 中的 Slime 补丁
3. 将 GLM-5.3 Flash 的模型指南和算子模块安装到子模块的 `slime_plugins/models/glm5_next/`，并验证构建结果

## 维护补丁

### 添加新补丁

```bash
# 1. 在 slime 工作区中修改并提交文件
cd slime
# ... 修改代码 ...
git add <files>
git commit -s -m "[Ascend] describe the change"
git format-patch -1 HEAD --stdout > ../patches/npu-compat/new_feature.patch

# 2. 更新 MANIFEST.toml
# 追加 [[patches]] 条目并登记稳定 ID、issue、base commit 和 SHA256
```

### 更新现有补丁

在目标仓库的开发分支找到该补丁对应的提交。修改前先核对提交主题和涉及的文件；如果目标提交之后还有其他提交，使用交互式 rebase 编辑目标提交。重新导出该提交，保留补丁说明中的 `Base-commit`，并更新清单中的 SHA-256。然后从同一上游基线重放完整系列，检查后续补丁是否仍可应用。

### 同步上游

Slime 升级时，同步更新 slime-ascend 仓库的 `slime` 子模块指针、清单中的 `upstream_commit` 和对应补丁的 `base_commit`。CANN 9.2 第三方源码由 `[glm53.sources]` 管理。已打补丁的仓库通过补丁 ID 选择上游基线，同一目标和基线的所有补丁按清单顺序应用；直接使用上游源码的组件通过 `commit` 指定完整提交 SHA。

从仓库根目录检查清单和 Slime 应用顺序：

```bash
python scripts/check_patch_manifest.py
bash build.sh --dry-run
```
