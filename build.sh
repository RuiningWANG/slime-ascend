#!/bin/bash
# slime-ascend 构建脚本
#
# 功能：
#   1. 重置 slime submodule 到指定 commit
#   2. 按 MANIFEST.toml 顺序应用 Slime 补丁
#   3. 生成可用的 slime-ascend

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLIME_DIR="${REPO_ROOT}/slime"
PATCHES_DIR="${REPO_ROOT}/patches"
python3 "${REPO_ROOT}/scripts/check_patch_manifest.py"
UPSTREAM_COMMIT=$(python3 - "${PATCHES_DIR}/MANIFEST.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

print(tomllib.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["upstream_commit"], end="")
PY
)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;36m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[!]${NC} $*"
}

log_error() {
    echo -e "${RED}[✗]${NC} $*"
}

# 显示帮助
show_help() {
    cat << EOF
slime-ascend 构建脚本

用法:
    ./build.sh [选项]

选项:
    --clean         清理 slime 目录并重新构建
    --dry-run       仅显示将要执行的操作，不实际构建
    --skip-submodule 跳过 submodule 更新（用于快速重建）
    --help          显示此帮助信息

构建流程:
    1. 初始化/更新 slime submodule
    2. 应用 patches/npu-compat/ 补丁
    3. 应用 patches/upstream-fixes/ 补丁
    4. 应用 patches/features/ 补丁
    5. 验证构建结果

构建产物:
    - slime/                打满补丁的 slime 代码（即 submodule 工作区）
EOF
}

# 解析命令行参数
CLEAN_BUILD=false
DRY_RUN=false
SKIP_SUBMODULE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --skip-submodule)
            SKIP_SUBMODULE=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "slime-ascend 构建系统"
echo "=========================================="
echo ""

# 步骤 1: 清理 slime 目录
if [ "$CLEAN_BUILD" = true ]; then
    log_info "清理 slime 目录..."
    if [ "$DRY_RUN" = false ]; then rm -rf "${SLIME_DIR}"; fi
    log_success "slime 目录已清理"
    echo ""
fi

# 步骤 2: 初始化 submodule
if [ "$SKIP_SUBMODULE" = false ]; then
    log_info "初始化/更新 slime submodule..."

    if [ ! -e "${SLIME_DIR}/.git" ]; then
        log_info "首次初始化 submodule..."
        if [ "$DRY_RUN" = false ]; then
            git submodule update --init --recursive "${SLIME_DIR}"
        else
            echo "  [DRY-RUN] git submodule update --init --recursive"
        fi
    else
        log_info "更新已存在的 submodule..."
        if [ "$DRY_RUN" = false ]; then
            cd "${SLIME_DIR}"
            git clean -xfd
            git checkout .
            git fetch origin
            git checkout --detach "${UPSTREAM_COMMIT}"
            cd "${REPO_ROOT}"
        else
            echo "  [DRY-RUN] git fetch && checkout pinned commit"
        fi
    fi

    log_success "Submodule 准备完成"
    echo ""
else
    log_warn "跳过 submodule 更新"
    echo ""
fi

# The manifest defines the complete order, including patches from different categories.
patch_list=$(python3 - "${PATCHES_DIR}/MANIFEST.toml" <<'PY'
import sys
import tomllib
from pathlib import Path

manifest = Path(sys.argv[1])
data = tomllib.loads(manifest.read_text(encoding="utf-8"))
sys.stdout.reconfigure(newline="\n")
for patch in data["patches"]:
    if patch["target"] == "slime" and patch["base_commit"] == data["upstream_commit"]:
        print((manifest.parent / patch["category"] / patch["file"]).as_posix())
PY
)
while IFS= read -r patch; do
    [[ -n "$patch" ]] || continue
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] 应用 $(basename "$patch")"
    elif git -C "$SLIME_DIR" am "$patch"; then
        log_success "已应用 $(basename "$patch")"
    else
        git -C "$SLIME_DIR" am --abort || true
        log_error "补丁应用失败: $patch"
        exit 1
    fi
done <<< "$patch_list"

# Install the Ascend model guide and operators into the Slime model package.
model_dir="${REPO_ROOT}/models/glm5.3flash"
model_dest="${SLIME_DIR}/slime_plugins/models/glm5_next"
for model_file in "$model_dir/README.md" "$model_dir"/operators/*/*.py; do
    if [ "$DRY_RUN" = true ]; then
        echo "  [DRY-RUN] install $(basename "$model_file")"
    else
        install -m 0644 "$model_file" "$model_dest/"
    fi
done

# 步骤 4: 验证构建
log_info "验证构建结果..."
if [ "$DRY_RUN" = false ]; then
    # 检查关键文件是否存在
    critical_files=(
        "${SLIME_DIR}/slime/__init__.py"
        "${SLIME_DIR}/setup.py"
    )

    all_exist=true
    for file in "${critical_files[@]}"; do
        if [ ! -f "$file" ]; then
            log_error "关键文件缺失: $file"
            all_exist=false
        fi
    done

    if [ "$all_exist" = false ]; then
        log_error "构建验证失败"
        exit 1
    fi
else
    echo "  [DRY-RUN] 验证关键文件存在"
fi
log_success "构建验证通过"
echo ""

echo "=========================================="
echo "构建完成！"
echo "=========================================="
echo ""
if [ "$DRY_RUN" = false ]; then
    echo "构建产物位置: ${SLIME_DIR}"
    echo ""
    echo "安装方法:"
    echo "  cd ${SLIME_DIR}"
    echo "  pip install -e ."
    echo ""
    echo "或者设置 PYTHONPATH:"
    echo "  export PYTHONPATH=\"${SLIME_DIR}:\$PYTHONPATH\""
else
    echo "[DRY-RUN] 模拟构建完成"
fi
echo ""
