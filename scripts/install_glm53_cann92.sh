#!/usr/bin/env bash
# Install the GLM-5.3 Flash source profile in an activated CANN 9.2 environment.
set -euo pipefail

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ASCEND_HOME_PATH:?Source the CANN 9.2 toolkit environment first}"
: "${SLIME_ASCEND_DEPS_ROOT:?Set a directory for fresh dependency checkouts and build products}"
case "${1:-}" in
    a3) sgl_soc=Ascend910_9382; ops_soc=ascend910_93; fla_soc=ascend910_93 ;;
    a5) sgl_soc=Ascend950; ops_soc=ascend950; fla_soc=ascend950 ;;
    *) echo "Usage: bash scripts/install_glm53_cann92.sh {a3|a5}" >&2; exit 2 ;;
esac
python -c 'import platform, sys; assert platform.machine() == "aarch64" and sys.version_info[:2] == (3, 12)'
grep -Eq '^version=9\.2\.0([.[:space:]]|$)' "$ASCEND_HOME_PATH/aarch64-linux/ascend_toolkit_install.info"
python "$repo/scripts/check_patch_manifest.py"
mkdir -p "$SLIME_ASCEND_DEPS_ROOT"
deps=$(cd "$SLIME_ASCEND_DEPS_ROOT" && pwd)
wheels="$deps/wheels"
mkdir -p "$wheels"

python "$repo/scripts/prepare_glm53_sources.py" "$deps"

python -m pip install wheel setuptools setuptools-scm setuptools-rust packaging ninja pybind11 pyyaml numpy scipy attrs decorator psutil einops expecttest nvdlfw-inspect
python -m pip install torch==2.10.0 torchvision==0.25.0 torch_npu==2.10.0.post7 triton-ascend==3.2.2

(
    cd "$deps/ops-transformer"
    git submodule update --init --recursive
    python -m pip install -r requirements.txt
    ops=mhc_pre_sinkhorn,mhc_post,mhc_pre_sinkhorn_backward,mhc_post_backward,pool_key_indexer
    bash build.sh --pkg --soc="$ops_soc" --vendor_name=custom --ops="$ops" --package-path "$ASCEND_HOME_PATH" -j"${MAX_JOBS:-8}"
    # Install the original CANN binaries to a user-selected OPP directory.
    installer=(build_out/*.run)
    [[ ${#installer[@]} = 1 && -f ${installer[0]} ]]
    mkdir -p "$deps/opp"
    bash "${installer[0]}" --quiet --install-path="$deps/opp"
    # The unsuffixed SDK package discovers the custom wheel through entry points.
    python -m pip install --no-build-isolation --no-deps ./torch_extension
    bash build.sh --torch_extension --ops="$ops" --vendor_name=custom
    bindings=(torch_extension/dist/*custom*.whl)
    [[ ${#bindings[@]} = 1 && -f ${bindings[0]} ]]
    python "$repo/scripts/stage_glm53_ops_wheel.py" "${bindings[0]}" "$wheels"
    python -m pip install --no-deps "$wheels/$(basename "${bindings[0]}")"
)
(
    cd "$deps/sgl-kernel-npu"
    git submodule update --init --recursive
    for component in kernels deepep memory-saver; do
        case "$component" in
            kernels) bash build.sh -a kernels "$sgl_soc"; wheel=sgl_kernel_npu ;;
            deepep) bash build.sh -a deepep "$sgl_soc"; wheel=deep_ep ;;
            memory-saver) bash build.sh -a memory-saver; wheel=torch_memory_saver ;;
        esac
        python -m pip install --no-deps output/"$wheel"-*.whl
    done
)
(
    # DeepEP's public Python API imports the extension by its top-level name.
    python - <<'PY'
import importlib.util
import pathlib
import sysconfig

if importlib.util.find_spec("deep_ep_cpp") is None:
    site = pathlib.Path(sysconfig.get_path("platlib"))
    extension, = (site / "deep_ep").glob("deep_ep_cpp*.so")
    (site / extension.name).symlink_to(extension.relative_to(site))
PY
    cd "$deps/flash-linear-attention-npu"
    git submodule update --init --recursive
    python -m pip install -r requirements.txt
    FLA_NPU_SOC="$fla_soc" python scripts/build_wheel.py
    python -m pip install --no-deps dist/flash_linear_attention_npu_"$1"-*.whl
)
SGLANG_BUILD_RUST_EXTS=none python -m pip install --no-build-isolation -e "$deps/sglang/python"
python -m pip install --no-build-isolation --no-deps -e "$deps/Megatron-LM" -e "$deps/MegatronAdaptor" -e "$deps/TransformerEngineNPU"
if [[ -e "$repo/.git" ]]; then
    git -C "$repo" submodule update --init --recursive slime
else
    # A Docker build context contains the patch sources without the host's Git metadata.
    slime_url=$(git config -f "$repo/.gitmodules" --get submodule.slime.url)
    slime_base=$(python -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["upstream_commit"])' "$repo/patches/MANIFEST.toml")
    git clone --no-checkout "$slime_url" "$repo/slime"
    git -C "$repo/slime" checkout --detach "$slime_base"
    git -C "$repo/slime" submodule update --init --recursive
fi
(cd "$repo" && bash build.sh --skip-submodule)
python -m pip install --no-build-isolation -e "$repo/slime"
