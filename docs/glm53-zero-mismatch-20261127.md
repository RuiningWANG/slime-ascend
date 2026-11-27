# GLM-5.3-Flash train–serving consistency on Ascend

The release baseline and the consistency changes are reviewed separately. Use
the corresponding Slime, SGLang and slime-ascend revisions together. The
slime-ascend review branch pins the matching Slime revision as a submodule.

## Release environment and source layout

Use the GLM-5.3 release Conda and CANN 9.2 environment, full BF16 checkpoint,
Geo3K data, and the release versions of Megatron-LM, MegatronAdaptor,
TransformerEngineNPU, sgl-kernel-npu, and the batch-invariance runtime.
Set `GLM53_RELEASE_ENV`, `HF_CHECKPOINT`, `PROMPT_DATA` and
`FULL_COMM_IFNAME` to the values for the target installation.

From the slime-ascend checkout, initialize the Slime submodule, then assemble
the launch layout in that checkout:

~~~bash
git submodule update --init slime
mkdir -p src
ln -s .. src/slime-ascend
git clone --branch 20261127_zero_mismatch_aligned https://github.com/RuiningWANG/sglang.git src/sglang
~~~

Link matching release source trees for Megatron-LM, MegatronAdaptor and
TransformerEngineNPU under `src/`. Provide the release batch-invariance
runtime at `dependencies/batch-invariant/`. The checked-in Ray and
TorchMemorySaver files are compatibility overrides. Overlay them on complete
copies of release Ray and TorchMemorySaver in the deployment checkout, and
build TorchMemorySaver against the release CANN. Keep the release installation
itself as the source of the clean dependencies.

## Launch and validation

Set `GLM53_NODE_IPS` to the space-separated communication addresses in rank
order. Run the same recipe and run ID on every node. Use `--print-command`
to inspect the effective arguments and `--check` to validate the inputs.

| Entry | Purpose |
| --- | --- |
| `script/run-aligned-quality-sharedtp1-once.sh` | Paired Geo3K inference quality with BF16 LM Head and a full TP1 shared expert. |
| `script/run-first3-save-sharedtp1-fixedckpt.sh` | Three-step online verification and native checkpoint save. |
| `script/run-long150-sharedtp1-fixedckpt.sh` | Resume the native 150-step recipe from its persisted checkpoint tracker. |

The optional private fast-storage wrappers use a local disk on the head node
and verified SSHFS mounts on workers. Supply `GLM53_FASTIO_BACKEND`,
`GLM53_FASTIO_DEADLINE` (a timezone-qualified timestamp),
`GLM53_FASTIO_TOOL_CACHE`, `GLM53_FASTIO_KNOWN_HOSTS` and
`GLM53_FASTIO_PASSWORD_FILE` for the allocation. Place the pinned RPMs listed
in `dependencies/fastio/tool-manifest.json` in `dependencies/fastio/packages/`.
Keep the host trust file and credentials in the deployment environment.

The acceptance run used 8 nodes, 64 NPUs, 45 layers and 288 routed experts.
Its three steps had nonzero gradients and zero logprob mismatches across
464,535 generated tokens. Geo3K scored 455/601 with the native build and
454/601 with the final consistency build. Checkpoint 2 was saved and its
structure was verified. The follow-on run reached the resource deadline during
initialization, leaving 3/150 steps completed and live resume unverified.
