# GLM-5.3-Flash 全尺寸：8 台物理机、12 个容器、128 die

入口已部署到共享代码目录：

```bash
/mnt/share/l00937981/glm_53_flash_slime/slime-ascend/lh_scripts/run_glm53_8host_full288.sh
```

脚本自动使用 `/mnt/share/l00937981/conda/envs/glm53-flash-slime` 和项目 `activate.sh` 的 CANN，不需要另外激活旧环境。保留新单机脚本的 AscendC 算子和 packed KDA 路径，参考旧八机脚本的全尺寸架构与并行配置。原单机 Top16 入口保持原样。

## 配置

| 项目 | 默认 smoke | `PROFILE=long` |
|---|---|---|
| 模型 | 45 层、288 专家、top-k 8 | 相同 |
| 权重 | `/mnt/share/w00936111/weights/GLM-5.3-Flash-BF16`，120 分片 | 相同 |
| 训练 | 128 die，TP1 / PP4 / CP8 / EP32，DP4 | 相同 |
| 流水线层数 | 12 + 11 + 11 + 11 | 相同 |
| 推理 | 2 个 TP64 文本推理引擎 | 相同 |
| Global batch | 16（2 prompt × 8） | 128（16 prompt × 8） |
| 上下文 / 最大响应 | 16384 / 8192 | 65536 / 32768 |
| Rollout 数 | 1，每次 1 个 optimizer step | 1000，每次 1 个 optimizer step |
| 训练 / 推理切换 | colocate + offload；训练 optimizer CPU offload 0.8 | 相同 |
| 路由 / 传输 | R3 + TIS；object-store | 相同 |
| Checkpoint | 不保存 | 不保存 |

默认小 batch 用于首轮验证，模型维度完全不裁剪。`NUM_ROLLOUT`、`ROLLOUT_BATCH_SIZE`、`N_SAMPLES_PER_PROMPT`、`GLOBAL_BATCH_SIZE`、`CONTEXT_LENGTH`、`RESPONSE_LENGTH`、`EP_SIZE`、`CP_SIZE`、`HF_CHECKPOINT`、`PROMPT_DATA`、`LOG_ROOT` 可显式覆盖；脚本会检查并行与 batch 的整除关系。

## 当前资源分配

| 宿主机组 | 容器地址 | 设备文件的物理 die 编号 |
|---|---|---|
| 1 | 80.5.27.69 / 80.5.27.70 | 0–7 / 8–15 |
| 2 | 80.5.27.71 / 80.5.27.73 | 0–7 / 8–15 |
| 3 | 80.5.27.74 / 80.5.27.75 | 0–7 / 8–15 |
| 4 | 80.5.27.76 / 80.5.27.77 | 0–7 / 8–15 |
| 5–8 | 80.5.27.237 / .238 / .239 / .240 | 每容器 0–15 |

上半机容器中，ACL 将物理 8–15 重新编号为逻辑 0–7。因此 `ASCEND_RT_VISIBLE_DEVICES` 使用容器内逻辑编号；物理编号只用于分配核对与日志。实测 `.70` 设置逻辑 0–7 可见 8 die，设置 8–15 则可见 0 die。

框架参数中的 `--actor-num-nodes 16 --actor-num-gpus-per-node 8` 表示 **16 个逻辑工作组**，与 12 个 Ray 节点、8 台物理机相区分。全部 128 die 分成连续的 8-die 组；整机容器容纳两个组。第一组 TP64 引擎使用 8 个半机容器，第二组使用 4 个整机容器，每个引擎都有 8 个工作进程。

## 检查与启动

先在任一容器检查：

```bash
cd /mnt/share/l00937981/glm_53_flash_slime/slime-ascend
bash lh_scripts/run_glm53_8host_full288.sh --check
bash lh_scripts/run_glm53_8host_full288.sh --check-env
bash lh_scripts/run_glm53_8host_full288.sh --print-command
```

实际启动时，**在全部 12 个容器各执行一次**下面的相同命令。每次运行使用一个新的、完全相同的 run ID；先启动 `.69`，再启动其他容器也可以，入口会等待集群就绪：

```bash
bash /mnt/share/l00937981/glm_53_flash_slime/slime-ascend/lh_scripts/run_glm53_8host_full288.sh \
  --run-id 20261104_full288_smoke1
```

首轮通过后，长跑命令同样需要在全部 12 个容器执行，并保持所有配置一致：

```bash
PROFILE=long bash /mnt/share/l00937981/glm_53_flash_slime/slime-ascend/lh_scripts/run_glm53_8host_full288.sh \
  --run-id 20261104_full288_long1
```

空参数只打印帮助，不启动任务。脚本包含空闲 NPU 检查、每容器互斥锁、run ID 防重用、节点与真实 die ID 核验。正式加载模型前，会在 Ray 中做一次不分配 NPU 张量的 128-slot 资源映射检查，并输出 `rank-map.json`。

头节点为 `.69`，GCS / dashboard 端口为 25400 / 25401。日志位于：

```text
/mnt/share/l00937981/glm_53_flash_slime/cache/work/glm53-full128/<run-id>/
```

每个容器有独立 `node-<IP>/launcher.log`、`ray-start.log`、`npu-before.txt`、`identity.json` 和 `cleanup.json`；头节点另保存实际 `command.sh`、`contract.json`、`rank-map.json` 和 `head-exit-code`。

停止时，在头节点入口按 Ctrl-C；各容器收到头节点退出标记后，只清理带本次 run ID 的自属进程。不要执行全局 `ray stop`。启动后保留各容器入口进程运行，或自行用已有的终端会话管理器托管它们。

## 调度修正与验证边界

SGLang 原端口分配按逻辑节点维护游标。在整机容器容纳两个 8-die 进程时，这两个进程可能同时拿到尚未绑定的同一端口。已将游标改为按实际 IP 维护，补丁为 `engine_group.patch`。原文件备份：

```text
runtime/glm53-full128-engine_group.before-c26697de.py
```

原文件 SHA256：`c26697de57d15a067251024380f9392efe1220d04cbb9f1a53c6dc652c5d1083`。应用补丁前已检查原文件未被并发修改。入口会拒绝在缺少该修正时启动。

已执行的检查见 `validation.json`：Bash/Python 语法、12 容器设备分配与静态配置、全尺寸 checkpoint 分片、原生参数解析、上半机逻辑编号映射、端口冲突复现与回归。原生检查确认 NPU 未初始化。

**本次没有启动 Ray 集群、HCCL 通信验证、模型加载或训练。** 脚本与参数检查通过不代表首个 optimizer step 已通过；隔壁旧环境曾出现的 HCCL 问题也不能据此视为已解决。当前共享盘空间较少，默认不保存大 checkpoint 或训练张量转储。
