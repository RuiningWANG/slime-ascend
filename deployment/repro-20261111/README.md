# GLM-5.3 shared training source update

`versions.json` pins the component branches and commits. The outer repository pins the matching Slime submodule commit.

The update preserves the existing NPU `BASH_ENV` and CANN compiler forkserver fixes. A3 sparse MLA now uses official `cann_ops_transformer.ops` forward/backward with the framework layout adapter. The operator repository is the unchanged upstream tree at the recorded SHA. This source update does not rebuild or replace installed CANN operators or Python wheels; install the matching package before running the new SMLA path in an older image.

The shared-host launch recipes retain their environment-specific paths, node addresses, model and data settings. Review those settings before using a different allocation or the `/opt/src` container environment. Credentials, Ray tokens, caches, logs, generated build outputs and experimental candidate files are not published. Existing profiling scripts are retained.

The sgl-kernel-npu source already matches the pinned private commit. Its generated `config.ini` is not a source update.

Validation for this source publication: 38 CPU compatibility tests, Python parsing, shell syntax, and staged whitespace checks. Historical eight-host numerical/training evidence belongs to the recorded run; no new training run is implied.
