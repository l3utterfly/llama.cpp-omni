# Layla bindings

Everything Layla adds to llama.cpp-omni lives in this directory; no file outside it is
changed, so syncing `upstream/ios-omni` stays a plain merge.

Layla runs MiniCPM-V 4.7 35B-A3B (MoE-SSD expert streaming) on iOS through this
code. It is built into `minicpm.framework` by `modules/minicpm/ios/` in the Layla repo,
which adds llama.cpp-omni and then this directory.

| File | What it is |
|---|---|
| `minicpm_capi.h` / `.cpp` | The `minicpm_*` C ABI the React Native bridge calls, and the memory policy that sizes the expert window (a port of the MiniCPM-V iOS app's `MiniCPMV47Const`). |
| `vendor/minicpm-v-apps/` | The MiniCPM-V iOS app's engine bridge (`MBMtmd`), MIT, see `LICENSE`. It holds the tuning the app ships: eager image encode, mmproj release, expert prewarm, adaptive ubatch, DSpark. |
| `CMakeLists.txt` | The `minicpm_layla` static library. |

## Vendored bridge

Copied from [OpenBMB/MiniCPM-V-Apps](https://github.com/OpenBMB/MiniCPM-V-Apps)
`MiniCPM-V-demo/MTMDWrapper/Bridge/` at commit `fa9d837`, whose `llama.cpp-omni`
submodule is pinned to `96bbf61`, the revision this branch is based on. The bridge and
the engine must stay on matching revisions: the bridge hand-declares the
`common_speculative_*` structs and checks their sizes at compile time.

Layla's edits are marked `// LAYLA:` (`git grep -n "LAYLA" vendor/`):

- The run log (every log line plus each prompt and reply) is only written with
  `LAYLA_MINICPM_RUN_LOG=1`, to the temporary directory instead of Documents.
- The prewarm table is written to the temporary directory instead of Documents.

To update: merge `upstream/ios-omni`, copy `MBMtmd.h`, `MBMtmd.mm` and
`MBPrewarmTable.h` from the MiniCPM-V-Apps commit that pins the merged revision,
reapply the `LAYLA` edits, and check `MiniCPMV47Const` in
`MiniCPM-V-demo/Sources/Settings/ModelConst/MiniCPMModelConst.swift` for changed
defaults to port into `minicpm_capi.cpp`.
