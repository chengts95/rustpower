# Actual-path integration results

Persistent worktree `/home/cts/workspace/rustpower-amd-supervariable`, branch `codex/ybus-amd-supervariable`, base1ad70a7. No commits/pushes. Original main clean. Restored prototype sources/patch/data hashes verified after reboot; old/tmp source/binaries/P-core output were lost. New code and outputs are persistent.

Dataset `integration-results/cpu0-20261026-085534`: actual CPU0 verified before/after every solve, Intel Core Ultra9 288V P core; performance power profile, EPP performance; unchanged power snapshots per round. Same release binary and original initial voltage for both policies. BTF1 default and BTF0 compared separately;200 samples per path/mode/case/config after warmup.4800 valid records. User-controlled CPU/power settings were not changed.

BTF path clarification: baseline common.btf=1 executes general KLU BTF via klu_l_analyze. Optimized common.btf=1 retains Symbolic.do_btf=1 but directly supplies the proven single block and full structural rank; it does not execute general BTF. Therefore BTF1 savings include bypassing BTF, graph preparation and seeded AMD, not AMD alone. BTF0 has general BTF absent on both paths and still includes graph preparation plus seeded AMD improvements.

Default BTF1, cold NR medians (baseline -> enabled):39:21.395->18.450us(-13.77%);118:86.407->74.430us(-13.86%);9241:18.004->16.279ms(-9.58%). Reused:39:10.530->10.721us;118:31.011->30.968us;9241:11.419->11.650ms. Reuse is not a demonstrated speedup.

BTF0 cold:39:19.609->17.785us;118:80.463->73.850us;9241:16.998->16.188ms(-4.77%). Reused:39:10.562->10.253us;118:30.196->29.658us;9241:11.258->11.559ms.

Same-batch9241/default BTF1: actual Ybus/Sbus/V permutation0.495ms; ordering4.019->2.225ms; J symbolic0.266->0.258ms. Median of the per-sample sum of permutation+ordering+J symbolic4.800->2.988ms. Ordering saving1.794ms exceeds one permutation cost0.495ms by3.62x. BTF0 ordering saving0.933ms versus permutation0.494ms; total preparation3.905->2.958ms. This is same-machine/core/power-mode arithmetic, not subtraction of historical E-core data.

All candidates converged with same iteration counts4/5/6. Maximum complex voltage error4.5356643598081007e-13. Existing AMD calls: baseline1 per cold solve,0 reused; optimized path0 (seeded core called by integrated preparation), and cache generation stays fixed on reused solves. Default BTF flag preserved in resulting Symbolic. Fill L/U each baseline/new:39:338/336;118:768/748;9241:116171/115817. Extra retained exact Ybus key:1368/4760/375176bytes; no duplicate full J graph in the ordering path.

Tests already completed:86 root library tests+4 new native/ECS tests;5 KLU crate tests;47 Python regressions and comprehensive NewtonSolver API checks. Two pre-existing performance tests ignored. No overlap implementation: independent preparation inputs delivered, reusable low-overhead numerical executor remains the missing piece.

Remote0.5.5 inspected and fetched at a14032021836475355a36d11de86ac6c8f43126e. It changes legacy pandapower load deserialization and changelog only. The exact remote delta (legacy load serde defaults, its regression and changelog;22 lines in2 files) was subsequently applied locally with git apply, without conflict, on the persistent integration branch. This integrates remote0.5.5 content without creating a merge commit or changing HEAD ancestry. Patch saved as remote-0.5.5.patch. No version bump/release/commit/push.

After applying remote0.5.5 content: root KLU library87 passed/2 pre-existing ignored (including new legacy-load test), Ybus native/ECS integration4 passed. Logs:0.5.5-klu-regressions.log and0.5.5-ybus-regressions.log. Performance dataset predates only the unrelated deserialization change; no performance rerun was added.
