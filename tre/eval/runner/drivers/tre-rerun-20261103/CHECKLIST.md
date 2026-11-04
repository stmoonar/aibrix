> [repo] Repo copy of the checklist as written on 2026-11-03. Only the data root is replaced by `$CAMPAIGN_DATA_ROOT`.
> In this checkout `$T` is `<repo>/tre`; the specs are in `$T/eval/runner/campaigns/`, the driver, guards and make_rep2.sh in
> `$T/eval/runner/drivers/tre-rerun-20261103/`, the reference registry `0ab5...` in `$T/eval/registries/`; `$C` (logs, STOP file) is
> `$CAMPAIGN_WORK_DIR` (default `$CAMPAIGN_DATA_ROOT/campaign-tre-rerun-draft-20261103`). The text below keeps the original names.

# TRE 重跑（新 TSS 参数）开跑清单（2026-11-03，约 16:00）

目录：`C=$CAMPAIGN_DATA_ROOT/campaign-tre-rerun-draft-20261103`，`T=$CAMPAIGN_DATA_ROOT/aibrix/tre`。全部命令在 76 上跑。
spec：rep 1 `campaign-tre-rerun-r3e-20261103.yaml`，rep 2 `campaign-tre-rerun-r3e-rep2-20261103.yaml`（由 `make_rep2.sh` 从 rep 1 生成）。
每个 rep 7 臂（只有 TRE），约 1.8 h；两个 rep 约 3.6 h。来源、差异、ETA 见 spec 头注释。

标【批准】的步骤要先得到用户同意。

## A. 发布参数

1. 确认前一批已结束（只读）：
   `cat $CAMPAIGN_DATA_ROOT/TRE_EXCLUSIVE_WINDOW`（应不存在）；`python3 -c "import json;print(json.load(open('$CAMPAIGN_DATA_ROOT/campaign-bl-rep2-r3e-20261103-run/status.json'))['state'])"`（应为 `done`）；`pgrep -af '^python3 .*campaign\.py run'`（应为空）。
2. 【批准】用户确认：选定的候选参数，同意原子发布 θ / τ_crit / τ_high（以及 λ / w_p），同意重跑 TRE 臂（HANDOFF"待用户回答"第 1 项）。
3. 离线校验新参数（不访问集群）：
   `cd $T && PYTHONPATH=common:deploy python3 deploy/scripts/release/params_set.py --registry-file $C/registry-live-0ab5352747bdd70e.yaml --set dsqwen-7b.trs.theta_m=<v> --set dsqwen-7b.trs.tau_crit=<v> --set dsqwen-7b.trs.tau_high=<v> …（3 个模型）`
   - λ（`trs.lambda_wait`）> 20 时会被拒（`registry_params.py` 的 `_TRS_BOUNDS` 上限 20）。要先改上限并走评审合入；main 因此前进，第 9 步要用新的 `MAIN_SHA`，spec 头里的 runner sha 也要改。push 要先问用户。
4. 【批准】发布（一次 ConfigMap 写入；脚本先重启 SM，再重启 controller）：
   `cd $T && PYTHONPATH=common:deploy python3 deploy/scripts/release/params_set.py <第 3 步的全部 --set> --apply --restart --reason tss-rerun-20261103 --backup-dir $CAMPAIGN_DATA_ROOT/backups/tss-rerun-20261103`
   - 回滚：用备份目录里发布前的 ConfigMap 做 `kubectl replace`，再跑 `params_set.py --restart`。回滚后 `arm_registry.py id` 应为 `0ab5352747bdd70e`。
5. 看是否已生效：`PYTHONPATH=$T/common:$T/deploy python3 $T/deploy/scripts/release/params_set.py --status`（最后一行应为 `in sync`）。

## B. 填 spec

6. 读出新的 registry id：`ID=$(PYTHONPATH=$T/common:$T/deploy python3 $T/eval/runner/arm_registry.py id); echo $ID`（16 位十六进制，不应是 `0ab5352747bdd70e`）。
7. 填进 rep 1 的 `registry_expect.registry_id`（占位符 `"FILL_REGISTRY_ID"`，引号保留：全是数字的 id 不加引号会被 YAML 读成整数）：`sed -i "s/FILL_REGISTRY_ID/$ID/" $C/campaign-tre-rerun-r3e-20261103.yaml && grep -n 'registry_id:' $C/campaign-tre-rerun-r3e-20261103.yaml`
8. 重新生成 rep 2：`bash $C/make_rep2.sh`

## C. 可选：换 controller 镜像（删 η）

9. 【批准】只有用户同意删 η 时才做（HANDOFF"待用户回答"第 2 项）：
   - 按 `docs/eta-donor-order-usage-20261102.md` 部署（只换 controller 镜像；runner 改动一起合入 main；`kubectl -n tre-v2 set env deploy/tre-v2-controller TRE_DISABLE_ETA_GATE-`）。
   - 之后的每条命令前加 `CTL_IMAGE=tre-v2-controller:<新 tag> MAIN_SHA=<新 main 8 位>`。
   - 改 spec 头里 controller 和 runner 那两行，并注明"HIGH donor 顺序变了，是方法改动"；再跑第 8 步。
   - 不换镜像时跳过本步：默认 `CTL_IMAGE=tre-v2-controller:20261102-420c69ef`，`MAIN_SHA=d17bccea`。

## D. 预检和 dry-run

10. 预检（只读，包括 dry-run）：`bash $C/run_campaign-tre-rerun-20261103.sh --check`，应以 `CHECK OK` 结束。
    检查项：没有别的批次和 marker；main = origin/main = `MAIN_SHA`；占位符已填，spec id = 线上 id；7 个 trace 的 sha256；registry 相对 `0ab5…` 只改了 `trs.{theta_m,tau_crit,tau_high,lambda_wait,w_p}`（`tss_only_guard.py`）；`params_set --status` 为 in sync；controller 镜像 = `CTL_IMAGE`，`TRE_METRICS_WINDOW_MS=10000`；20/20 模型 pod，磁盘余量。
11. rep 2 也预检一次：`NAME=campaign-tre-rerun-r3e-rep2-20261103 bash $C/run_campaign-tre-rerun-20261103.sh --check`
12. 留一份 dry-run 记录：`bash $T/eval/runner/run_campaign.sh $C/campaign-tre-rerun-r3e-20261103.yaml --dry-run > $C/dryrun-launch.txt 2>&1; tail -1 $C/dryrun-launch.txt`（应为 `0 problem(s)`）。

## E. 启动和检查

13. 【批准】启动前通知并行会话，然后二选一：
    - 只跑 rep 1：`cd $C && setsid nohup bash run_campaign-tre-rerun-20261103.sh >> run_campaign-tre-rerun-20261103.out 2>&1 < /dev/null &`
    - rep 1 跑完（state = done）后自动接 rep 2：`cd $C && setsid nohup bash -c 'bash run_campaign-tre-rerun-20261103.sh && NAME=campaign-tre-rerun-r3e-rep2-20261103 bash run_campaign-tre-rerun-20261103.sh' >> run_campaign-tre-rerun-20261103.out 2>&1 < /dev/null &`
    - 驱动脚本自己写 marker（`validation … round=<NAME>`），结束时删掉。
14. 第 1 臂（TRE × Alternating，约 15 min）一结束就查：`python3 $T/eval/runner/first_run_check.py $CAMPAIGN_DATA_ROOT/campaign-tre-rerun-r3e-20261103-run/Alternating_hot_model_periodic_A_s1/tre`；再看 `status.json` 的 `last_validity`（完整性、信号丢失计数、V_req、TTFT p95 是否合理）。没问题后每 50 min 查一次。
15. 停止：`touch $C/STOP`（当前臂跑完后停，也会挡住接着跑的 rep 2）。续跑：删掉 `$C/STOP` 和 `<results_root>/STOP`，再启动一次（跑完的臂会跳过）。runner 被 SIGKILL 时手动跑 `reset_canonical.sh`。
16. 全部跑完：重新生成 `RUNS.md`（`docs/scripts/runs_catalog_20261030/runs_catalog.py`）；离线出图和报告（`docs/scripts/campaign_viz_20261030/viz.sh`）；更新 HANDOFF。
