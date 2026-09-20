# Observer × schedule factorial report

主终点是 convergence。Wilson CI 只作区间描述。
同 seed 比较用 McNemar exact，族内 Holm 校正。
预先检验 schedule × observer interaction。
不把 CI 重叠写成“无差异”；不拒绝 H0 只写“未拒绝”。

## 设计

- observers: hard, smooth, memoryless, static, leaky
- schedules: sync, seq_ABC
- env: [5,2,6] 主分析；[8,8,8] 对照
- n=200 seeds / 格，50 steps，CRN 按 seed/t/agent 预生成
- σ=0.4, leak λ=0.2, smooth ε=0.6227 (匹配 memoryless E[H]=1.8384)
- snapshot 对齐；(NLL/Brier/ACF/振荡) 不消耗机制 RNG
- post-consensus misID 仅在收敛轨迹上报告

## 1. Convergence（Wilson 95% CI）

### 5_2_6（主环境）

| observer | sync | seq_ABC |
|----------|------|---------|
| hard | 39/200 (19.5%) [14.6%,25.5%] | 193/200 (96.5%) [93.0%,98.3%] |
| smooth | 67/200 (33.5%) [27.3%,40.3%] | 68/200 (34.0%) [27.8%,40.8%] |
| memoryless | 44/200 (22.0%) [16.8%,28.2%] | 49/200 (24.5%) [19.1%,30.9%] |
| static | 154/200 (77.0%) [70.7%,82.3%] | 136/200 (68.0%) [61.2%,74.1%] |
| leaky | 113/200 (56.5%) [49.6%,63.2%] | 125/200 (62.5%) [55.6%,68.9%] |

### 8_8_8（容易对照）

| observer | sync | seq_ABC |
|----------|------|---------|
| hard | 200/200 (100.0%) [98.1%,100.0%] | 200/200 (100.0%) [98.1%,100.0%] |
| smooth | 90/200 (45.0%) [38.3%,51.9%] | 87/200 (43.5%) [36.8%,50.4%] |
| memoryless | 50/200 (25.0%) [19.5%,31.4%] | 44/200 (22.0%) [16.8%,28.2%] |
| static | 196/200 (98.0%) [95.0%,99.2%] | 198/200 (99.0%) [96.4%,99.7%] |
| leaky | 125/200 (62.5%) [55.6%,68.9%] | 105/200 (52.5%) [45.6%,59.3%] |

## 2. Interaction（仅 5_2_6）

H0：各 observer 的 schedule 效应（sync−seq）可交换。
统计量 = Var_o(mean_s[conv_sync−conv_seq]) = 0.097334
块置换 p = 0.0002（5000 perms，按 seed 置换 observer 标签）
**预先检验：拒绝可加性，schedule 效应随 observer 变化。随后看 simple effects。**

读结果（描述，不是新检验）：交互几乎全由 hard 贡献——seq 96.5% vs sync 19.5%。smooth / memoryless / leaky 的 schedule McNemar 经 Holm 后均未拒绝；static 的 schedule 检验 p_Holm=0.28。也就是说，日程主要改写“读真值的硬耦合”，对带噪/推断观察者的收敛率没有过校正后的 paired 证据。

## 3. 同 seed McNemar exact + Holm（5_2_6，19 个预先比较）

| 比较 | n11 | n00 | n10 | n01 | p_raw | p_Holm | 结论 |
|------|-----|-----|-----|-----|-------|--------|------|
| schedule | hard (sync vs seq) | 36 | 4 | 3 | 157 | 0.0000 | 0.0000 | 拒绝 H0 |
| schedule | smooth (sync vs seq) | 35 | 100 | 32 | 33 | 1.0000 | 1.0000 | 未拒绝 H0 |
| schedule | memoryless (sync vs seq) | 14 | 121 | 30 | 35 | 0.6201 | 1.0000 | 未拒绝 H0 |
| schedule | static (sync vs seq) | 105 | 15 | 49 | 31 | 0.0567 | 0.2833 | 未拒绝 H0 |
| schedule | leaky (sync vs seq) | 80 | 42 | 33 | 45 | 0.2127 | 0.8507 | 未拒绝 H0 |
| smooth vs hard | sync | 12 | 106 | 55 | 27 | 0.0026 | 0.0211 | 拒绝 H0 |
| memoryless vs hard | sync | 5 | 122 | 39 | 34 | 0.6400 | 1.0000 | 未拒绝 H0 |
| static vs hard | sync | 25 | 32 | 129 | 14 | 0.0000 | 0.0000 | 拒绝 H0 |
| leaky vs hard | sync | 17 | 65 | 96 | 22 | 0.0000 | 0.0000 | 拒绝 H0 |
| smooth vs hard | seq_ABC | 64 | 3 | 4 | 129 | 0.0000 | 0.0000 | 拒绝 H0 |
| memoryless vs hard | seq_ABC | 45 | 3 | 4 | 148 | 0.0000 | 0.0000 | 拒绝 H0 |
| static vs hard | seq_ABC | 130 | 1 | 6 | 63 | 0.0000 | 0.0000 | 拒绝 H0 |
| leaky vs hard | seq_ABC | 119 | 1 | 6 | 74 | 0.0000 | 0.0000 | 拒绝 H0 |
| memoryless vs smooth | sync | 24 | 113 | 20 | 43 | 0.0052 | 0.0361 | 拒绝 H0 |
| static vs smooth | sync | 54 | 33 | 100 | 13 | 0.0000 | 0.0000 | 拒绝 H0 |
| leaky vs smooth | sync | 46 | 66 | 67 | 21 | 0.0000 | 0.0000 | 拒绝 H0 |
| memoryless vs smooth | seq_ABC | 21 | 104 | 28 | 47 | 0.0370 | 0.2217 | 未拒绝 H0 |
| static vs smooth | seq_ABC | 42 | 38 | 94 | 26 | 0.0000 | 0.0000 | 拒绝 H0 |
| leaky vs smooth | seq_ABC | 49 | 56 | 76 | 19 | 0.0000 | 0.0000 | 拒绝 H0 |

n10 = 左列成功且右列失败；n01 相反。未拒绝 ≠ 证明等价。

## 4. 次要终点（全轨迹，不筛收敛）

### 5_2_6

| observer | sched | NLL½ | Brier½ | switch | acf1 | mis_acf1 | turn | p2 | spec_share |
|----------|-------|------|--------|--------|------|----------|------|----|------------|
| hard | sync | 0.000 | 0.000 | 0.619 | 0.370 | nan | 0.333 | 0.319 | 0.321 |
| hard | seq_ABC | 0.000 | 0.000 | 0.187 | 0.620 | nan | 0.123 | 0.109 | 0.228 |
| smooth | sync | 0.806 | 0.345 | 0.470 | 0.697 | nan | 0.173 | 0.166 | 0.385 |
| smooth | seq_ABC | 0.806 | 0.345 | 0.464 | 0.686 | nan | 0.172 | 0.165 | 0.360 |
| memoryless | sync | 1.711 | 0.743 | 0.559 | 0.622 | -0.015 | 0.229 | 0.222 | 0.322 |
| memoryless | seq_ABC | 1.700 | 0.738 | 0.563 | 0.644 | -0.012 | 0.229 | 0.221 | 0.337 |
| static | sync | 1.555 | 0.666 | 0.311 | 0.670 | 0.334 | 0.132 | 0.125 | 0.281 |
| static | seq_ABC | 1.889 | 0.691 | 0.365 | 0.673 | 0.290 | 0.157 | 0.148 | 0.308 |
| leaky | sync | 1.046 | 0.517 | 0.372 | 0.661 | 0.268 | 0.142 | 0.136 | 0.298 |
| leaky | seq_ABC | 1.024 | 0.503 | 0.377 | 0.694 | 0.200 | 0.140 | 0.132 | 0.324 |

## 5. 条件性次要终点：post-consensus misID（仅收敛）

| env | observer | sched | n_conv | post-misID mean |
|-----|----------|-------|--------|-----------------|
| 5_2_6 | hard | sync | 39/200 | 0.000 |
| 5_2_6 | hard | seq_ABC | 193/200 | 0.000 |
| 5_2_6 | smooth | sync | 67/200 | 0.000 |
| 5_2_6 | smooth | seq_ABC | 68/200 | 0.000 |
| 5_2_6 | memoryless | sync | 44/200 | 0.560 |
| 5_2_6 | memoryless | seq_ABC | 49/200 | 0.611 |
| 5_2_6 | static | sync | 154/200 | 0.343 |
| 5_2_6 | static | seq_ABC | 136/200 | 0.338 |
| 5_2_6 | leaky | sync | 113/200 | 0.230 |
| 5_2_6 | leaky | seq_ABC | 125/200 | 0.240 |
| 8_8_8 | hard | sync | 200/200 | 0.000 |
| 8_8_8 | hard | seq_ABC | 200/200 | 0.000 |
| 8_8_8 | smooth | sync | 90/200 | 0.000 |
| 8_8_8 | smooth | seq_ABC | 87/200 | 0.000 |
| 8_8_8 | memoryless | sync | 50/200 | 0.560 |
| 8_8_8 | memoryless | seq_ABC | 44/200 | 0.522 |
| 8_8_8 | static | sync | 196/200 | 0.168 |
| 8_8_8 | static | seq_ABC | 198/200 | 0.162 |
| 8_8_8 | leaky | sync | 125/200 | 0.240 |
| 8_8_8 | leaky | seq_ABC | 105/200 | 0.236 |

此终点以收敛为条件，不作主推断。

## 6. 不能写的话

- 不用 Wilson / bootstrap CI 重叠宣布 observer 或 schedule“无差异”。
- 未拒绝 Holm 校正后的 H0，只写未拒绝。
- 不把 leaky/static 写成欲望或大他者。
- 不写 phase transition。

## 文件

| 文件 | 说明 |
|------|------|
| opaque_other_observer_schedule.py | 本脚本 |
| opaque_other_observer_schedule_results.csv | 每 seed 每格 |
| opaque_other_observer_schedule_summary.csv | 格子汇总 |
| opaque_other_observer_schedule_report.md | 本报告 |