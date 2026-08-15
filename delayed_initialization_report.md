# Delayed Initialization Sensitivity Report

## 研究目的

检验 delayed coupling 在第一步使用 INIT_OBS 兜底，是否影响了 delayed 机制的结果。

## 两种 delayed 初始化

| init_mode | 第一步 C_S | 第二步起 C_S |
|-----------|-----------|--------------|
| current | onehot(INIT_OBS[next_i][1])（与 hard 第一步相同） | onehot(prev_obs[next_i][1]) |
| no_update_first_step | 不更新，保持初始 onehot(c_s) | onehot(prev_obs[next_i][1]) |

`current` 是 coupling_mechanism_control.py 的 delayed 实现；
`no_update_first_step` 是替代方案，避免第一步用 INIT_OBS 兜底可能引入的偏差。

## 实验条件

- env_mode: isolated
- env_init: [[5, 2, 6], [8, 8, 8]]
- seeds: 0..99 (100 seeds)
- 50 steps；收敛阈值 sym_var_final < 0.01
- agent 配置完全沿用 coupling_mechanism_control.py / factorial_env_symbolic.py

## 汇总表

| env_init | init_mode | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half (mean±std) | %S=0 | %S=8 | %other | %no_conv |
|----------|-----------|-----------|--------------------------|----------------------------|------|------|--------|----------|
| [5, 2, 6] | current | 24/100 (24%) | 0.1689 ± 0.0949 | 0.1689 ± 0.0949 | 17% | 0% | 7% | 76% |
| [5, 2, 6] | no_update_first_step | 15/100 (15%) | 0.1889 ± 0.0793 | 0.2111 ± 0.0809 | 5% | 10% | 0% | 85% |
| [8, 8, 8] | current | 100/100 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0% | 100% | 0% | 0% |
| [8, 8, 8] | no_update_first_step | 100/100 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0% | 100% | 0% | 0% |

## 核心对比：两种初始化方式是否影响 delayed 结果？

### env_init=[5, 2, 6]

- **current**: conv 24/100, sym_var_final=0.1689±0.0949, S=0: 17, S=8: 0, other: 7, no_conv: 76
- **no_update_first_step**: conv 15/100, sym_var_final=0.1889±0.0793, S=0: 5, S=8: 10, other: 0, no_conv: 85
- **差异**: conv_rate 24% vs 15% (Δ=+9%), sym_var_final 0.1689 vs 0.1889 (Δ=-0.0200)

### env_init=[8, 8, 8]

- **current**: conv 100/100, sym_var_final=0.0000±0.0000, S=0: 0, S=8: 100, other: 0, no_conv: 0
- **no_update_first_step**: conv 100/100, sym_var_final=0.0000±0.0000, S=0: 0, S=8: 100, other: 0, no_conv: 0
- **差异**: conv_rate 100% vs 100% (Δ=+0%), sym_var_final 0.0000 vs 0.0000 (Δ=+0.0000)

## 判断

两种初始化方式产生了**显著差异**，说明 delayed 机制的结果对第一步处理方式敏感。
coupling_mechanism_control.py 中 delayed 的 4/20（isolated+[5,2,6]）结果可能部分受 INIT_OBS 兜底影响。

## 与 coupling_mechanism_control.py 20-seed 结果对照

coupling_mechanism_control.py 的 delayed（=本实验 current 模式）20 seeds 结果：
- isolated + [5,2,6]: conv 4/20 (20%), sym_var_final=0.1778
- isolated + [8,8,8]: conv 20/20 (100%), sym_var_final=0.0000

本实验 current 模式 100 seeds 结果：
- isolated + [5, 2, 6]: conv 24/100 (24%), sym_var_final=0.1689
- isolated + [8, 8, 8]: conv 100/100 (100%), sym_var_final=0.0000

100 seeds 下的收敛率与 20 seeds 一致或接近，说明 20 seeds 的结果具有代表性。

## 逐 seed 收敛对比（seed 0..19，便于与之前实验对照）

| seed | [5,2,6] current | [5,2,6] no_update | [8,8,8] current | [8,8,8] no_update |
|------|------------------|---------------------|------------------|---------------------|
| 0 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 1 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 2 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 3 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 4 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 5 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 6 | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) |
| 7 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 8 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 9 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 10 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 11 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 12 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 13 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 14 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 15 | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) |
| 16 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 17 | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) |
| 18 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 19 | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |

## 声明

1. 本实验仅验证 delayed 机制的初始化敏感性，不修改已有脚本和结果。
2. 复用 factorial_env_symbolic.py 的模型函数。
3. 结果是 toy model 上的计算观察，基于 100 seeds 的聚合统计。
4. 不使用"证明拉康理论"等表述。
5. S=0/S=8 称为"endpoint regimes"或"终态模式"，不称为严格 attractors（除非有扰动恢复证据，本实验未做扰动恢复测试）。
6. 区分 symbolic alignment（Symbolic 层方差趋零）、endpoint regime convergence（终态落到 S=0/S=8）和 full RSI convergence（R/S/I 三层都收敛）。

## 文件清单

| 文件 | 说明 |
|------|------|
| delayed_initialization_sensitivity.py | 本实验脚本 |
| delayed_initialization_results.csv | 每 seed 每 env_init 每 init_mode 详细记录 |
| delayed_initialization_summary.csv | 4 组汇总 |
| delayed_initialization_report.md | 本报告 |