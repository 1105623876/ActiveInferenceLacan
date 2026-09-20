# Factorial 2×2 Experiment Report

## 研究问题

在三主体循环 Symbolic coupling 模型中，Symbolic 层收敛究竟来自：
1. 共享物理环境；
2. agent 间 change_C / Symbolic preference coupling；
3. 二者的组合。

## 实验设计

2×2 factorial，两个因素各两个水平：

| 因素 | 水平 1 | 水平 2 |
|------|--------|--------|
| 环境模式 (env_mode) | shared（三 agent 共享同一组 env） | isolated（每 agent 独立 env 副本） |
| inter-agent Symbolic coupling (c_update) | C_update_on（保持 change_C） | C_update_off（保留初始固定 C_S） |

### 关键设计说明

- **shared 环境是原始模型（agent.py 中 Three 类）有意的建模设定**：三个 agent 共用同一组 env_r/env_s/env_i 实例，A 的动作会改变 B/C 看到的环境状态。v2 修正版将其改为 isolated 以排除共享环境的混淆。本实验同时测试两种模式以分解环境因素的贡献。
- **isolated 环境是替代模型**（v2 修正版采用的设计）：每个 agent 拥有独立的环境实例副本。
- **C_update_off 是 inter-agent Symbolic preference coupling ablation**，不是"完全移除 Symbolic Order"。每个 agent 内部仍保留 R/S/I 三界结构、w_S 权重和 Symbolic active-inference unit；只是不再把 next agent 的 Symbolic observation 设为自己的 C_S，而是保持初始固定 C_S。Symbolic Order 作为 agent 内部结构依然存在。
- w_S（agent 内部 RSI coupling 权重）在四个条件中保持 v2 triadic 配置不变：weights = [(2,0.5,1), (0.5,2,2), (0.2,3,5)]。
- 三个 agent 的 C_R/C_S/C_I/D_R/D_S/D_I 配置、policy_len（R=2, S=4, I=2）、T（R=2, S=1, I=1）、A→B→C sequential update order 均与 v2 run_triadic_isolated 一致。
- shared 和 isolated 条件的环境初始状态统一为 [5, 2, 6]（不沿用 v2 中按 agent d_r/d_s/d_i 创建不同环境初始状态的做法，以避免环境模式与初始状态混淆）。
- 20 seeds (0..19)，50 steps，每 seed 每条件开始前重新初始化所有环境、agent 参数和 observation。
- 收敛阈值：sym_var_final < 0.01（与 v2 Exp4 一致）。

## 四条件汇总表

| 条件 | 收敛率 | sym_var_final (mean±std) | sym_var_2nd_half (mean±std) | collective_var_final (mean±std) | pairwise_dist_final (mean±std) |
|------|--------|--------------------------|----------------------------|--------------------------------|-------------------------------|
| shared + C_update_on   | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1296 ± 0.1383 | 0.7483 ± 0.6101 |
| shared + C_update_off  | 4/20 (20%) | 0.2222 ± 0.1721 | 0.2942 ± 0.0480 | 0.2556 ± 0.1681 | 1.3135 ± 0.5531 |
| isolated + C_update_on  | 19/20 (95%) | 0.0111 ± 0.0484 | 0.0111 ± 0.0484 | 2.4481 ± 0.0161 | 4.4290 ± 0.0131 |
| isolated + C_update_off | 0/20 (0%) | 4.7111 ± 0.1937 | 4.1431 ± 0.2267 | 3.9963 ± 0.0161 | 5.8897 ± 0.0095 |

## 主效应分析

### C_update 主效应（inter-agent Symbolic coupling）

mean(sym_var_final | C_update_on) - mean(sym_var_final | C_update_off)，跨 env_mode：

- C_update_on 均值（跨 env_mode, 40 seeds）: 0.0056
- C_update_off 均值（跨 env_mode, 40 seeds）: 2.4667
- **主效应 = -2.4611**

C_update_on 显著降低了 sym_var_final（负值），说明 inter-agent Symbolic coupling 是 Symbolic 层收敛的主要驱动因素。

### 环境模式主效应

mean(sym_var_final | shared) - mean(sym_var_final | isolated)，跨 c_update：

- shared 均值（跨 c_update, 40 seeds）: 0.1111
- isolated 均值（跨 c_update, 40 seeds）: 2.3611
- **主效应 = -2.2500**

shared 环境显著降低了 sym_var_final，说明共享物理环境对 Symbolic 收敛有贡献。

### 环境模式 × C_update 交互差异

(shared_C_on - shared_C_off) - (isolated_C_on - isolated_C_off)：

- shared 条件下 C_update 效应: -0.2222
- isolated 条件下 C_update 效应: -4.7000
- **交互差异 = 4.4778**

交互差异较大，说明 C_update 的效果依赖于环境模式。

## 20 seeds 逐 seed 收敛表

| seed | shared/C_on | shared/C_off | isolated/C_on | isolated/C_off |
|------|-------------|--------------|---------------|----------------|
| 0 | ✓ (0.000) | ✗ (0.222) | ✗ (0.222) | ✗ (4.667) |
| 1 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (5.556) |
| 2 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 3 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 4 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 5 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 6 | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) | ✗ (4.667) |
| 7 | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) | ✗ (4.667) |
| 8 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 9 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 10 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 11 | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) | ✗ (4.667) |
| 12 | ✓ (0.000) | ✗ (0.667) | ✓ (0.000) | ✗ (4.667) |
| 13 | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) | ✗ (4.667) |
| 14 | ✓ (0.000) | ✗ (0.667) | ✓ (0.000) | ✗ (4.667) |
| 15 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 16 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 17 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 18 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |
| 19 | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✗ (4.667) |

## 与 v2 Exp 4 的差异

v2 Exp 4（run_triadic_isolated, 50 steps）使用的是 **isolated + C_update_on** 条件，但环境初始状态按 agent 各自的 D 创建：
- Agent A env 初始: [8, 1, 6]（来自 d_r=8, d_s=1, d_i=6）
- Agent B env 初始: [3, 6, 8]
- Agent C env 初始: [7, 5, 4]

本实验的 isolated + C_update_on 条件与之区别在于：环境初始状态统一为 [5, 2, 6]（公平性要求：避免环境模式与初始状态混淆）。agent 配置（C/D/weights/obs/policy_len）完全一致。

v2 Exp 4 报告 sym_var 从 1.556 收敛到 0.000。本实验的 isolated + C_update_on 条件 20 seeds 收敛率为 95%，sym_var_final 均值为 0.0111。差异（如有）主要来自环境初始状态的不同（[5,2,6] vs [8,1,6]/[3,6,8]/[7,5,4]）。

## 重要声明

1. **shared 环境是原模型有意的建模设定**（对应原始 agent.py 中 Three 类的共享 env 设计），isolated 环境是替代模型（v2 修正版采用）。本实验不评判哪种设计"正确"，而是分解两种因素各自对 Symbolic 收敛的贡献。
2. **C_update_off 不是"完全移除 Symbolic Order"**。每个 agent 内部仍保留完整的 R/S/I 三界结构、w_S 权重和 Symbolic active-inference unit；只是移除了 agent 之间的 Symbolic preference 耦合（inter-agent coupling ablation）。Symbolic Order 作为 agent 内部结构依然存在。
3. 本实验结果**不构成拉康理论的证明**。这是一个在有限状态空间、固定策略长度、one-hot preference 的 toy model 上的计算观察。Symbolic 收敛的具体数值依赖于 A 矩阵 sharpness、policy_len、weights 等参数配置。
4. 本实验报告基于 20 seeds 的聚合统计，而非单个 seed 或单条代表性轨迹。

## 文件清单

| 文件 | 说明 |
|------|------|
| factorial_env_symbolic.py | 本实验脚本（独立，不修改 deep_experiments_v2.py） |
| factorial_results.csv | 每 seed 每条件的详细记录 |
| factorial_summary.csv | 四条件汇总（跨 20 seeds 聚合） |
| factorial_timeseries.csv | 每 seed 每条件每 step 的时间序列 |
| plot_factorial_2x2.png | 四条件 sym_var 曲线 + 收敛率 + 分布图 |
| factorial_report.md | 本报告 |
