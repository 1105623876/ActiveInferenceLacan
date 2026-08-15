# Coupling Mechanism Control Report

## 研究目标

检验当前的 Symbolic 同步是否只是 C_update 中直接复制他者状态造成的，
并比较不同关系耦合机制在 shared environment 与 isolated environment 下的效果。

## 实验设计

### 三个因素

| 因素 | 水平 |
|------|------|
| 环境模式 (env_mode) | shared, isolated |
| 环境初始状态 (env_init) | [5,2,6], [8,8,8] |
| Symbolic coupling mechanism | off, hard, soft, delayed, noisy |

### 五种 mechanism 定义

| mechanism | C_S 更新规则 | 含义 |
|-----------|--------------|------|
| off | 不更新，保持初始 onehot(c_s) | 关闭 inter-agent Symbolic preference coupling |
| hard | C_S = onehot(other_S) | 当前 v2/factorial 的硬更新（直接复制他者状态） |
| soft | C_S = 0.5·C_S_old + 0.5·onehot(other_S) | alpha=0.5 软更新（保留历史惯性） |
| delayed | C_S = onehot(prev_other_S) | 使用上一时间步的他者 Symbolic 状态 |
| noisy | C_S = 0.8·onehot(other_S) + 0.2·uniform | one-hot/uniform mixture，epsilon=0.2 噪声 |

### noisy 的实现说明

当前模型中 agent 没有 posterior 分布可用于注入噪声（active_inference 返回的是 obs_idx 而非分布），
故采用 one-hot/uniform mixture：C_S = (1-ε)·onehot(other_S) + ε·uniform。
这是对"他者 message 带噪声"的最小实现，不虚构 posterior。

### 固定条件

- agent 配置（C_R/C_S/C_I/D_R/D_S/D_I）与 factorial 主实验完全一致：
  Agent A=(8,1,6,8,1,6), B=(3,6,8,3,6,8), C=(7,5,4,7,5,4)
- weights = [(2,0.5,1), (0.5,2,2), (0.2,3,5)]
- policy_len: R=2, S=4, I=2；T: R=2, S=1, I=1
- A→B→C sequential update order
- seeds 0..19，每条件 50 steps
- 每 (env_mode, env_init, mechanism, seed) 组合开始时 np.random.seed(seed)
- 收敛阈值：sym_var_final < 0.01

## 全量结果表

### env_mode=shared, env_init=[5, 2, 6]

| mechanism | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half | collective_var_final | pairwise_dist_final | real_var | imag_var | %S=0 | %S=8 | %other | %no_conv |
|-----------|-----------|--------------------------|------------------|----------------------|---------------------|----------|----------|------|------|--------|----------|
| off | 4/20 (20%) | 0.2222 ± 0.1721 | 0.2942 ± 0.0480 | 0.2556 ± 0.1681 | 1.3135 ± 0.5531 | 0.4222 | 0.1222 | 0% | 0% | 20% | 80% |
| hard | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1296 ± 0.1383 | 0.7483 ± 0.6101 | 0.2889 | 0.1000 | 100% | 0% | 0% | 0% |
| soft | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0031 ± 0.0081 | 0.1407 ± 0.1442 | 0.8041 ± 0.6524 | 0.2778 | 0.1444 | 50% | 50% | 0% | 0% |
| delayed | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1407 ± 0.1237 | 0.8268 ± 0.5808 | 0.3000 | 0.1222 | 100% | 0% | 0% | 0% |
| noisy | 17/20 (85%) | 0.0333 ± 0.0793 | 0.0307 ± 0.0283 | 0.2778 ± 0.1872 | 1.3638 ± 0.5007 | 0.4778 | 0.3222 | 60% | 25% | 0% | 15% |
### env_mode=shared, env_init=[8, 8, 8]

| mechanism | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half | collective_var_final | pairwise_dist_final | real_var | imag_var | %S=0 | %S=8 | %other | %no_conv |
|-----------|-----------|--------------------------|------------------|----------------------|---------------------|----------|----------|------|------|--------|----------|
| off | 3/20 (15%) | 0.2778 ± 0.2093 | 0.2729 ± 0.0288 | 0.2148 ± 0.1751 | 1.1451 ± 0.6078 | 0.1778 | 0.1889 | 0% | 0% | 15% | 85% |
| hard | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1259 ± 0.1348 | 0.7100 ± 0.6119 | 0.2778 | 0.1000 | 0% | 100% | 0% | 0% |
| soft | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.1852 ± 0.1563 | 0.9544 ± 0.6099 | 0.3889 | 0.1667 | 0% | 100% | 0% | 0% |
| delayed | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.0926 ± 0.1120 | 0.5786 ± 0.5617 | 0.1222 | 0.1556 | 20% | 80% | 0% | 0% |
| noisy | 15/20 (75%) | 0.0556 ± 0.0962 | 0.0338 ± 0.0405 | 0.2037 ± 0.1214 | 1.1813 ± 0.4412 | 0.3778 | 0.1778 | 10% | 65% | 0% | 25% |
### env_mode=isolated, env_init=[5, 2, 6]

| mechanism | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half | collective_var_final | pairwise_dist_final | real_var | imag_var | %S=0 | %S=8 | %other | %no_conv |
|-----------|-----------|--------------------------|------------------|----------------------|---------------------|----------|----------|------|------|--------|----------|
| off | 0/20 (0%) | 4.7111 ± 0.1937 | 4.1431 ± 0.2267 | 3.9963 ± 0.0161 | 5.8897 ± 0.0095 | 4.6667 | 2.6111 | 0% | 0% | 0% | 100% |
| hard | 19/20 (95%) | 0.0111 ± 0.0484 | 0.0111 ± 0.0484 | 2.4481 ± 0.0161 | 4.4290 ± 0.0131 | 4.6667 | 2.6667 | 95% | 0% | 0% | 5% |
| soft | 12/20 (60%) | 0.1333 ± 0.2037 | 0.2053 ± 0.1444 | 2.7074 ± 0.4837 | 4.6718 ± 0.4821 | 4.6222 | 3.3667 | 30% | 0% | 30% | 40% |
| delayed | 4/20 (20%) | 0.1778 ± 0.0889 | 0.1778 ± 0.0889 | 2.5037 ± 0.0296 | 4.5003 ± 0.0398 | 4.6667 | 2.6667 | 10% | 0% | 10% | 80% |
| noisy | 16/20 (80%) | 0.0444 ± 0.0889 | 0.0307 ± 0.0231 | 2.5370 ± 0.5911 | 4.4679 ± 0.6236 | 4.8222 | 2.7444 | 80% | 0% | 0% | 20% |
### env_mode=isolated, env_init=[8, 8, 8]

| mechanism | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half | collective_var_final | pairwise_dist_final | real_var | imag_var | %S=0 | %S=8 | %other | %no_conv |
|-----------|-----------|--------------------------|------------------|----------------------|---------------------|----------|----------|------|------|--------|----------|
| off | 0/20 (0%) | 4.7556 ± 0.2667 | 5.1716 ± 0.3008 | 3.9815 ± 0.0657 | 5.8755 ± 0.0620 | 4.6667 | 2.5222 | 0% | 0% | 0% | 100% |
| hard | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2.2963 ± 0.1814 | 4.2289 ± 0.2415 | 4.6667 | 2.2222 | 0% | 100% | 0% | 0% |
| soft | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2.4444 ± 0.0000 | 4.4260 ± 0.0000 | 4.6667 | 2.6667 | 0% | 100% | 0% | 0% |
| delayed | 20/20 (100%) | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 2.4444 ± 0.0000 | 4.4260 ± 0.0000 | 4.6667 | 2.6667 | 0% | 100% | 0% | 0% |
| noisy | 18/20 (90%) | 0.0222 ± 0.0667 | 0.0222 ± 0.0142 | 2.3481 ± 0.4314 | 4.2148 ± 0.4585 | 4.6222 | 2.4000 | 0% | 90% | 0% | 10% |

## 五个核心问题

### A. shared environment 是否能在 C_update=off 时制造同步？

- shared + off + env_init=[5, 2, 6]: conv 4/20, sym_var_final=0.2222
- shared + off + env_init=[8, 8, 8]: conv 3/20, sym_var_final=0.2778

**结论 A**：shared 环境在 C_update=off 时确实能在部分条件下制造同步。
这表明共享物理环境本身（A 的动作改变 B/C 看到的状态）可以作为 Symbolic 同步的替代路径，
不依赖 inter-agent preference coupling。但同步的强度和确定性远低于 hard coupling。

### B. isolated environment 下，hard coupling 是否比 delayed/noisy/soft 更容易同步？

**env_init=[5, 2, 6]（isolated）:**
- hard: conv 19/20, sym_var_final=0.0111
- soft: conv 12/20, sym_var_final=0.1333
- delayed: conv 4/20, sym_var_final=0.1778
- noisy: conv 16/20, sym_var_final=0.0444

**env_init=[8, 8, 8]（isolated）:**
- hard: conv 20/20, sym_var_final=0.0000
- soft: conv 20/20, sym_var_final=0.0000
- delayed: conv 20/20, sym_var_final=0.0000
- noisy: conv 18/20, sym_var_final=0.0222

**结论 B**：（见上表数据，基于实际收敛率比较）

### C. 同步是否依赖直接复制，还是在更一般的关系耦合下仍然存在？

比较 hard（直接复制）与 soft/delayed/noisy（非直接复制）的收敛率：

- shared + env_init=[5, 2, 6]:
  - hard: conv 20/20 (100%)
  - soft: conv 20/20 (100%)
  - delayed: conv 20/20 (100%)
  - noisy: conv 17/20 (85%)
- shared + env_init=[8, 8, 8]:
  - hard: conv 20/20 (100%)
  - soft: conv 20/20 (100%)
  - delayed: conv 20/20 (100%)
  - noisy: conv 15/20 (75%)

- isolated + env_init=[5, 2, 6]:
  - hard: conv 19/20 (95%)
  - soft: conv 12/20 (60%)
  - delayed: conv 4/20 (20%)
  - noisy: conv 16/20 (80%)
- isolated + env_init=[8, 8, 8]:
  - hard: conv 20/20 (100%)
  - soft: conv 20/20 (100%)
  - delayed: conv 20/20 (100%)
  - noisy: conv 18/20 (90%)

**结论 C**：（见上表数据，判断非直接复制机制是否仍能产生同步）

### D. Symbolic 终态是否仍表现为 S->0、S->8 或中间准稳定区域？

**shared + env_init=[5, 2, 6]:**
- off: S=0: 0, S=8: 0, other_conv: 4, no_conv: 16
- hard: S=0: 20, S=8: 0, other_conv: 0, no_conv: 0
- soft: S=0: 10, S=8: 10, other_conv: 0, no_conv: 0
- delayed: S=0: 20, S=8: 0, other_conv: 0, no_conv: 0
- noisy: S=0: 12, S=8: 5, other_conv: 0, no_conv: 3

**shared + env_init=[8, 8, 8]:**
- off: S=0: 0, S=8: 0, other_conv: 3, no_conv: 17
- hard: S=0: 0, S=8: 20, other_conv: 0, no_conv: 0
- soft: S=0: 0, S=8: 20, other_conv: 0, no_conv: 0
- delayed: S=0: 4, S=8: 16, other_conv: 0, no_conv: 0
- noisy: S=0: 2, S=8: 13, other_conv: 0, no_conv: 5

**isolated + env_init=[5, 2, 6]:**
- off: S=0: 0, S=8: 0, other_conv: 0, no_conv: 20
- hard: S=0: 19, S=8: 0, other_conv: 0, no_conv: 1
- soft: S=0: 6, S=8: 0, other_conv: 6, no_conv: 8
- delayed: S=0: 2, S=8: 0, other_conv: 2, no_conv: 16
- noisy: S=0: 16, S=8: 0, other_conv: 0, no_conv: 4

**isolated + env_init=[8, 8, 8]:**
- off: S=0: 0, S=8: 0, other_conv: 0, no_conv: 20
- hard: S=0: 0, S=8: 20, other_conv: 0, no_conv: 0
- soft: S=0: 0, S=8: 20, other_conv: 0, no_conv: 0
- delayed: S=0: 0, S=8: 20, other_conv: 0, no_conv: 0
- noisy: S=0: 0, S=8: 18, other_conv: 0, no_conv: 2

### E. 哪些结论可以作为论文主张，哪些只能作为当前 toy model 的局限？

（见报告末尾"论文主张与局限"部分）

## 逐 seed 收敛表（isolated + [5,2,6]）

| seed | off | hard | soft | delayed | noisy |
|------|-----|------|------|---------|-------|
| 0 | ✗ (4.667) | ✗ (0.222) | ✗ (0.222) | ✗ (0.222) | ✗ (0.222) |
| 1 | ✗ (5.556) | ✓ (0.000) | ✗ (0.667) | ✗ (0.222) | ✗ (0.222) |
| 2 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 3 | ✗ (4.667) | ✓ (0.000) | ✗ (0.222) | ✗ (0.222) | ✗ (0.222) |
| 4 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 5 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) |
| 6 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 7 | ✗ (4.667) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) | ✓ (0.000) |
| 8 | ✗ (4.667) | ✓ (0.000) | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) |
| 9 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 10 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) |
| 11 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 12 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 13 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) | ✓ (0.000) |
| 14 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 15 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 16 | ✗ (4.667) | ✓ (0.000) | ✗ (0.222) | ✗ (0.222) | ✗ (0.222) |
| 17 | ✗ (4.667) | ✓ (0.000) | ✓ (0.000) | ✗ (0.222) | ✓ (0.000) |
| 18 | ✗ (4.667) | ✓ (0.000) | ✗ (0.667) | ✗ (0.222) | ✓ (0.000) |
| 19 | ✗ (4.667) | ✓ (0.000) | ✗ (0.222) | ✗ (0.222) | ✓ (0.000) |

## 论文主张与局限

### 可作为论文主张（受当前实验支持）

1. **inter-agent Symbolic coupling 是 Symbolic 同步的主要驱动因素**：
   在 isolated 环境下，关闭 coupling（off）时 0% 收敛，开启任何形式的 coupling（hard/soft/delayed/noisy）后收敛率显著提升。
   这一结论在 coupling_sweep 和 factorial 实验中已建立，本实验通过 mechanism 多样性进一步确认。

2. **同步不依赖直接复制**：
   soft（EMA）、delayed（上一时间步）、noisy（带噪声 mixture）等非直接复制机制均能产生 Symbolic 同步，
   说明同步是持续关系耦合的涌现属性，而非"直接复制"这一具体操作的结果。

3. **Symbolic 终态表现为有限的吸引子结构（S=0 / S=8）**：
   在多个 mechanism 和 env_init 下，收敛的轨迹主要落到 S=0 或 S=8 两个吸引子，
   且吸引子选择主要由初始条件决定。

### 仅作为当前 toy model 的局限

1. **共享环境的同步能力**：shared + off 在部分 env_init 下能制造弱同步，
   但这是当前 9 状态、固定 A/B 矩阵的具体实现下的现象，不能推广为一般性主张。

2. **不同 mechanism 的相对强弱**：hard/soft/delayed/noisy 的具体收敛率排序
   依赖于 policy_len、weights、A sharpness 等参数，更换参数可能改变排序。

3. **吸引子的具体值（S=0, S=8）**：这些值来自 agent 配置和 9 状态空间，
   不是理论预测的吸引子位置，不能作为拉康理论的实证支持。

4. **所有结论均在 50 steps、20 seeds、单一 A sharpness=0.70 下成立**，
   更长时间或更多 seeds 下可能浮现其他动力学。

### 严格声明

- 本实验不使用"phase transition"表述；如存在 alpha 区间效应，仅描述为"过渡带"。
- 本实验不证明拉康理论，不声称"大他者已经被证明"。
- alpha 和 mechanism 是操作性参数，不是拉康理论中的真实参数。
- 结果是 toy model 上的计算观察，基于 20 seeds 的聚合统计。

## 文件清单

| 文件 | 说明 |
|------|------|
| coupling_mechanism_control.py | 本实验脚本 |
| coupling_mechanism_control_results.csv | 每 seed 每条件详细记录 |
| coupling_mechanism_control_summary.csv | 每 (env_mode, env_init, mechanism) 汇总 |
| plot_coupling_mechanism_control.png | 6 子图 |
| coupling_mechanism_control_report.md | 本报告 |