# Paper Claims Matrix v2

> 基于 paper_statistics.md 的统计检验结果修正。v1 中基于 ad-hoc 阈值的判断已删除。
> 数据来源：coupling_mechanism_control (n=20/条件), delayed_initialization (n=100/条件, McNemar 配对检验), opaque_other_coupling (n=20/条件)
>
> v3 增补：A7、A8、B7、B8、C7；术语补 inferred-other / misID / post-sync misID。

## 修正摘要（v1 -> v2）

| 项目 | v1 | v2 | 修正原因 |
|------|----|----|----------|
| 主要驱动因素 | "是 Symbolic 同步的主要驱动因素" | "在当前模型中是主要驱动因素" | 限定 scope |
| 不依赖直接复制 | "不依赖直接复制机制" | "hard copying 不是必要条件" | 表述更精确 |
| noisy | "noisy mechanism" | "uncertain-message / precision-attenuated message" | 术语准确 |
| S=0/S=8 | "attractor" | "endpoint regime" | 无扰动恢复证据 |
| shared+off | "弱替代路径" | "weak alternative alignment pathway" | 术语统一 |
| A4 | "5 env_init × 11 alpha factorial" | "5 env_init hard coupling + 2 env_init alpha sweep" | 不是完整 factorial |
| delayed init | "显著差异 (ad-hoc 5%)" | "方向性影响但不显著 (McNemar p=0.1628)" | 统计检验 |
| ad-hoc 阈值 | "差异 >5% 即显著" | 已删除 | 改用统计检验 |
| inferred-other | （无） | A7 滤波信念足以产生共识；A8 共识后可仍误认 | opaque_other Phase 1 |

## 术语约定（全文适用）

| 术语 | 定义 |
|------|------|
| symbolic alignment | 三 agent 的 Symbolic observation 方差趋零（sym_var → 0） |
| endpoint regime | 收敛轨迹的终态模式，当前观察到 S=0 和 S=8 两种 |
| endpoint regime convergence | 终态落到 S=0 或 S=8 且 sym_var < 0.01 |
| full RSI convergence | Real/Symbolic/Imaginary 三层方差都趋零（当前从未发生） |
| inter-agent Symbolic coupling | agent i 把 agent (i+1)%3 的 Symbolic 状态纳入自己的 C_S |
| uncertain-message mechanism | C_S = 0.8·onehot(other_S) + 0.2·uniform（他者 message 带不确定性/精度衰减） |
| precision-attenuated message | 同上，强调精度衰减语义 |
| weak alternative alignment pathway | shared+off 的偶发弱同步（15%–20%），由共享物理环境偶发产生 |
| inferred-other coupling | 从带噪社会观察滤波得到 q(s_j)，令 C_S := q；推断的是位置不是欲望 |
| misID | argmax(belief) ≠ 对方真实 Symbolic 状态 |
| post-sync misID | 进入并保持 sym_var < 0.01 之后的 misID 均值 |

## A. 可以作为论文主张

### A1. 在当前模型中，inter-agent Symbolic coupling 是 Symbolic 同步的主要驱动因素

**claim**: 在当前 triadic active-inference toy model 配置下，inter-agent Symbolic preference coupling 是 Symbolic 层集体收敛的主要驱动因素。
**evidence**: 
- coupling_mechanism_control (n=20)：isolated+off 0% [0%,0%] vs isolated+hard 95% [85%,100%]（[5,2,6]）/ 100% [100%,100%]（[8,8,8]），CI 不重叠
- coupling_mechanism_control (n=20)：isolated+off vs 任何机制 CI 均不重叠（off=0% vs soft 60% [40%,80%], delayed 20% [5%,40%], uncertain-message 80% [60%,95%]）
**scope**: 当前模型配置（9 状态空间，A sharpness=0.70，固定 A/B/weights，A→B→C sequential update）。
**limitation**: "主要驱动因素"基于 off (0%) vs coupling (60%–100%) 的量级差异和 CI 不重叠。不排除在其他模型配置下共享环境或其他机制可能成为主因。
**recommended wording**: "In the current triadic active-inference model, inter-agent Symbolic preference coupling is the primary driver of Symbolic-layer collective convergence: with coupling off, isolated-environment convergence is 0% (95% CI [0%, 0%], n=20); with coupling on (hard copying), convergence rises to 95%–100% (95% CI [85%, 100%], n=20), with non-overlapping CIs."

### A2. Hard copying 不是 Symbolic 同步的必要条件

**claim**: Symbolic 同步不依赖于"直接复制他者状态"这一具体操作。soft EMA、delayed、uncertain-message 等非直接复制机制均能产生 Symbolic 同步。
**evidence**: 
- coupling_mechanism_control (n=20)：在 isolated+[8,8,8] 下，soft 100% [100%,100%], delayed 100% [100%,100%], uncertain-message 90% [75%,100%]（均 CI 不与 off 0% [0%,0%] 重叠）
- coupling_mechanism_control (n=20)：在 isolated+[5,2,6] 下，uncertain-message 80% [60%,95%], soft 60% [40%,80%], delayed 20% [5%,40%]（均 CI 不与 off 0% [0%,0%] 重叠）
**scope**: 当前 5 种机制中的 3 种非直接复制机制。机制间相对强弱在 n=20 下无法区分（CI 重叠）。
**limitation**: "不是必要条件"基于定性结论（非 off 机制均显著优于 off），不基于机制间精细排序。hard vs uncertain-message (95% [85%,100%] vs 80% [60%,95%]) CI 重叠，无法在 n=20 下区分。
**recommended wording**: "Hard copying is not necessary for Symbolic convergence: soft EMA (α=0.5), delayed, and uncertain-message (precision-attenuated) mechanisms all produce convergence significantly above the off baseline (non-overlapping CIs), though their relative rankings cannot be distinguished at n=20."

### A3. 共享物理环境是 Symbolic 同步的 weak alternative alignment pathway

**claim**: 共享物理环境（shared environment）在没有 inter-agent coupling 的情况下可以产生偶发的 Symbolic alignment（15%–20% 收敛率），构成 weak alternative alignment pathway，但效率远低于 inter-agent coupling。
**evidence**: 
- coupling_mechanism_control (n=20)：shared+off [5,2,6] 20% [5%,40%], [8,8,8] 15% [0%,30%]；vs isolated+off 0% [0%,0%]
- shared+off vs shared+hard/soft/delayed (均 100%) CI 不重叠
**scope**: 当前 9 状态、固定 A/B 矩阵配置。
**limitation**: 15%–20% 的弱同步依赖 env_s 初始状态和 A/B 矩阵结构。"weak"的判断基于与 coupling 机制（80%+）的量级对比。shared+off 的 CI [5%,40%] / [0%,30%] 较宽，下界包含 0%。
**recommended wording**: "A shared physical environment constitutes a weak alternative alignment pathway: without inter-agent coupling, shared+off produces 15%–20% convergence (95% CI [0%, 40%], n=20), above the isolated+off baseline (0%) but far below any coupling mechanism (80%+)."

### A4. Symbolic 同步对环境初始状态高度鲁棒（在 hard coupling 下）

**claim**: 在 inter-agent Symbolic coupling 存在（尤其 hard coupling）的条件下，Symbolic 同步对环境初始状态高度鲁棒。env_init 主要影响收敛到哪个 endpoint regime，而非是否收敛。
**evidence**: 
- env_init_sensitivity (n=20)：5 组 env_init 在 isolated+hard 下均 95%–100% 收敛
- coupling_sweep (n=20)：2 组 env_init（[5,2,6] 和 [8,8,8]）在 alpha≥0.9 下均 90%+ 收敛；[8,8,8] 在所有 alpha>0 下 100%
- 注意：这不是完整 5×11 factorial。env_init_sensitivity 只测 hard coupling（5 env_inits），coupling_sweep 只测 2 env_inits × 11 alphas。
**scope**: hard coupling 下 5 个 env_init + alpha sweep 下 2 个 env_init。中等 alpha (0.3–0.8) 下 [5,2,6] 收敛率降至 45%–70%，鲁棒性有例外。
**limitation**: "高度鲁棒"限定于 hard coupling 或高 alpha。中等 coupling strength + 远离 endpoint regime 的初始条件下，鲁棒性下降。5 env_inits × 11 alphas 的完整 factorial 未做。
**recommended wording**: "Under hard coupling, Symbolic convergence is robust across 5 env_init configurations (95%–100%, n=20 each). The α-sweep (11 levels) was conducted for only 2 env_inits, so the interaction between coupling strength and env_init is characterized for 2 cases, not as a full 5×11 factorial."

### A5. 存在有限的 endpoint regime 结构（S=0 / S=8）

**claim**: 在当前模型配置下，Symbolic 同步的终态表现为有限的 endpoint regime，主要观察到 S=0 和 S=8 两种。endpoint regime 的选择主要由初始条件决定。
**evidence**: 
- env_init_sensitivity (n=20)：收敛轨迹主要落 S=0 或 S=8（5 env_inits）
- coupling_sweep (n=20)：[8,8,8] 下所有 alpha>0 落 S=8；[5,2,6] 下 alpha=1.0 落 S=0
- coupling_mechanism_control (n=20)：跨 5 机制 × 2 env_inits × 2 env_modes，收敛轨迹主要落 S=0 或 S=8
**scope**: 当前 9 状态空间和 agent 配置。
**limitation**: **不称为严格 attractor**--未做扰动恢复测试（perturbation recovery），无法证明 S=0/S=8 是动力学吸引子而非仅收敛终点。S=0/S=8 的具体值依赖状态空间大小和 agent 配置。isolated+[5,2,6]+soft 出现 30% other_conv（收敛但终态不在 S=0/S=8）。
**recommended wording**: "Symbolic convergence terminates in a finite set of endpoint regimes (primarily S=0 and S=8 in the current 9-state configuration). We call these endpoint regimes rather than attractors, as no perturbation-recovery test was conducted to verify dynamical attractor status."

### A6. Real 和 Imaginary 层不随 Symbolic coupling 变化（full RSI convergence 从未发生）

**claim**: 在当前模型中，inter-agent Symbolic coupling 只作用于 Symbolic 层，不影响 Real 和 Imaginary 层的集体方差。full RSI convergence 从未发生。
**evidence**: 
- 所有实验：real_var_final 稳定在 ~4.67，imag_var_final 在 ~2.2–3.5，不受 env_mode/env_init/mechanism/alpha 影响
- coupling_mechanism_control (n=20)：5 种机制下 real_var/imag_var 几乎不变
**scope**: 当前模型结构（coupling 只通过 C_S）的必然结果。
**limitation**: 是模型结构（coupling 只通过 C_S）的直接后果，不是独立经验发现。把它解读为"Symbolic 是共享层，R/I 是私有层"的理论主张超出模型支持范围（见 C5）。
**recommended wording**: "In the current model, inter-agent coupling acts only on the Symbolic layer; Real and Imaginary variances remain constant (~4.67 and ~2.2–3.5) across all conditions. Full RSI convergence (all three layers converging) never occurs. This is a structural consequence of coupling via C_S only, not an independent empirical finding."

### A7. 滤波信念足以在独立环境中产生 Symbolic 共识

**claim**: 当他者 Symbolic 位置只能通过带噪社会观察被推断时，`C_S := q(s_j)` 仍能产生 Symbolic-layer 集体收敛。
**evidence**:
- opaque_other (n=20, isolated)：off 0% [0%,0%] vs infer@1.00 100% [100%,100%]（两个初态）；vs infer@0.40 75% [55%,90%]（[5,2,6]）/ 100% [100%,100%]（[8,8,8]），与 off 的 CI 不重叠
- infer@1.00 的 misID 为 0.026 / 0.020，接近 hard 的 0，说明 σ=1 通道塌回可读真值
**scope**: isolated；推断对象是位置 `s_j`；A_social = make_A_matrix(σ)；D_other 均匀初始化；无他者运动模型。
**limitation**: 未做 shared 环境。收敛率与 noisy 的 CI 重叠，不能说 infer 比涂糊真值“更会同步”。
**recommended wording**: "When the other's Symbolic state is available only through a noisy social channel, preference coupling via a filtered belief q(s_j) still produces Symbolic consensus in isolated environments (75%–100% vs 0% off; non-overlapping CIs, n=20)."

### A8. 高不透明度下，共识之后仍可误认他者位置

**claim**: 在高不透明度（σ=0.40）下，Symbolic 对齐之后，argmax q 仍可系统性地不等于对方真值。共同符号坐标不要求读准他者。
**evidence**:
- opaque_other (n=20, isolated)：infer@0.40 共识后 misID = 0.395（[5,2,6]，15/20 收敛）和 0.254（[8,8,8]，20/20 收敛）
- 同条件后半段 misID 为 0.489 / 0.263，与共识后同数量级，不是“到站即清零”
- hard / noisy / infer@1.00 的共识后 misID ≈ 0
**scope**: σ=0.40 的 inferred-other；post-sync 定义为进入并保持 sym_var < 0.01 之后。
**limitation**: 描述统计，未对 post-sync misID 做相对 0 的正式检验（与 hard/noisy 的 0 对照是结构性的）。不是错误信念吸引子的扰动恢复证明。σ=0.70 在 [8,8,8] 上共识后 misID 已降到 0.034，该主张限定于较高不透明度。
**recommended wording**: "Under high opacity (σ=0.40), agents can share a Symbolic coordinate while still misidentifying one another's position after consensus (post-sync misID 0.25–0.40). Consensus therefore does not require an accurate representation of the other."

## B. 只能作为 toy model 内的观察

### B1. S=0 主导 [5,2,6]，S=8 主导 [8,8,8]

**claim**: endpoint regime 选择与 env_init 相关：[5,2,6] 倾向 S=0，[8,8,8] 倾向 S=8。
**evidence**: env_init_sensitivity, coupling_sweep, mechanism_control 一致显示。
**scope**: 当前 9 状态空间。
**limitation**: S=0/S=8 的具体值完全依赖状态空间和 agent 配置。不能推广为"Symbolic 同步总是收敛到状态空间端点"。
**recommended wording**: "Within the 9-state configuration, [5,2,6] tends toward S=0 and [8,8,8] toward S=8. This mapping is configuration-specific and not generalizable."

### B2. [8,8,8] 下 alpha=0.1 即达 100% 收敛（sharp 过渡）

**claim**: [8,8,8] 下 coupling strength 从 0 到 0.1 表现为 sharp 过渡。
**evidence**: coupling_sweep (n=20)：alpha=0 时 0%, alpha=0.1 时 100%。
**scope**: [8,8,8] 单一 env_init。
**limitation**: 阈值位置依赖 A sharpness、policy_len、weights。[5,2,6] 下过渡是渐变的（alpha 0.1–0.8 收敛率 45%–70%）。20 seeds 不足以做严格相变分析。**不使用 phase transition 表述**。
**recommended wording**: "For env_init=[8,8,8], convergence rises sharply between α=0 (0%) and α=0.1 (100%). For [5,2,6], the transition is gradual. We do not characterize this as a phase transition given n=20 and the env_init-dependent transition shape."

### B3. delayed 在 isolated+[5,2,6] 下弱于其他机制（方向性，未达统计显著）

**claim**: delayed 机制在 isolated+[5,2,6] 下收敛率方向上低于 hard/uncertain-message/soft，但机制间 CI 重叠。
**evidence**: coupling_mechanism_control (n=20)：delayed 20% [5%,40%] vs hard 95% [85%,100%]（CI 不重叠）vs uncertain-message 80% [60%,95%]（CI 重叠）vs soft 60% [40%,80%]（CI 重叠）。
**scope**: isolated+[5,2,6] 单一条件。[8,8,8] 下所有机制 100%，无差异。
**limitation**: delayed vs hard CI 不重叠可判显著，但 delayed vs uncertain-message/soft CI 重叠，无法在 n=20 下区分。delayed 的弱表现可能部分受第一步初始化影响（见 B6）。
**recommended wording**: "Under isolated+[5,2,6], delayed coupling (20%, 95% CI [5%, 40%]) is significantly lower than hard coupling (95%, 95% CI [85%, 100%], non-overlapping CIs), but cannot be distinguished from uncertain-message (80%) or soft (60%) at n=20."

### B4. uncertain-message 与 soft 的相对强弱无法在 n=20 下区分

**claim**: isolated+[5,2,6] 下 uncertain-message (80%) 方向上高于 soft (60%)，但 CI 重叠。
**evidence**: coupling_mechanism_control (n=20)：uncertain-message 80% [60%,95%] vs soft 60% [40%,80%]，CI 重叠。
**scope**: isolated+[5,2,6] 单一条件。
**limitation**: 反直觉排序（带不确定性比保留历史惯性更利于同步）依赖当前参数。CI 重叠无法确认排序。
**recommended wording**: "The directional ranking uncertain-message (80%) > soft (60%) under isolated+[5,2,6] is not statistically significant (overlapping CIs at n=20)."

### B5. shared+off 的 weak alternative alignment pathway（15%–20%）

**claim**: shared+off 产生 15%–20% 的偶发 Symbolic alignment。
**evidence**: coupling_mechanism_control (n=20)：shared+off [5,2,6] 20% [5%,40%], [8,8,8] 15% [0%,30%]。
**scope**: 当前 A/B 矩阵结构。
**limitation**: 偶发同步依赖 env_s 初始状态和 A/B 矩阵。CI 下界包含 0%。不能推广为"共享环境总能产生弱同步"。
**recommended wording**: "Shared+off produces a weak alternative alignment pathway (15%–20%, 95% CI [0%, 40%]), configuration-dependent and with CIs including 0%."

### B6. delayed 初始化方式有方向性影响但未达统计显著

**claim**: delayed 的 current (24%) vs no_update_first_step (15%) 有方向性差异（current 略高），但 McNemar 检验不显著。
**evidence**: delayed_initialization (n=100, paired)：McNemar exact p=0.1628（n10=21, n01=12, discordant=33）；paired mean diff = -0.0200, bootstrap 95% CI [-0.0444, 0.0044] 包含 0。
**scope**: isolated+[5,2,6] 单一条件。[8,8,8] 下两者均 100%，无差异。
**limitation**: **v1 的"显著差异"判断（基于 ad-hoc 5% 阈值）不成立**。方向上 current 略高于 no_update（第一步 hard 式耦合的额外推力），但统计上无法拒绝 H0。n=100 仍可能功效不足。
**recommended wording**: "Delayed initialization mode (current 24% vs no_update_first_step 15%) shows a directional but non-significant difference (McNemar exact p=0.1628, paired bootstrap CI [-0.044, 0.004] includes 0, n=100). The v1 ad-hoc 5% threshold judgment is retracted."

### B7. infer@1 与 hard 的 endpoint 混合不同

**claim**: σ=1 时收敛率接近 hard，但 [5,2,6] 上 infer@1 有 60% other_conv，hard 有 95% S=0。
**evidence**: opaque_other (n=20)。
**scope**: isolated+[5,2,6]。[8,8,8] 上两者都是 100% S=8。
**limitation**: infer 的社会通道采样会额外消耗 RNG，终态差异可能部分来自随机流错位，不能单独解释为机制效应。
**recommended wording**: "At σ=1, convergence rates match hard copying, but the [5,2,6] endpoint mix differs (60% other_conv vs 95% S=0). We do not treat this as an established mechanism effect."

### B8. infer 与 noisy 的差别主要是误认同，不是收敛率

**claim**: 两边收敛 CI 重叠；noisy 的 misID=0 是“先读真值再涂糊”的结构事实。
**evidence**: [5,2,6] noisy 80% [60%,95%] vs infer@0.40 75% [55%,90%]；H(C) 0.84 vs 0.73。
**scope**: 当前 ε=0.2 的 noisy 定义，未做熵匹配的 Phase 2。
**limitation**: 不能写“infer 比 noisy 更会同步”或“infer 只是另一种噪声”。
**recommended wording**: "Inferred-other and mixture-noise coupling are not distinguishable by convergence rate at n=20. They differ in whether the preference mode can be wrong: mixture noise reads the true state first, so argmax is always correct."

## C. 当前证据不足，论文中不应提出

### C1. "S=0/S=8 是动力学吸引子"

**claim**: (不应提出) S=0/S=8 是严格 attractor。
**evidence**: 无扰动恢复测试。
**scope**: 不适用。
**limitation**: 未做扰动恢复实验。当前只观察到"收敛到 S=0/S=8"，无法区分吸引子 vs 收敛终点。
**recommended wording**: 使用"endpoint regime"，不使用"attractor"。如需主张 attractor status，需补扰动恢复实验。

### C2. "存在 phase transition"

**claim**: (不应提出) coupling strength 存在 phase transition。
**evidence**: [8,8,8] 的 sharp 过渡看似支持，但 [5,2,6] 是渐变。
**scope**: 不适用。
**limitation**: 20 seeds 统计分辨率不足，两个 env_init 形态不同。需更多 seeds、更细 alpha 扫描、Binder cumulant 等统计检验。
**recommended wording**: 不使用"phase transition"。可描述为"sharp transition"（[8,8,8]）或"gradual transition"（[5,2,6]），并明确 env_init 依赖。

### C3. "Symbolic 同步对应拉康的大他者涌现"

**claim**: (不应提出) 计算现象证明大他者。
**evidence**: 无理论映射论证。
**scope**: 不适用。
**limitation**: "大他者"是拉康理论概念，不能由 sym_var→0 直接推导。这是理论解读，非实验主张。
**recommended wording**: 不提出。如需讨论，明确为哲学解读而非实验结论，并独立论证映射关系。

### C4. "inter-agent coupling 是充分条件"

**claim**: (不应提出) coupling 是 Symbolic 同步的充分条件。
**evidence**: 部分条件下非 100%（delayed 15%–24%, soft 60%）。
**scope**: 不适用。
**limitation**: "充分条件"过强。
**recommended wording**: 使用"主要驱动因素"或"显著促进"，不使用"充分条件"。

### C5. "Real/Imaginary 差异对应主体间性结构"

**claim**: (不应提出) R/I 方差不变对应拉康主体间性。
**evidence**: 模型结构直接后果，非独立发现。
**scope**: 不适用。
**limitation**: 是 coupling 只通过 C_S 的结构后果，非经验发现。
**recommended wording**: 可作为模型设计动机，不作为实验结论。

### C6. "coupling strength 过渡带对应临床现象"

**claim**: (不应提出) alpha 过渡带对应临床边缘状态。
**evidence**: 无临床映射。
**scope**: 不适用。
**limitation**: toy model 参数扫描与临床现象无已建立映射。
**recommended wording**: 不提出。需独立临床研究建立映射。

### C7. "inferred-other 实现了大他者 / Che vuoi?"

**claim**: (不应提出) 滤波信念等于大他者或对他者欲望的形式化。
**evidence**: 推断对象是 s_j，不是 C_j；状态没有语义。
**scope**: 不适用。
**limitation**: 位置不透明 ≠ 欲望之谜。
**recommended wording**: 不提出。可写“他者位置不必可读”，不可写“我们实现了大他者的欲望”。

## 统计支撑总结

| 主张 | 支持检验 | 样本量 | 结果 |
|------|----------|--------|------|
| A1 coupling 是主因 | bootstrap CI 重叠检验 | n=20/条件 | off (0% [0%,0%]) vs hard (95% [85%,100%]) CI 不重叠 ✓ |
| A2 hard copying 非必要 | bootstrap CI 重叠检验 | n=20/条件 | off vs soft/delayed/uncertain-message CI 不重叠 ✓；机制间 CI 重叠（无法排序） |
| A3 weak alternative pathway | bootstrap CI 重叠检验 | n=20/条件 | shared+off 15-20% vs isolated+off 0% |
| A4 env_init 鲁棒性 | 描述统计 | n=20/env_init | 5 env_inits (hard) + 2 env_inits (alpha sweep)，非完整 factorial |
| A5 endpoint regime | 描述统计 | n=20/条件 | 跨条件一致，无扰动恢复 |
| A6 R/I 不变 | 描述统计 | n=20–100 | 所有条件一致 |
| B6 delayed init | McNemar exact + paired bootstrap | n=100 (paired) | p=0.1628 不显著，CI 包含 0 |
| A7 inferred-other 共识 | bootstrap CI vs off | n=20/条件 | off 0% vs infer 75%–100%，CI 不重叠 ✓ |
| A8 共识后误认 | 描述统计 | n=20/条件 | σ=0.40 共识后 misID 0.25–0.40；hard/noisy/infer@1 ≈ 0 |

## 实验覆盖度

| 实验 | 脚本 | seeds | 核心产出 |
|------|------|-------|----------|
| factorial 2×2 | `src/factorial_env_symbolic.py` | 20 | A1, A3 |
| env_init sensitivity | `src/env_init_sensitivity.py` | 20 | A4 (5 env_inits, hard), A5 |
| coupling strength sweep | `src/symbolic_coupling_sweep.py` | 20 | A1, A2, A4 (2 env_inits, alpha sweep), B2 |
| coupling mechanism control | `src/coupling_mechanism_control.py` | 20 | A1, A2, A3, B3, B4, B5 |
| delayed init sensitivity | `src/delayed_initialization_sensitivity.py` | 100 | B6 (McNemar 配对检验) |
| opaque other coupling | `src/opaque_other_coupling.py` | 20 | A7, A8, B7, B8 |

## 建议的论文表述模板

**可以写**：
> "In the current triadic active-inference toy model, inter-agent Symbolic preference coupling is the primary driver of Symbolic-layer collective convergence (A1). Hard copying is not necessary (A2). A shared physical environment is only a weak alternative pathway (A3). When the other is available only as a noisy observation, a filtered belief q(s_j) still produces consensus (A7). Under high opacity, agents can remain systematically wrong about one another's position after they have already aligned (A8). Convergence terminates in a finite set of endpoint regimes (A5); full RSI convergence never occurs (A6)."

**不应写**：
- "We prove the emergence of the Lacanian big Other" (C3)
- "A phase transition exists in coupling strength" (C2)
- "S=0 and S=8 are dynamical attractors" (C1)
- "Inter-agent coupling is a sufficient condition" (C4)
- "Delayed initialization significantly affects convergence" (B6, McNemar p=0.1628)
- "Inferred-other implements the desire of the Other / Che vuoi?" (C7)

## 文件清单

| 文件 | 说明 |
|------|------|
| `docs/analysis/paper_statistics.md` | 统计检验详细结果（McNemar + bootstrap CI） |
| `docs/analysis/paper_claims_matrix_v2.md` | 本文件，主张分类 v2 |
| `outputs/manuscript/paper_tables.csv` | 所有条件的统计汇总表（26 行，32 列） |
