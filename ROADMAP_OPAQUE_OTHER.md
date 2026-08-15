# Roadmap: Opaque Other Coupling

> 路径 B。把 `change_C` 从“复制他者 Symbolic 观察”改成“根据带噪社会观察推断他者位置，再用该信念塑造自己的 preference”。
>
> 本文是研究计划，不是实验结果。现有脚本一律不改。

---

## 1. 一句话

当前模型里，主体直接读取对方的整数状态并写入 `C_S`。下一步只改这一件事：他者不再可读，只可推断。

---

## 2. 为什么这是下一刀，而不是又一次扫描

已完成的机制对照（off / hard / soft / delayed / noisy）都还在**读取真值**：

| 机制 | 实际在做什么 |
|---|---|
| hard | `C = onehot(true S_j)` |
| soft | 对真值 one-hot 做 EMA |
| delayed | 复制上一拍真值 |
| noisy | `0.8·onehot(true S_j) + 0.2·uniform` |

`noisy` 看起来像不确定性，其实只是把已经读到的真值涂糊。主体没有关于他者的隐藏状态，也没有跨时间的信念。

另外，生成过程里 `Agent.step` 返回的就是真实状态，`A` 只存在于主体自己的生成模型中。所以“观察噪声”目前对主体间通道不成立。

要做的不是再加一种 smearing，而是补上被跳过的那一层：

```text
S_j  --(社会通道噪声)-->  o_{i←j}  --(滤波推断)-->  q_i(s_j)  --(写入)-->  C_i
```

这同时更接近 Friston & Frith 的广义同步，也更接近拉康式命题：你不知道他者要什么，只能从失败的信号里构造一个位置。

---

## 3. 锁定的设计（会改变实现方向的假设）

这些默认已经选定。改其中任何一条，等于换实验，不要在实现中途改。

| 项 | 决定 | 理由 |
|---|---|---|
| 推断对象 | v1 只推断 `q(s_j)`，不推断 `q(C_j)` | 位置推断已经能和 noisy 分开；欲望推断是 v2，有独立停机门 |
| 社会通道 | 必须在生成过程采样 `o ~ A_social[:, S_j]` | 若仍把真值 `S_j` 喂给 `infer_states`，后验会接近 hard，opacity 是假的 |
| 信念是否有记忆 | 每个主体对他者维持 `D_other`，逐步更新 | 没有记忆就只是单步 likelihood，和 noisy 差太小 |
| 他者运动模型 | v1 不用 `B` 预测他者 | 预测动作等于暗中假设已知 `C_j`，会把 v2 偷运进来 |
| preference 写入 | `C_i := q_i(s_j)` | 最干净：欲望直接等于对他者位置的信念 |
| 环境 | 先只做 isolated | 先排除共享环境这条替代路径 |
| 初态 | `[5,2,6]` 和 `[8,8,8]` | 沿用已有两个 regime，便于对照旧结果 |
| 对照 | `off`, `hard`, `noisy`, `infer` | soft/delayed 不再占预算 |
| 更新顺序 | 保持 A→B→C sequential | 不把异步/同步问题混进来 |
| 其余参数 | 复用 `factorial_env_symbolic.py` 的 agent / A / B / weights / policy_len / T | 只改主体间通道 |
| 不做的事 | objet a、公共符号场、Imaginary 耦合、改旧脚本 | 一次只加一个机制 |

`A_social` 用现有 `make_A_matrix(sharpness=σ)`。`σ` 是唯一新参数，表示社会通道清晰度：`σ=1` 近似无噪，`σ` 越小越不透明。

---

## 4. 机制规格

对主体 `i`，`j = (i+1) % 3`，每步：

1. 读取 `S_j = obs[j][1]`（真值只用于采样通道，不写入 `C`）。
2. 采样社会观察：`o_{i←j} ~ Categorical(A_social[:, S_j])`。
3. 滤波：`q_i(s_j) = softmax(log A_social[o] + log D_other_i)`。
4. 写入：`C_S_i = q_i(s_j)`。
5. 记忆：`D_other_i ← q_i(s_j)`。
6. 再跑原来的 R/S/I active inference。

初始化：`D_other_i` 用均匀分布，避免第一步偷偷等于 hard。

Sanity：`σ = 1.0` 时 `A_social` 接近单位阵，`o = S_j` 几乎必然，`q` 应迅速变尖，`infer` 应接近 `hard`。若此时仍不接近，是实现错误，不要调别的参数。

---

## 5. 新指标

旧指标继续用：`sym_var`、收敛率（`< 0.01`）、endpoint regime、`real_var` / `imag_var`。

必须新增，否则无法证明 infer ≠ noisy：

| 指标 | 定义 | 用来看什么 |
|---|---|---|
| time-to-sync | 首次连续 N 步 `sym_var < 0.01` 的时刻；未发生则记缺失 | 推断应比硬复制更慢 |
| misID rate | `argmax q_i(s_j) ≠ S_j` 的步数比例 | 错误认同是否存在 |
| belief lag | `argmax q` 相对真值 `S_j` 的平均延迟 | 滤波记忆是否造成滞后 |
| KL(q ∥ onehot(S_j)) | 信念与真值的散度 | 不透明度是否真的进入了 C |

`noisy` 没有 `q`。对照时用 `C_S` 本身代替 `q` 算 misID / KL，保证两边可比较。

---

## 6. 实验阶段

### Phase 0 — 最小可运行（先证明通道是真的）

单脚本，isolated，`[5,2,6]`，少量 seed 即可。

检查：

- `σ=1.0`：`infer` ≈ `hard`（收敛率接近，misID ≈ 0）
- `σ=0.4`：misID 明显大于 0，time-to-sync 变长或收敛下降
- `off` 仍为 0%

通不过就停，查实现，不要加参数。

### Phase 1 — 主实验：infer 是否不同于 noisy

条件（先 isolated）：

```text
mechanism ∈ {off, hard, noisy, infer}
env_init  ∈ {[5,2,6], [8,8,8]}
σ         ∈ {0.40, 0.70, 1.00}     # 仅 infer 扫 σ
```

`noisy` 保持旧定义（`ε=0.2`），作为“涂糊真值”的基线，不按 σ 改写。  
`hard` / `off` 不依赖 σ。

建议 20 seeds、50 steps，统计口径与 `paper_claims_matrix_v2.md` 一致（bootstrap CI，不用 ad-hoc 5%）。

若 `infer` 与 `noisy` 的 CI 完全重叠，且 misID / lag 也无稳定差，则本路径失败，不要靠加 seed 硬撑。

### Phase 2 — 只在 Phase 1 出现可区分签名后做

二选一，不要并行：

1. **噪声匹配**：把 `noisy` 的 ε 调到与某档 `infer` 的平均 `H(C)` 相近，再比 time-to-sync 和 misID。用来排除“只是平均更糊”。
2. **σ 细扫**：只在 `[5,2,6]` 上扫更密的 σ。用来看不透明度如何改变收敛，**不写 phase transition**。

共享环境放到最后，且只作为附录。主叙事必须先在 isolated 上成立。

### Phase 3 — 欲望推断（可选，有独立门）

仅当 Phase 1–2 显示：位置推断是真机制，且我们明确要问“他者想要什么”而不是“他者在哪”。

那时再把隐藏量从 `s_j` 换成离散假设 `C_j ∈ {e_0,…,e_8}`，用对方的位移/动作做似然。这是另一个实验，不在 v1 脚本里预留开关。

---

## 7. 成功 / 失败 / 停机

**成功（值得写进报告的最小结果）**

同时成立：

1. `σ=1` 接近 hard（通道没写坏）
2. `σ` 下降时，收敛变慢或变差，misID / lag 上升
3. 在相近“糊度”下，`infer` 的错误有时间持续性，`noisy` 的错误更像逐步独立涂抹
4. isolated + off 仍不收敛

可写的主张形状：

> 在当前 toy model 中，当他者只通过带噪社会观察可及时，基于滤波信念的 preference coupling 仍可产生 Symbolic 共识；其错误是历史依赖的，与直接涂糊真值的 noisy 机制不同。

**失败（诚实收口）**

- `infer` 在各 σ 上都 ≈ `noisy`：推断没有多出来，路径 B 在此模型里不成立。
- `σ=1` 也不接近 hard：实现有 bug。
- 中等 σ 完全不收敛：先查 `D_other` 初始化和 `C := q` 是否把 preference 摊得过平，只允许调 `σ` 或 `D_other` 初值，不加新机制。

**停机**

出现以下任一情况就停，写失败报告：

- 为了让图好看开始改 weights / A sharpness / policy_len
- 想顺便加上 objet a 或公共符号场
- 用“这就是大他者”代替机制对比

---

## 8. 主张纪律

沿用 claims matrix 的分层。

可以追求：

- inferred-other preference coupling 是一种**非直接复制**机制
- 社会通道清晰度调节共识速度和误认同
- 滤波错误具有时间持续性，mixture noise 没有

不要写：

- 我们实现了拉康的大他者
- 误认同 = 精神病 / 镜像认同
- σ 扫描是相变
- S=0 / S=8 是 attractor（仍未做扰动恢复）

术语统一用 **inferred-other coupling** / **opaque-other coupling**，论文叙事里不要把 `q(s_j)` 直接叫“欲望”。

---

## 9. 文件与实现约束

新建，不改旧实验：

| 文件 | 用途 |
|---|---|
| `opaque_other_coupling.py` | 主实验脚本，import `factorial_env_symbolic` |
| `opaque_other_results.csv` | 每 seed 明细 |
| `opaque_other_summary.csv` | 条件汇总 |
| `plot_opaque_other.png` | 收敛率、time-to-sync、misID |
| `opaque_other_report.md` | 结果 + 可写/不可写主张 |

实现约束：

- 只加 numpy / matplotlib
- 每 `(mechanism, env_init, σ, seed)` 开头 `np.random.seed(seed)`
- `infer` 在 `σ` 间比较时，社会通道采样必须走 RNG，并计入该 seed
- 不抽取新的公共模型库；现有脚本已经靠复制/import 隔离，保持这个习惯

---

## 10. 工作顺序

1. 写本 roadmap（本文件）
2. Phase 0 冒烟：确认 `σ=1 ≈ hard`、`σ` 低则 misID 上升
3. Phase 1 主实验 + 报告
4. 看停机门：可区分则做 Phase 2 噪声匹配；不可区分则停
5. 只有明确要问“他者想要什么”时才开 Phase 3

预估：Phase 0 一次短跑；Phase 1 与现有 mechanism control 同量级（约 2 初态 × 4 机制，外加 infer 的 3 档 σ）。不要先把 Phase 3 设计进同一脚本。
