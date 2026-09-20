# 从欲望同步到符号秩序

## 一个基于主动推断的拉康式多主体 toy model 研究报告

> 本报告整合原始论文、原始代码复现、v2 修正实验，以及后续的环境因果分解、耦合机制对照，和不透明他者（inferred-other）实验。
>
> 报告的目标不是证明拉康理论，而是回答两个更具体的问题：在一个极简的主动推断模型中，把他人的 Symbolic 位置写入自己的 preference，是否足以产生共同符号坐标？若该位置不能直接读取、只能带噪推断，共识是否仍然出现，以及它是否要求读准他者？

---

## 摘要

拉康精神分析的核心概念——Real、Symbolic、Imaginary（RSI）、欲望和大他者——具有强烈的关系性：它们不是可以简单对应到某个神经元或变量的实体，而是主体、他者、环境和符号网络互动后形成的结构效果。2025 年的论文 *Formalizing Lacanian Psychoanalysis through the Free Energy Principle* 尝试使用自由能原理（Free Energy Principle, FEP）将这些概念转化为可计算的主动推断模型：三个 FEP unit 分别表示 RSI，主体间欲望被表示为 Symbolic 状态的广义同步，三主体循环则被解释为大他者的集体涌现。

本项目对该论文及其开源仓库进行了复现和延伸。复现过程中发现并修正了状态空间不可达、似然矩阵未严格归一化、KL 散度数值误差、自由能变量标量化，以及共享环境造成因果混淆等问题。修正后的实验首先显示：在当前模型配置下，Symbolic coupling ablation 会增加 preference mismatch 的波动和轨迹长度；dyadic Symbolic synchronization 在 20 个随机种子上稳定收敛；三主体循环在独立环境中仍然产生 Symbolic 层集体收敛，而 Real 和 Imaginary 保持主体间差异。

随后进行的 2×2 factorial 实验进一步分离了共享环境和 inter-agent Symbolic preference coupling 的作用。结果显示：在独立环境中，开启 `change_C` 的 Symbolic coupling 足以产生 Symbolic 收敛；关闭该机制后，独立环境中的主体完全不收敛。共享环境本身也能提供较弱的同步路径，但在 `change_C` 已经开启时，其对 Symbolic 收敛的额外贡献接近于零。

环境初态敏感性实验显示，Symbolic 收敛在五组代表性环境初态下高度鲁棒，但最终会进入不同的 Symbolic regime：主要包括所有主体的 S=0 和 S=8 两个端点吸引盆。进一步的 coupling strength sweep 表明，耦合强度的作用不是简单的单调增强：在某些初始条件下，极弱的持续耦合已经足以锁定共同符号状态；在另一些初始条件下，中等耦合产生较多非同步或中间共识状态，只有接近硬更新时才高度稳定。

机制对照进一步表明：硬复制他者状态不是同步的必要条件。软更新、延迟和带不确定性的 message 都能在独立环境中产生 Symbolic 共识，尽管强弱因初态而异。共享环境在关闭耦合时仍可提供 15%–20% 的弱对齐路径。

随后的不透明他者实验把“读取真值”改成“从带噪社会观察滤波得到 `q(s_j)`，再令 `C := q`”。在独立环境中，这一机制仍然产生共识；社会通道越不透明，误认同越高、锁定越慢，但共识并不消失。更关键的是，共识之后误认同可以仍然很高：σ=0.40、初态 [5,2,6] 时，15/20 次轨迹已经对齐，对齐之后仍有约 40% 的步数 `argmax q ≠` 对方真值。也就是说，共同符号坐标可以由错误信念维持，不需要先有一个正确的他者表象。

这些结果提供了一个有限的拉康式计算类比：欲望可以被操作化为把他者位置纳入自身 preference 的过程；该位置不必可读；共同 Symbolic 秩序可以在 Real/Imaginary 差异之上、甚至在对他者的误认之上形成。`change_C` 和 `C := q` 都是操作性控制器，不能等同于拉康理论本身。本文把结果定位为“拉康概念的操作性计算模型”，而非临床模型、标准 FEP 实现或拉康理论的证明。

---

## 1. 我们到底在研究什么？

### 1.1 用一句话说明

本项目研究的是两件事：

> 当一个主体把另一个主体的 Symbolic 位置纳入自己的偏好时，是否会形成共同的符号秩序？若这个位置不能直接读取，只能从带噪观察中推断，共识是否仍然出现？它是否要求读准他者？

第一问在代码里对应硬更新：

```python
C_S_i = onehot(S_j)
```

第二问对应不透明他者：

```python
o ~ A_social[:, S_j]
q = filter(o, D_other)
C_S_i = q
```

这不是把“欲望”或“大他者”命名成一个变量，而是把一个拉康式命题拆成可干预的机制：preference 由他者的符号位置组织；该位置可以是真值，也可以是带记忆的错误信念。

### 1.2 这不是在做什么？

本项目不是：

- 用 9 个离散状态“代表”拉康的全部概念；
- 证明 FEP 等于拉康精神分析；
- 证明某种参数对应临床上的精神病或其他诊断；
- 用 `min G` 直接定义剩余快感；
- 建造一个已经具有语言理解或主体性的 LLM。

更准确的说法是：我们使用一个小型计算模型作为“概念实验室”，把拉康式关系结构变成可运行、可干预、可复现的动力学假设。

---

## 2. 理论背景：为什么用 FEP 讨论拉康？

### 2.1 原论文的基本主张

原论文 [*Formalizing Lacanian Psychoanalysis through the Free Energy Principle*](https://doi.org/10.3389/fpsyg.2025.1574650) 从几个理论相似性出发，尝试把拉康放入 FEP 的数学框架：

1. **不可直接把握的现实**：FEP 中环境的真实状态通常是隐藏的，主体只能通过观察和内部模型进行推断；拉康理论中，主体也不是直接拥有“物自身”，而是通过表征系统构造心理现实。
2. **建构性的表征**：感知不是被动复制环境，而是基于先验、似然和行动的主动推断。
3. **非线性的时间关系**：FEP 中的预测和回溯与拉康的 logical time 都强调，当前意义并不是简单地从过去线性延续而来。
4. **表征失败的驱动作用**：FEP 中 prediction error/free energy 推动更新和行动；原论文把拉康意义上的缺失、不可表征部分和 object petit a 与这种差距联系起来。

这些对应关系使 FEP 成为一种可能的形式化语言。但“理论上可以对应”并不等于“计算模型已经证明了对应”。本项目的后续实验正是要检查：这些计算机制究竟产生了什么，而不是只重复概念名称。

### 2.2 研究线索

该项目位于一条连续的研究轨迹中：

- [*Return to Lacan*（2023）](https://arxiv.org/abs/2309.06707)：提出 FEP-RSI 和 digital twin mind 的初步框架；
- [*Enabling self-identification in intelligent agent*（2024）](https://arxiv.org/abs/2403.07664)：讨论 mirror/rubber hand、identification、Graph II 和 LLM 作为大他者化身；
- [*Formalizing Lacanian Psychoanalysis through the Free Energy Principle*（2025）](https://doi.org/10.3389/fpsyg.2025.1574650)：将 RSI、欲望同步和三主体集体动力学组织为完整 toy model。

本报告集中研究第三步中的 RSI、欲望和大他者部分，并把“Symbolic 集体收敛”拆成环境效应、关系耦合效应、初态依赖、耦合强度、耦合机制，以及他者是否可读。

---

## 3. 原始模型：把拉康概念变成一个主动推断系统

### 3.1 每个 RSI 层都是一个 FEP unit

原始模型把一个主体拆成三个相互作用的 FEP unit：

| 层 | 模型中的角色 | 拉康式解释（操作性） |
|---|---|---|
| Real（R） | 身体/情感状态的离散动力学 | 不完全受符号控制的身体和生物性维度 |
| Symbolic（S） | 语言样、规则样、社会性状态 | 能指、语言、文化和社会规范的层面 |
| Imaginary（I） | 感知、自我形象和身体图式 | 主体对自身和环境的形象化建构 |

每个 unit 使用四类矩阵或向量：

- (A)：似然矩阵，把隐藏状态映射到观察；
- (B)：转移矩阵，描述不同动作如何改变状态；
- (C)：preference，表示系统偏好的未来观察；
- (D)：prior，表示对隐藏状态的先验信念。

每一步大致经历以下循环：

1. 根据当前 observation、似然矩阵和 prior 推断隐藏状态 (q(s))；
2. 计算不同 policy 的 expected free energy；
3. 采样并执行 UP、DOWN 或 STAY；
4. 根据行动后的 observation 更新下一步 prior；
5. 通过 RSI 间的 residual/update 机制传递预测误差。

在本项目的修正版中，状态空间为 0–8，共 9 个离散状态。UP 在 8 处封顶，DOWN 在 0 处封底。

### 3.2 个体内部的 RSI 耦合与主体间的 Symbolic 耦合

这两个机制必须区分：

#### 个体内部：(w_R,w_S,w_I)

这是同一个主体内部 R/S/I 之间的精度或 residual update 权重。它回答：

> 一个主体的 Symbolic 改变，会在多大程度上影响它自己的 Real 和 Imaginary？

#### 主体之间：`change_C`

这是不同主体之间的 Symbolic preference 耦合。它回答：

> 一个主体是否把另一个主体的 Symbolic 位置纳入自己的欲望和行动目标？

原论文把后者解释为欲望的 generalized synchronization，并使用三主体循环：

```text
A → B → C → A
```

三主体的作用不是简单增加一个 agent，而是让每个主体的欲望都被另一个欲望所中介，从而把“我想要什么”转化为“我在他者的符号结构中处于什么位置”。原论文把这种集体动力学进一步解释为大他者的计算性类比。

---

## 4. 复现审计：原始 toy model 的问题与修正

本项目最初并不是直接接受原始代码的结果，而是先进行复现审计。发现的问题如下：

| 问题 | 影响 | 修正 |
|---|---|---|
| 多主体共享同一组环境实例 | 一个主体的动作直接改变其他主体后续读取的环境 | v2 中为每个主体建立独立环境，用于因果隔离实验 |
| 状态写成 0–10，但 UP 在 8 封顶 | 状态 9/10 实际不可达 | 改为 0–8 的 9 状态空间 |
| A 矩阵存在列和为 1.01 | 似然矩阵不严格归一化 | 重新生成并严格归一化 |
| KL 的数值实现出现负值 | 数值误差被误读成负的 divergence | 使用 mask-based 实现并截断极小负值 |
| F 被标量化 | D update 丢失了原本按状态的结构 | 恢复 F 为 state-wise vector |
| J1 被称为 residual free energy | 概念命名过强 | 改为 preference mismatch |
| J2 的 sharpness 扫描混入 (H(A)) 的平凡下降 | 把观测熵下降误当成剩余快感证据 | 分解为 (H(A)) 与 preference-KL |
| Exp 1 直接称为 psychosis model | 把参数消融过度临床化 | 改称 Symbolic-coupling ablation |

### 4.1 共享环境到底是不是 bug？

复现时最初把共享环境当作严重问题，因为它会让同步结果混合两个来源：共同环境和主体间 Symbolic 耦合。但作者后来明确回复：共享环境是原始模型有意的理论设定，表示多个个体在同一个小型物理世界中交互；如果研究重点是社会文化、自我认同或纯粹的主体间关系，则可以使用独立环境、独立似然矩阵甚至不同的 agent 架构。[作者回复](https://github.com/DigitalTwinMind/ActiveInferenceLacan/issues/1#issuecomment-5162354910)

因此，v2 的 independent environment 不是“修复后的唯一正确实现”，而是一个替代模型：

- shared environment：共同物理世界中的交互；
- isolated environment：排除共同物理状态后，观察关系性 Symbolic 耦合是否仍然产生作用。

这个区分后来成为 2×2 factorial 实验的基础。

---

## 5. v2 修正版实验

### 5.1 Exp 1：内部 RSI coupling ablation

这个实验改变主体内部的 (w_R,w_S,w_I)，而不是改变主体之间的 `change_C`。因此它只能说明不同 RSI 精度配置下，当前主体内部动力学如何变化，不能直接被称为精神病模型。

关键结果：

| 配置 | J1 mean | J1 std | path length |
|---|---:|---:|---:|
| Baseline ((2,0.5,1)) | 8.87 | 13.84 | 12.25 |
| (w_S=0) | 24.55 | 25.58 | 19.86 |
| Equal weights ((1,1,1)) | 3.22 | 8.55 | 4.41 |
| Full decoupling | 9.05 | 15.01 | 4.00 |
| Imaginary-emphasis | 15.05 | 14.46 | 15.47 |

在当前参数下，(w_S=0) 增大了 preference mismatch 的波动，也增加了轨迹长度。这支持一个较弱的计算观察：削弱 Symbolic 在主体内部的精度耦合，会使系统更不稳定。

但它不能推出“(w_S=0) 就是精神病”。它只是一个内部耦合消融；它没有临床数据、脑连接指标，也没有覆盖精神病理学的现象结构。

### 5.2 Exp 2：dyadic Symbolic synchronization

在独立环境和 20 个随机种子下，主体间 J3（两个主体 Symbolic 后验的 KL gap）全部收敛到 0。

这说明：在当前模型中，dyadic 的 generalized synchronization 是一个稳定吸引子，而不是单个 seed 的偶然轨迹。

但这里必须牢记机制：主体 A 将 B 的当前 Symbolic 状态作为自己的 preference，B 反之亦然。因此收敛并不是一个没有解释的“自发奇迹”，而是 `change_C` 这一关系性 preference update 在 active-inference 循环中的动力学结果。

### 5.3 Exp 3：expected free energy 分解

原先曾把 J2 直接解释为“剩余快感”或“信息损失”。分解后结果改变了这个判断：

| A sharpness | J2 total | (H(A)) term | KL term |
|---:|---:|---:|---:|
| 0.50 | 48.2 | 3.55 | 44.65 |
| 0.70 | 33.7 | 2.73 | 30.96 |
| 0.90 | 17.0 | 1.33 | 15.68 |
| 0.99 | 3.54 | 0.22 | 3.31 |
| 1.00 | 0.00 | 0.00 | 0.00 |

(H(A)) 随 A 变得 sharp 而下降是平凡的：观测越接近 deterministic，观测熵自然越低。KL 项虽然也是非平凡的，但它反映的是当前预测 observation 与 one-hot preference 的偏离，不能单独证明拉康意义上的剩余快感。

因此目前最稳妥的说法是：

> 在当前有限状态、有限 policy 和 one-hot preference 模型中，lossy likelihood 会产生正的 expected-free-energy floor；这是一种计算观察，而不是“剩余快感等于信息损失”的理论证明。

### 5.4 Exp 4：三主体 Symbolic collective dynamics

在独立环境下运行三主体循环：

```text
A → B → C → A
```

结果为：

| 指标 | 初始 | 最终 |
|---|---:|---:|
| Collective variance | 2.963 | 2.444 |
| Symbolic variance | 1.556 | 0.000 |
| Pairwise distance | 4.970 | 4.426 |

典型终态为：

```text
A = (8, 0, 6)
B = (3, 0, 8)
C = (7, 0, 4)
```

也就是说，三个主体的 Symbolic 状态收敛到了 0，但 Real 和 Imaginary 仍然不同。这是整个 v2 中最重要的现象：共同 Symbolic 状态不要求主体在其他维度完全融合。

它可以作为“大他者是集体 Symbolic 动力学的涌现属性”的一个计算示例，但不能被理解为大他者已经被证明存在。

---

## 6. 2×2 factorial：共享环境和 Symbolic coupling 谁在起作用？

### 6.1 实验设计

两个因素分别是：

| 因素 | 条件 |
|---|---|
| 环境 | shared / isolated |
| 主体间 Symbolic coupling | C_update_on / C_update_off |

这里的 `C_update_off` 只关闭主体之间的 preference update，不关闭主体内部的 Symbolic unit，也不把 (w_S) 置零。

每个条件运行 20 个 seeds、50 steps。shared 和 isolated 都使用同样的环境初始状态 `[5,2,6]`，以避免把环境模式和环境初态混为一谈。

### 6.2 结果

| 条件 | 收敛率 | 最终 Symbolic 方差 | Collective variance | Pairwise distance |
|---|---:|---:|---:|---:|
| shared + C_update_on | 20/20 | 0.0000 ± 0.0000 | 0.1296 ± 0.1383 | 0.7483 ± 0.6101 |
| shared + C_update_off | 4/20 | 0.2222 ± 0.1721 | 0.2556 ± 0.1681 | 1.3135 ± 0.5531 |
| isolated + C_update_on | 19/20 | 0.0111 ± 0.0484 | 2.4481 ± 0.0161 | 4.4290 ± 0.0131 |
| isolated + C_update_off | 0/20 | 4.7111 ± 0.1937 | 3.9963 ± 0.0161 | 5.8897 ± 0.0095 |

### 6.3 正确的解释方式

最清楚的不是粗略比较两个“主效应”，而是看 simple effects：

| 比较 | Symbolic 方差差异 |
|---|---:|
| isolated：C_on − C_off | -4.70 |
| shared：C_on − C_off | -0.22 |
| C_on：shared − isolated | -0.011 |
| C_off：shared − isolated | -4.49 |

结果说明：

1. 在 isolated 环境中，`change_C` 几乎是唯一的 Symbolic 收敛机制；
2. 在 shared 环境中，即使关闭 `change_C`，共同环境也能提供一条较弱的同步路径；
3. 一旦 `change_C` 开启，shared 和 isolated 都接近完全 Symbolic 收敛，因此共享环境对 Symbolic 收敛的边际贡献很小；
4. shared 环境仍然显著降低整体 RSI 的方差和主体间距离，因此它对 R/I 及整体轨迹的同步影响很大。

这不是两个因素简单相加，而是两条可以替代的耦合路径：

```text
共同物理环境  ─────┐
                   ├──> 主体间 Symbolic / RSI 同步
他者参照的 C 更新 ──┘
```

在拉康式语言中，可以说：共同的“现实遭遇”能够促成某种协调，但 Symbolic 秩序并不需要先拥有一个完全共同的物理世界。主体之间的欲望关系本身，就可能生成一个共同的符号位置。

需要注意的是，`change_C` 直接把他者状态写进自身 preference，因此这里的 Symbolic 收敛带有明显的机制必然性。更准确的表述是：

> 该实验分离出了“他者参照的 preference update”对 Symbolic 收敛的作用，而不是证明了 Symbolic Order 会在没有任何人为结构的情况下自发出现。

---

## 7. 环境初态敏感性：Symbolic 秩序是否依赖具体遭遇？

### 7.1 实验设置

固定 isolated + C_update_on，只改变每个主体所拥有的环境初始状态。每组 20 seeds、50 steps：

| 环境初态 | 收敛率 | 最终 Symbolic 方差 | 主要 Symbolic regime |
|---|---:|---:|---|
| [5,2,6] | 19/20 | 0.0111 ± 0.0484 | S=0 |
| [0,0,0] | 20/20 | 0.0000 ± 0.0000 | S=0 |
| [8,8,8] | 20/20 | 0.0000 ± 0.0000 | S=8 |
| [0,2,8] | 20/20 | 0.0000 ± 0.0000 | S=0 |
| [8,6,0] | 20/20 | 0.0000 ± 0.0000 | S=8 |

### 7.2 结果

Symbolic 收敛对这五组环境初态高度鲁棒，但终态选择具有路径依赖：

- [0,0,0] 统一进入 S=0；
- [8,6,0] 统一进入 S=8；
- [5,2,6] 主要进入 S=0，但有一个 seed 未同步；
- [8,8,8] 进入 S=8；
- [0,2,8] 主要进入 S=0，同时出现少量 Imaginary 的采样变体。

这使我们需要区分两个问题：

1. Symbolic 秩序是否形成？——在这些条件下，答案通常是会。
2. 形成哪一种 Symbolic 秩序？——答案取决于初态、随机采样和耦合过程。

因此，当前模型更像是具有多个 Symbolic basin 的系统，而不是总会收敛到唯一的公共状态。

### 7.3 拉康式解释

如果把环境遭遇视为主体进入关系网络时的具体情境，那么结果可以被解释为：情境影响主体进入哪个能指组织，但 Symbolic 组织本身并不等同于物理环境。共同的 S=0 或 S=8 可以被理解为两种不同的集体符号 regime，类似于不同的“缝合点”或能指组织方式。

不过，0 和 8 在代码中只是状态编号，没有语言意义。我们只能说它们具有“不同 Symbolic regime”的结构类比，不能说状态 0 或状态 8 本身代表某个特定能指。

另外，这五组初态是代表性扫描，不是 9³ 个环境状态的穷尽证明。因此报告使用“在测试的环境初态下高度鲁棒”，而不使用“完全不依赖环境”。

---

## 8. Coupling strength sweep：他者规定强度如何改变 Symbolic 动力学？

### 8.1 从硬更新到软更新

把原来的硬更新推广为：

```python
C_S_new = (1 - alpha) * C_S_old \
          + alpha * onehot(other_agent_symbolic_state)
```

其中：

- `alpha=0`：完全不纳入他者的 Symbolic 状态；
- `alpha=1`：每一步完全采用他者的 Symbolic 位置；
- 中间值：主体保留自身 Symbolic 历史，同时部分吸收他者位置。

`alpha` 不是拉康理论中的真实参数。它只是一个操作性参数，用来表示主体对他者符号位置的依赖程度，或更准确地说，表示他者参照在 preference update 中的权重。

### 8.2 主要结果

| alpha | [5,2,6] 收敛率 | [5,2,6] final sv | [8,8,8] 收敛率 | [8,8,8] final sv |
|---:|---:|---:|---:|---:|
| 0.0 | 0% | 4.7111 | 0% | 4.7556 |
| 0.1 | 65% | 0.1000 | 100% | 0.0000 |
| 0.2 | 60% | 0.1333 | 100% | 0.0000 |
| 0.3 | 45% | 0.1889 | 100% | 0.0000 |
| 0.4 | 60% | 0.1111 | 100% | 0.0000 |
| 0.5 | 60% | 0.1333 | 100% | 0.0000 |
| 0.6 | 55% | 0.1667 | 100% | 0.0000 |
| 0.7 | 70% | 0.1556 | 100% | 0.0000 |
| 0.8 | 70% | 0.1778 | 100% | 0.0000 |
| 0.9 | 90% | 0.0222 | 100% | 0.0000 |
| 1.0 | 95% | 0.0111 | 100% | 0.0000 |

### 8.3 没有一个普遍的单一阈值

在 [8,8,8] 环境初态下，alpha 从 0 增加到 0.1 就出现近似 step-like 的转变：只要存在很弱的持续耦合，系统就全部进入 S=8。

在 [5,2,6] 下，情况完全不同：alpha=0.1–0.8 的收敛率在 45%–70% 之间波动，没有单调提升；直到 alpha=0.9 和 1.0 才接近完全收敛。中间区间还出现了较多 `other_conv`，即 Symbolic 方差已经很低，但终态并非简单的全 S=0 或全 S=8。

因此，目前不能把它称为一个普遍、单一的 phase transition。更准确的说法是：

> Symbolic coupling 产生了依赖初态的 regime change；其表现可以是 sharp threshold、宽过渡带、多个中间共识状态或非同步轨迹。

### 8.4 拉康式解释

从拉康的角度，alpha=1 并不一定代表“更健康”或“更真实”的欲望。它意味着主体几乎完全让他者的 Symbolic 位置重写自己的 preference，属于强制性的他者规定。alpha<1 则保留了主体自己的符号惯性，因此他者要求与主体既有位置之间会发生张力。

这提供了一个更有意义的解释：

- alpha=0：主体没有进入他者的 Symbolic 关系；
- 小 alpha：主体开始被他者的符号位置影响；
- 中等 alpha：自身历史与他者规定竞争，容易出现 Symbolic slippage 或中间共识；
- alpha=1：主体被强制纳入他者位置，系统迅速锁定到少数 regime。

这不是说 alpha 直接测量了拉康的“异化”或“认同”，而是说明：当“欲望由他者规定”被写成 preference update 后，主体保留自身位置和纳入他者位置之间的张力，会自然表现为多稳态、路径依赖和不完全收敛。

---

## 9. 机制对照：同步是不是只因为直接复制？

`change_C` 把对方的整数状态写进自己的 preference，同步带有机制内置性。为了检验这一点，在 shared / isolated 和两组初态上比较五种机制（20 seeds，50 steps）：

| 机制 | 规则 | 他者是否可读 |
|---|---|---|
| off | 不更新 C_S | 不读 |
| hard | `C = onehot(S_j)` | 直接读真值 |
| soft | `C ← 0.5 C + 0.5 onehot(S_j)` | 读真值，保留惯性 |
| delayed | `C = onehot(S_j^{t-1})` | 读上一拍真值 |
| noisy | `C = 0.8·onehot(S_j) + 0.2·uniform` | 读真值后涂糊 |

isolated 下的收敛率：

| 机制 | [5,2,6] | [8,8,8] |
|---|---|---|
| off | 0/20 | 0/20 |
| hard | 19/20 | 20/20 |
| soft | 12/20 | 20/20 |
| delayed | 4/20 | 20/20 |
| noisy | 16/20 | 18/20 |

可以站住的只有两句：硬复制不是必要条件——soft、delayed、noisy 的收敛率都显著高于 off；机制之间的精细排序在 n=20 下大多分不清（CI 重叠）。delayed 在 isolated+[5,2,6] 上明显弱于 hard（20% vs 95%），但与 soft / noisy 分不清。

shared+off 仍有 15%–20% 收敛，是弱替代路径；CI 下界碰到 0%，不能写成“共享环境总能同步”。delayed 第一步用不用初始观察兜底，在 n=100 的配对检验下不显著（McNemar p=0.16）。

这一节的作用是收窄 A2：同步来自持续的他者参照，不是来自“每一步精确复制”这一个实现细节。但五种机制仍然都在读真值。`noisy` 尤其容易被误读成不确定性——它只是把已经读到的真值涂糊，argmax 不会错。

---

## 10. 不透明他者：看不见真值时，共识还在不在？

### 10.1 为什么要单独做这一刀

上一节的对照还没有碰到拉康更难的那一层：他者的位置不必是透明的。现有 `noisy` 也不是推断，生成过程里观察本来就等于状态。路径 B 只改主体间通道：

```text
S_j  --(A_social, σ)-->  o  --(滤波, D_other)-->  q(s_j)  --(写入)-->  C_i
```

锁定的设计：只推断位置 `s_j`，不推断欲望 `C_j`；`D_other` 有记忆，初值为均匀分布；不用 `B` 预测他者怎么走；先只做 isolated；对照为 off / hard / noisy / infer。σ=1 时 `A_social` 是单位阵，infer 应接近 hard，作为通道没写坏的体检。

### 10.2 主结果

isolated，20 seeds，50 steps。time-to-sync 定义为首次进入并保持 `sym_var < 0.01` 直到结束的时刻。

**[5,2,6]**

| 条件 | 收敛 | misID 全程 / 后半 / 共识后 | TTS |
|---|---|---|---|
| off | 0/20 | 0.92 / 0.84 / — | — |
| hard | 19/20 [85%,100%] | 0 / 0 / 0 | 10.1 |
| noisy | 16/20 [60%,95%] | 0 / 0 / 0 | 38.2 |
| infer σ=0.40 | 15/20 [55%,90%] | 0.57 / 0.49 / **0.40** | 33.7 |
| infer σ=0.70 | 18/20 [75%,100%] | 0.37 / 0.23 / 0.19 | 26.2 |
| infer σ=1.00 | 20/20 [100%,100%] | 0.03 / 0.00 / 0.00 | 13.2 |

**[8,8,8]**：off 仍为 0%；hard / 三档 infer 均为 20/20；noisy 18/20。infer@0.40 的共识后 misID 仍为 0.25。

Phase 0 体检通过：σ=1 的 misID 接近 0，收敛率接近 hard；σ 下降则 misID 上升；off 仍为 0%。

### 10.3 这一刀真正分开的是什么

1. **共识不要求读准他者。** infer@0.40 在 [5,2,6] 上 15/20 对齐，对齐之后仍有约 40% 的步数认错对方位置。错误会连续维持约 5.5 步，不是逐步独立的涂抹。这不是滤波滞后：如果只是路上认错、到站看清，共识后 misID 应塌到接近 0。
2. **不透明度主要拖慢，不是关掉同步。** [5,2,6] 的 TTS 从 hard 的 10.1、infer@1 的 13.2，升到 infer@0.40 的 33.7。[8,8,8] 上即使 σ=0.40 也 20/20 收敛。
3. **和 noisy 的差别首先是会不会认错，不是收敛率。** 两边收敛 CI 多半重叠。noisy 的 misID=0 是结构事实：它先读真值再涂糊，argmax 不会错。H(C) 上 infer@0.40（0.73）已经接近 noisy（0.84），因此“只是更糊”解释不了持续误认同。
4. **通道可读不等于同一个 endpoint。** infer@1 与 hard 收敛率接近，但 [5,2,6] 上 hard 有 95% 进入 S=0，infer@1 有 60% other_conv。额外的随机抽样会改变终态混合，不改变“能否对齐”。

### 10.4 可以写、不能写

可以写：在当前 toy model 中，基于滤波信念的 preference coupling 是一种非直接复制机制；共同符号坐标可以由对他者位置的错误信念维持。

不能写：这是大他者、欲望或 `Che vuoi?` 的形式化。推断的是“他在哪”，不是“他要什么”。σ 不是相变参数。S=0 / S=8 仍称 endpoint regime。

---

## 11. 从计算结果回到拉康：我们究竟形式化了什么？

### 11.1 欲望：已经覆盖的是位置参照，不是欲望之谜

原论文将欲望解释为主体间 Symbolic 状态的 generalized synchronization。代码层面并不是凭空命名：preference 会根据他者的 Symbolic 状态被重写。

必须分开两层：

| 机制 | 在问什么 | 更接近的理论层 |
|---|---|---|
| `C_i = S_j` | 我想站到他现在站的地方，而且我直接知道那个地方 | 模仿 / 认同式趋近 |
| `C_i = q(s_j)` | 我想站到我以为他站的地方 | 他者位置不透明 |
| `C_i = q(C_j)`（未做） | 他到底要什么 | `Che vuoi?` |

已经覆盖的是第一层，以及第二层的计算版。尚未覆盖：欲望的转喻延宕、对象滑移、语言语义，以及主体不可能填补的缺失。`change_C` 和 inferred-other 都只是“欲望经由他者的操作性代理”，不是欲望本身。

### 11.2 不透明：共同秩序不必以正确表象为前提

拉康那里，位置相对外显，欲望相对隐蔽。本项目没有形式化后者，但把前者做成了不透明：主体只能通过带噪通道构造他者坐标。

不透明他者实验的理论含量不在“又能同步”，而在：

> 共同符号坐标可以对齐，同时主体仍然认错对方站在哪。

这比硬复制更接近“他者不透明”，仍然远不是大他者。三个 agent 交换的是整数状态上的信念，不是能指、法或话语。更准确的说法是：共享的公开站位，不必等于关于他者的正确知识。

### 11.3 大他者：从实体变成集体动力学

当前模型没有一个叫作“Other”的中心节点。大他者存在于三角关系的整体动力学中：

```text
A 的 Symbolic preference 受 B 影响
B 的 Symbolic preference 受 C 影响
C 的 Symbolic preference 受 A 影响
```

这种设计使“大他者”不能被还原为某个单一 agent。它更接近一个由主体之间的相互规定关系产生的集体符号场。

在这个意义上，isolated + C_update_on 的结果很重要：共同物理环境不是 Symbolic 收敛的必要条件。Symbolic 秩序可以由关系网络本身形成，而不是简单复制一个共同外部状态。

但这仍然是一个非常有限的“集体符号场”：三个 agent 传递的是整数状态，不是语言、规则、规范或文化意义。因此只能说它展示了“关系性集体状态”的计算形式，而不能说它已经模拟了社会中的大他者。

### 11.4 RSI：共同 Symbolic 与不同 R/I 的并存

最有拉康解释价值的结果不是所有主体最终相同，而是：

```text
S_A = S_B = S_C
但 R_A ≠ R_B ≠ R_C，I_A ≠ I_B ≠ I_C
```

这可以被理解为：共同的符号秩序不消除主体在身体性和形象性层面的差异。主体可以共享语言、规则或社会位置，同时仍有不同的身体感受和自我表象。

不过必须承认，这个结果一部分来自模型设计：C_R、C_I、D_R、D_I 是主体特定的 one-hot preference，R/I 的终态因而被强烈锁定。它是一个有用的结构示范，但不是 R/S/I 分离在自然系统中的独立发现。

### 11.5 多稳态：Symbolic 秩序不是唯一答案

S=0 和 S=8 两个 regime 说明，在同样的耦合架构下，系统可以进入不同的共同符号配置。环境初态和采样历史影响进入哪个 basin，而耦合机制决定是否能形成稳定的共同状态。

这与拉康理论中符号秩序的历史性、主体位置的关系性和意义的非自然性形成了可讨论的结构类比：共同意义不是物理状态的直接反映，而是在关系、历史和位置中稳定下来。

但不能进一步说“拉康预言了 S=0 和 S=8 两个吸引子”。状态编号没有语义，吸引子数量也取决于状态空间、policy、矩阵和更新规则。

### 11.6 剩余快感与 object petit a：目前只到操作性假设

原论文把表征差距、prediction error 和 object petit a 联系起来。本项目一度尝试把 J2 解释为剩余快感，但 G 分解显示：

- (H(A)) 项的下降有明显的平凡来源；
- KL 项反映 preference mismatch 和预测偏离，但并不自动具有拉康语义；
- 当 A 接近 identity 时，当前模型的 J2 可以归零。

因此，本项目没有得到“剩余快感等于信息损失”的结论。比较稳妥的说法是：

> 只要似然是 lossy、preference 是有限的 one-hot 目标并且 policy 集合有限，系统就可能保留一个正的 expected-free-energy floor。这个 floor 可以作为研究“不可完全满足”的计算入口，但还不是 plus-de-jouir 的形式化。

---

## 12. 这项延续工作的真正贡献

这不是一个新的 FEP 算法，也不是一个大规模神经模型。它的价值主要体现在以下几个方面。

### 12.1 把原始同步结果拆成可解释的因果结构

原始共享环境中的同步结果无法直接判断来自哪里。2×2 factorial 显示：

- 共享环境是一条同步路径；
- `change_C` 是另一条更强的 Symbolic 同步路径；
- 两者存在替代关系和交互，而不是简单相加。

这使“Symbolic 集体收敛”从一个图像现象变成了一个有干预对照的计算观察。

### 12.2 从单条轨迹推进到可审计的机制实验

原始 toy model 主要依赖短时间、单 seed 的轨迹。后续实验加入了：

- 20-seed dyadic/triadic 统计；
- 环境 shared/isolated 因果分解；
- 五组环境初态敏感性；
- 11 个 alpha 的 coupling strength sweep；
- 五种耦合机制对照，以及 delayed 初始化的配对检验；
- 不透明他者：社会通道、滤波信念、共识后误认同。

这没有把 toy model 变成真实心理模型，但显著提高了它作为计算研究对象的可审计性。

### 12.3 暴露了原模型的多稳态结构

原始报告主要关注“是否收敛”。后续实验显示，更重要的问题是：

> 系统会收敛到哪个 Symbolic regime？

S=0、S=8 以及中等 alpha 下的其他共识状态，说明 Symbolic coupling 不只是一个同步开关，而是在有限状态空间中塑造了一个具有 basin、历史依赖和 regime change 的动力系统。

### 12.4 把“他者参照”从复制推进到错误信念

原先最强的机制批评是：`change_C` 直接复制，同步是写进去的。机制对照表明复制不是唯一写法；不透明他者表明，即使真值不可读，共识仍可出现，而且可以对齐后仍认错。这是本仓库里目前最强的新观察。

### 12.5 给拉康解释增加了可检验的中间层

“欲望是他者的欲望”不能直接变成实验变量。但可以先拆成可检验命题：

1. 主体的 preference 是否依赖他者的 Symbolic 状态？
2. 这种依赖是否足以产生跨主体 Symbolic 收敛？
3. 收敛是否需要共同物理环境？
4. 硬复制是不是必要？
5. 他者位置不可读时，共识是否还在？
6. 共识是否要求读准他者？
7. Symbolic 共识是否可以和 R/I 差异并存？

本项目已经对 1–7 给出了当前 toy model 范围内的答案。第 6 问的答案是：不要求。尚未问的是：主体能否推断他者的欲望而不是位置。

---

## 13. 局限与不能声称的内容

### 13.1 复制已经被拆开，但通道仍然极简

直接复制、软更新、延迟、涂糊真值、滤波推断都已经比过。剩余的机制内置性换成了另一件更小的事：社会通道是手写的离散似然，信念更新没有他者运动模型，写入规则仍是 `C := q`。未来若要增强理论意义，需要的是推断欲望 `q(C_j)`、共享语言表征，或同步更新对照，而不是再扫一遍 α。

### 13.2 Symbolic 状态没有语义

S=0 和 S=8 只是状态编号。没有语言、能指、规则、文化语境或可解释的符号内容。因此“Symbolic order”在这里指的是关系性状态层，而不是完整的语言秩序。

### 13.3 R/I 的独立性部分来自硬编码

主体的 R/I preference 和 prior 被固定为不同 one-hot 状态。R/I 保持差异是有意义的结构展示，但不能当作自然产生的心理事实。

### 13.4 当前不是标准或完整的 FEP 实现

模型使用了 FEP 风格的离散推断、expected free energy 和 policy selection，但 `D_update`、residual 传递和 `change_C` 都是项目特定的操作规则。它是 FEP-inspired toy implementation，不是对 canonical active inference 的完整推导。

### 13.5 没有实现真正的拓扑学

RSI 的 Borromean 性质在原论文和本项目中主要通过 message passing 和精度权重实现。它是动力学类比，不是 Borromean knot 的真正拓扑不变量、同伦或同调计算。

### 13.6 统计与参数空间仍然有限

多数条件是 20 seeds、50 steps。结果适合描述当前模型的机制，不足以支持普遍定理。报告使用“当前配置下成立”，不用未经检验的“必然”或“普遍”。共识后 misID 是轨迹内均值，不是对“错误信念吸引子”的扰动恢复证明。

### 13.7 不涉及临床诊断

`w_S=0` 只能称为内部 coupling ablation。误认同也不是精神病或镜像认同。没有临床数据，不能推出患者层面的结论。

### 13.8 推断的是位置，不是欲望

不透明他者实验问的是 `q(s_j)`。模型里的 `C` 仍是可满足的 one-hot 目标。即使下一步去推断 `C_j`，成功了也不能说形式化了拉康欲望。

---

## 14. 最终结论

本项目的最终结论可以压缩为五句：

1. 在这个有限状态主动推断 toy model 中，主体间的 preference coupling 足以在独立环境中产生 Symbolic consensus；硬复制不是必要条件。
2. 共享物理环境是一条弱替代路径，不是 Symbolic 收敛的必要条件。
3. 他者位置不可读时，滤波信念仍能产生共识；不透明度主要拖慢锁定，并提高误认同。
4. 共识之后仍可认错对方位置。共同符号坐标可以由错误信念维持，不要求正确的他者表象。
5. 这些是对“欲望经由他者组织、他者不必透明”的操作性计算类比，不是拉康理论的证明。

原论文提供了一个很粗糙但有方向性的形式化框架。本项目把它变成一个可审计的机制实验室：先修正实现，再拆环境和关系耦合，然后检查初态、强度、机制，最后把真值通道改成推断。

当前最有价值的对象不是“9 个状态分别代表什么”，而是：

> 一个主体如何在他者的符号位置——包括他以为的那个位置——中形成自己的 preference，以及多个主体如何在差异甚至误认仍然存在的情况下形成共同的 Symbolic regime。

---

## 15. 复现与文件清单

项目在 WSL 中运行。假设仓库路径为：

```bash
/Users/yui/Dev/vibecoding/ActiveInferenceLacan
```

核心运行入口和结果文件：

| 文件 | 用途 |
|---|---|
| `src/deep_experiments_v2.py` | v2 修正版四组基础实验 |
| `src/factorial_env_symbolic.py` | shared/isolated × C_update_on/off 的 2×2 实验 |
| `outputs/factorial/factorial_summary.csv` | 2×2 汇总结果 |
| `outputs/factorial/factorial_timeseries.csv` | 2×2 每 seed、每 step 的时间序列 |
| `src/env_init_sensitivity.py` | 环境初态敏感性实验 |
| `outputs/env_init/env_init_summary.csv` | 环境初态汇总结果 |
| `src/symbolic_coupling_sweep.py` | alpha=0.0–1.0 的耦合强度扫描 |
| `src/coupling_mechanism_control.py` | off / hard / soft / delayed / noisy |
| `src/delayed_initialization_sensitivity.py` | delayed 第一步初始化对照 |
| `src/opaque_other_coupling.py` | 不透明他者：infer `q(s_j)` |
| `outputs/opaque_other/opaque_other_report.md` | 不透明他者实验报告 |
| `docs/analysis/paper_claims_matrix_v2.md` | 可写 / 不可写主张 |
| `docs/notes/实验记录与观察.md` | v1/v2 实验过程记录 |
| `docs/analysis/FINAL_REPORT.md` | 本综合研究报告 |

运行命令：

```bash
cd /Users/yui/Dev/vibecoding/ActiveInferenceLacan
python3 src/deep_experiments_v2.py
python3 src/factorial_env_symbolic.py
python3 src/env_init_sensitivity.py
python3 src/symbolic_coupling_sweep.py
python3 src/coupling_mechanism_control.py
python3 src/opaque_other_coupling.py phase1
```

---

## 参考资料

1. Li, L. and Li, C. (2023). [*Return to Lacan: an approach to digital twin mind with free energy principle*](https://arxiv.org/abs/2309.06707).
2. Li, L. and Li, C. (2024). [*Enabling self-identification in intelligent agent: insights from computational psychoanalysis*](https://arxiv.org/abs/2403.07664).
3. Li, L. and Li, C. (2025). [*Formalizing Lacanian psychoanalysis through the free energy principle*](https://doi.org/10.3389/fpsyg.2025.1574650).
4. [原始 ActiveInferenceLacan 仓库](https://github.com/DigitalTwinMind/ActiveInferenceLacan).
5. [关于共享环境设定的作者回复](https://github.com/DigitalTwinMind/ActiveInferenceLacan/issues/1#issuecomment-5162354910).

---

## Checklist status

- 写作格式：未指定投稿 venue，使用 user-custom research report format。
- 主线：task → gap → computational insight → model mechanism → evidence → limitation，已闭合。
- 证据：原论文、v2 修正、2×2、初态敏感性、coupling sweep、机制对照、delayed 初始化、不透明他者均已纳入。
- 拉康解释：保留“欲望—他者—Symbolic 秩序—位置不透明”的理论翻译，并明确推断的是位置不是欲望。
- 未执行：投稿格式适配和外部 reviewer 打分。当前目标是仓库中的中文综合研究报告。
- 未解决风险：Symbolic 状态没有语言语义；未推断欲望；没有临床或真实多主体数据。
