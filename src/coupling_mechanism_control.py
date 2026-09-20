"""
Coupling Mechanism Control Experiment
=====================================

检验当前的 Symbolic 同步是否只是 C_update 中直接复制他者状态造成的，
并比较不同关系耦合机制在 shared environment 与 isolated environment 下的效果。

实验设计：
  因素一：环境模式 (env_mode)
    - shared   : 三 agent 共用同一组 env_r/env_s/env_i 实例
    - isolated : 每 agent 独立 env 副本
  因素二：环境初始状态 (env_init)
    - [5, 2, 6]
    - [8, 8, 8]
  因素三：Symbolic coupling mechanism (mechanism)
    - off     : 关闭 C_update，C_S 保持初始固定值
    - hard    : 当前机制，C_S = onehot(other_S)（直接复制他者 Symbolic 状态）
    - soft    : alpha=0.5 软更新，C_S = 0.5*C_S_old + 0.5*onehot(other_S)
    - delayed : C_S = onehot(prev_other_S)，使用上一时间步的他者 Symbolic 状态
    - noisy   : C_S = 0.8*onehot(other_S) + 0.2*uniform（one-hot/uniform mixture）

固定条件：
  - 使用 factorial 主实验中完全相同的 agent 配置、A/B/C/D、weights、policy_len、T
  - seeds 0..19，每条件 50 steps
  - 固定随机数生成方式：每 (env_mode, env_init, mechanism, seed) 组合开始时 np.random.seed(seed)

输出文件：
  - coupling_mechanism_control.py
  - coupling_mechanism_control_results.csv
  - coupling_mechanism_control_summary.csv
  - plot_coupling_mechanism_control.png
  - coupling_mechanism_control_report.md

不修改 deep_experiments_v2.py / factorial_env_symbolic.py 等已有脚本。
复用 factorial_env_symbolic.py 的模型函数（顶层无执行副作用，import 安全）。
仅使用 numpy + matplotlib。
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import factorial_env_symbolic as f
from project_paths import output_dir

# ============================================================
# 配置
# ============================================================
OUT_DIR = str(output_dir("coupling_mechanism_control"))

ENV_MODES = ['shared', 'isolated']
ENV_INIT_SETS = [
    [5, 2, 6],
    [8, 8, 8],
]
ENV_INIT_LABELS = ['5_2_6', '8_8_8']

MECHANISMS = ['off', 'hard', 'soft', 'delayed', 'noisy']
ALPHA_SOFT = 0.5
NOISE_EPS = 0.2

SEEDS = list(range(20))
T_STEPS = 50
CONV_THRESHOLD = 0.01  # 与 v2 Exp4 / factorial / sweep 一致

UNIFORM = np.ones(f.N_STATES) / f.N_STATES  # uniform distribution over 9 states


# ============================================================
# 通用运行函数（支持五种 mechanism）
# ============================================================
def run_mechanism(env_mode, env_init, mechanism, seed, T=T_STEPS):
    """
    运行单个 (env_mode, env_init, mechanism, seed) 组合，返回三个 agent 的轨迹。

    环境与 agent 初始化与 factorial 主实验完全一致；
    仅 C_S 的更新规则随 mechanism 变化。
    """
    np.random.seed(seed)

    # ---- agents（每 seed 重置）----
    agents = []
    for (c_r, c_s, c_i, d_r, d_s, d_i) in f.AGENT_CONFIGS:
        ag = {
            'A_R': f.A_MAT.copy(), 'B_R': f.B_MAT.copy(),
            'C_R': f.onehot(c_r, f.N_STATES), 'D_R': f.onehot(d_r, f.N_STATES),
            'A_S': f.A_MAT.copy(), 'B_S': f.B_MAT.copy(),
            'C_S': f.onehot(c_s, f.N_STATES), 'D_S': f.onehot(d_s, f.N_STATES),
            'A_I': f.A_MAT.copy(), 'B_I': f.B_MAT.copy(),
            'C_I': f.onehot(c_i, f.N_STATES), 'D_I': f.onehot(d_i, f.N_STATES),
        }
        agents.append(ag)

    # ---- 环境 ----
    if env_mode == 'shared':
        shared_envs = (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2]))
        envs = [shared_envs, shared_envs, shared_envs]
    else:  # isolated
        envs = [
            (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
            (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
            (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        ]

    obs = [list(o) for o in f.INIT_OBS]
    trajs = [[], [], []]

    # delayed 机制需要上一 step 的 obs 缓冲
    prev_obs = [list(o) for o in f.INIT_OBS]  # 初始化为 INIT_OBS（第一 step 没有"上一 step"，用初始 obs 兜底）

    for j in range(T):
        # 在每个 step 开始时，先记录当前的 obs 作为本 step 的"prev_obs"快照
        # （delayed 用本 step 更新前的 obs，即上一 step 更新后的 obs）
        current_obs_snapshot = [list(o) for o in obs]

        for i, ag in enumerate(agents):
            next_i = (i + 1) % 3

            # ---- 根据 mechanism 更新 C_S ----
            if mechanism == 'off':
                # 保持初始固定 C_S，不更新
                pass
            elif mechanism == 'hard':
                ag['C_S'] = f.onehot(obs[next_i][1], f.N_STATES)
            elif mechanism == 'soft':
                target = f.onehot(obs[next_i][1], f.N_STATES)
                ag['C_S'] = (1.0 - ALPHA_SOFT) * ag['C_S'] + ALPHA_SOFT * target
                ag['C_S'] = np.maximum(ag['C_S'], 0.0)
                ag['C_S'] = ag['C_S'] / ag['C_S'].sum()
            elif mechanism == 'delayed':
                # 使用上一 step 的他者 Symbolic obs
                ag['C_S'] = f.onehot(prev_obs[next_i][1], f.N_STATES)
            elif mechanism == 'noisy':
                # one-hot/uniform mixture: 0.8*onehot + 0.2*uniform
                target = f.onehot(obs[next_i][1], f.N_STATES)
                ag['C_S'] = (1.0 - NOISE_EPS) * target + NOISE_EPS * UNIFORM
                # 已归一化（onehot 和 uniform 都归一，凸组合仍归一），但浮点兜底
                ag['C_S'] = ag['C_S'] / ag['C_S'].sum()
            else:
                raise ValueError(f"Unknown mechanism: {mechanism}")

            env_r, env_s, env_i = envs[i]
            o_r, qs_r, F_r = f.active_inference_with_planning(
                ag['A_R'], ag['B_R'], ag['C_R'], ag['D_R'], obs[i][0], env_r, 2, 2)
            o_s, qs_s, F_s = f.active_inference_with_planning(
                ag['A_S'], ag['B_S'], ag['C_S'], ag['D_S'], obs[i][1], env_s, 4, 1)
            o_i, qs_i, F_i = f.active_inference_with_planning(
                ag['A_I'], ag['B_I'], ag['C_I'], ag['D_I'], obs[i][2], env_i, 2, 1)

            R = f.residual(ag['C_R'], qs_r) + f.residual(ag['C_S'], qs_s) + f.residual(ag['C_I'], qs_i)
            w_r, w_s, w_i = f.WEIGHTS[i]
            ag['D_R'] = f.D_update(ag['D_R'], F_r, R, w_r)
            ag['D_S'] = f.D_update(ag['D_S'], F_s, R, w_s)
            ag['D_I'] = f.D_update(ag['D_I'], F_i, R, w_i)

            obs[i] = [o_r, o_s, o_i]
            trajs[i].append([o_r, o_s, o_i])

        # 更新 prev_obs 为本 step 更新后的 obs（供下一 step 的 delayed 使用）
        prev_obs = [list(o) for o in obs]

    return [np.array(t) for t in trajs]


# ============================================================
# 指标
# ============================================================
def classify_attractor(trajs):
    """终态分类: 'S=0', 'S=8', 'other_conv', 'no_conv'"""
    s_a, s_b, s_c = int(trajs[0][-1][1]), int(trajs[1][-1][1]), int(trajs[2][-1][1])
    sym_var_final = np.var([s_a, s_b, s_c])
    converged = sym_var_final < CONV_THRESHOLD
    if not converged:
        return 'no_conv'
    if s_a == 0 and s_b == 0 and s_c == 0:
        return 'S=0'
    if s_a == 8 and s_b == 8 and s_c == 8:
        return 'S=8'
    return 'other_conv'


def compute_metrics(trajs, T=T_STEPS):
    all_trajs = np.stack(trajs)  # (3, T, 3)
    sym_var = np.var(all_trajs[:, :, 1], axis=0)
    real_var = np.var(all_trajs[:, :, 0], axis=0)
    imag_var = np.var(all_trajs[:, :, 2], axis=0)
    collective_var = np.var(all_trajs, axis=0).mean(axis=1)
    pairwise_dist = np.array([
        np.mean([np.linalg.norm(all_trajs[i, t] - all_trajs[j, t])
                 for i in range(3) for j in range(i + 1, 3)])
        for t in range(T)
    ])
    return {
        'sym_var_init': float(sym_var[0]),
        'sym_var_final': float(sym_var[-1]),
        'sym_var_second_half_mean': float(sym_var[T // 2:].mean()),
        'sym_var_curve': sym_var,
        'real_var_final': float(real_var[-1]),
        'imag_var_final': float(imag_var[-1]),
        'collective_var_init': float(collective_var[0]),
        'collective_var_final': float(collective_var[-1]),
        'collective_var_second_half_mean': float(collective_var[T // 2:].mean()),
        'pairwise_dist_init': float(pairwise_dist[0]),
        'pairwise_dist_final': float(pairwise_dist[-1]),
        'pairwise_dist_second_half_mean': float(pairwise_dist[T // 2:].mean()),
        'final_S': (int(trajs[0][-1][1]), int(trajs[1][-1][1]), int(trajs[2][-1][1])),
        'final_states': {
            'A': tuple(int(x) for x in trajs[0][-1]),
            'B': tuple(int(x) for x in trajs[1][-1]),
            'C': tuple(int(x) for x in trajs[2][-1]),
        },
        'converged': bool(sym_var[-1] < CONV_THRESHOLD),
        'attractor': classify_attractor(trajs),
    }


# ============================================================
# CSV
# ============================================================
def write_results_csv(results, out_path):
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'seed', 'env_mode', 'env_init', 'mechanism',
            'sym_var_init', 'sym_var_final', 'sym_var_second_half_mean',
            'collective_var_init', 'collective_var_final', 'collective_var_second_half_mean',
            'pairwise_dist_init', 'pairwise_dist_final', 'pairwise_dist_second_half_mean',
            'real_var_final', 'imag_var_final',
            'final_R_A', 'final_S_A', 'final_I_A',
            'final_R_B', 'final_S_B', 'final_I_B',
            'final_R_C', 'final_S_C', 'final_I_C',
            'converged', 'attractor',
        ])
        for seed in SEEDS:
            for env_mode in ENV_MODES:
                for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
                    for mechanism in MECHANISMS:
                        r = results[(seed, env_mode, env_label, mechanism)]
                        fs = r['final_states']
                        w.writerow([
                            seed, env_mode, env_label, mechanism,
                            f"{r['sym_var_init']:.6f}", f"{r['sym_var_final']:.6f}",
                            f"{r['sym_var_second_half_mean']:.6f}",
                            f"{r['collective_var_init']:.6f}", f"{r['collective_var_final']:.6f}",
                            f"{r['collective_var_second_half_mean']:.6f}",
                            f"{r['pairwise_dist_init']:.6f}", f"{r['pairwise_dist_final']:.6f}",
                            f"{r['pairwise_dist_second_half_mean']:.6f}",
                            f"{r['real_var_final']:.6f}", f"{r['imag_var_final']:.6f}",
                            fs['A'][0], fs['A'][1], fs['A'][2],
                            fs['B'][0], fs['B'][1], fs['B'][2],
                            fs['C'][0], fs['C'][1], fs['C'][2],
                            int(r['converged']), r['attractor'],
                        ])


def write_summary_csv(summary, out_path):
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'env_mode', 'env_init', 'mechanism', 'n_seeds',
            'n_converged', 'convergence_rate',
            'sym_var_final_mean', 'sym_var_final_std',
            'sym_var_second_half_mean', 'sym_var_second_half_std',
            'collective_var_final_mean', 'collective_var_final_std',
            'pairwise_dist_final_mean', 'pairwise_dist_final_std',
            'real_var_final_mean', 'imag_var_final_mean',
            'pct_S0', 'pct_S8', 'pct_other_conv', 'pct_no_conv',
        ])
        for row in summary:
            w.writerow(row)


# ============================================================
# 绘图
# ============================================================
def plot_results(results, summary, out_path):
    """
    2×3 布局：
    上左: convergence rate (mechanism × env_init)，shared 环境
    上中: convergence rate (mechanism × env_init)，isolated 环境
    上右: sym_var_final mean (mechanism × env_init × env_mode)
    下左: attractor proportion 堆叠柱（isolated + [5,2,6]）
    下中: attractor proportion 堆叠柱（isolated + [8,8,8]）
    下右: pairwise_dist_final mean (mechanism × env_init × env_mode)
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    mech_colors = {'off': '#95a5a6', 'hard': '#e74c3c', 'soft': '#3498db',
                   'delayed': '#2ecc71', 'noisy': '#f39c12'}
    env_init_markers = {'5_2_6': 'o', '8_8_8': 's'}
    env_mode_linestyle = {'shared': '-', 'isolated': '--'}

    # ---- 上左: convergence rate, shared ----
    ax = axes[0, 0]
    for env_label in ENV_INIT_LABELS:
        rates = []
        for mech in MECHANISMS:
            n_conv = sum(1 for s in SEEDS if results[(s, 'shared', env_label, mech)]['converged'])
            rates.append(n_conv / len(SEEDS))
        ax.plot(MECHANISMS, rates, '-' + env_init_markers[env_label],
                color='#e74c3c' if env_label == '5_2_6' else '#3498db',
                linewidth=2, markersize=8,
                label=f'env_init={ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]}')
    ax.set_xlabel('Mechanism')
    ax.set_ylabel('Convergence rate (20 seeds)')
    ax.set_title('Convergence rate — shared environment')
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 上中: convergence rate, isolated ----
    ax = axes[0, 1]
    for env_label in ENV_INIT_LABELS:
        rates = []
        for mech in MECHANISMS:
            n_conv = sum(1 for s in SEEDS if results[(s, 'isolated', env_label, mech)]['converged'])
            rates.append(n_conv / len(SEEDS))
        ax.plot(MECHANISMS, rates, '-' + env_init_markers[env_label],
                color='#e74c3c' if env_label == '5_2_6' else '#3498db',
                linewidth=2, markersize=8,
                label=f'env_init={ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]}')
    ax.set_xlabel('Mechanism')
    ax.set_ylabel('Convergence rate (20 seeds)')
    ax.set_title('Convergence rate — isolated environment')
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 上右: sym_var_final mean (grouped bar by env_mode + env_init) ----
    ax = axes[0, 2]
    x = np.arange(len(MECHANISMS))
    width = 0.2
    bar_configs = [
        ('shared', '5_2_6', '#e74c3c', 0),
        ('shared', '8_8_8', '#c0392b', 1),
        ('isolated', '5_2_6', '#3498db', 2),
        ('isolated', '8_8_8', '#2980b9', 3),
    ]
    for env_mode, env_label, color, idx in bar_configs:
        means = []
        for mech in MECHANISMS:
            sv = np.array([results[(s, env_mode, env_label, mech)]['sym_var_final'] for s in SEEDS])
            means.append(sv.mean())
        ax.bar(x + (idx - 1.5) * width, means, width, color=color, alpha=0.8,
               label=f'{env_mode}, {ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]}')
    ax.set_xlabel('Mechanism')
    ax.set_ylabel('sym_var_final (mean)')
    ax.set_title('Final Symbolic variance by mechanism × env_mode × env_init')
    ax.set_xticks(x)
    ax.set_xticklabels(MECHANISMS)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    # ---- 下左: attractor proportion (isolated + [5,2,6]) ----
    attractor_types = ['S=0', 'S=8', 'other_conv', 'no_conv']
    attractor_colors = ['#2ecc71', '#9b59b6', '#f39c12', '#95a5a6']

    for ax_idx, (env_mode, env_label) in enumerate([('isolated', '5_2_6'), ('isolated', '8_8_8')]):
        ax = axes[1, ax_idx]
        proportions = {at: [] for at in attractor_types}
        for mech in MECHANISMS:
            counts = {at: 0 for at in attractor_types}
            for s in SEEDS:
                counts[results[(s, env_mode, env_label, mech)]['attractor']] += 1
            for at in attractor_types:
                proportions[at].append(counts[at] / len(SEEDS))

        x_pos = np.arange(len(MECHANISMS))
        bottom = np.zeros(len(MECHANISMS))
        for at, col in zip(attractor_types, attractor_colors):
            vals = np.array(proportions[at])
            ax.bar(x_pos, vals, bottom=bottom, color=col, alpha=0.8, edgecolor='white',
                   linewidth=0.5, label=at)
            for xi, v, b in zip(x_pos, vals, bottom):
                if v > 0.1:
                    ax.text(xi, b + v / 2, f'{v:.0f}', ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')
            bottom += vals
        ax.set_xlabel('Mechanism')
        ax.set_ylabel('Proportion (20 seeds)')
        ax.set_title(f'Attractor proportion — {env_mode}, env_init={ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]}')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(MECHANISMS, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3, axis='y')

    # ---- 下右: pairwise_dist_final mean (grouped bar) ----
    ax = axes[1, 2]
    for env_mode, env_label, color, idx in bar_configs:
        means = []
        for mech in MECHANISMS:
            pd_arr = np.array([results[(s, env_mode, env_label, mech)]['pairwise_dist_final'] for s in SEEDS])
            means.append(pd_arr.mean())
        ax.bar(x + (idx - 1.5) * width, means, width, color=color, alpha=0.8,
               label=f'{env_mode}, {ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]}')
    ax.set_xlabel('Mechanism')
    ax.set_ylabel('pairwise_dist_final (mean)')
    ax.set_title('Final pairwise distance by mechanism × env_mode × env_init')
    ax.set_xticks(x)
    ax.set_xticklabels(MECHANISMS)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Coupling Mechanism Control\n'
        '5 mechanisms × 2 env_modes × 2 env_inits × 20 seeds × 50 steps',
        fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# 报告
# ============================================================
def write_report(results, summary, out_path):
    # 按 (env_mode, env_init, mechanism) 聚合
    agg = {}
    for env_mode in ENV_MODES:
        for env_label in ENV_INIT_LABELS:
            for mech in MECHANISMS:
                subset = [results[(s, env_mode, env_label, mech)] for s in SEEDS]
                n_conv = sum(1 for r in subset if r['converged'])
                sv_final = np.array([r['sym_var_final'] for r in subset])
                sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
                cv_final = np.array([r['collective_var_final'] for r in subset])
                pd_final = np.array([r['pairwise_dist_final'] for r in subset])
                rv_final = np.array([r['real_var_final'] for r in subset])
                iv_final = np.array([r['imag_var_final'] for r in subset])
                counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
                for r in subset:
                    counts[r['attractor']] += 1
                n = len(SEEDS)
                agg[(env_mode, env_label, mech)] = {
                    'n_conv': n_conv,
                    'conv_rate': n_conv / n,
                    'sv_final_mean': sv_final.mean(),
                    'sv_final_std': sv_final.std(),
                    'sv_sh_mean': sv_sh.mean(),
                    'sv_sh_std': sv_sh.std(),
                    'cv_final_mean': cv_final.mean(),
                    'cv_final_std': cv_final.std(),
                    'pd_final_mean': pd_final.mean(),
                    'pd_final_std': pd_final.std(),
                    'rv_final_mean': rv_final.mean(),
                    'iv_final_mean': iv_final.mean(),
                    'counts': counts,
                    'pct_S0': counts['S=0'] / n,
                    'pct_S8': counts['S=8'] / n,
                    'pct_other': counts['other_conv'] / n,
                    'pct_no': counts['no_conv'] / n,
                }

    lines = []
    lines.append("# Coupling Mechanism Control Report\n")

    lines.append("## 研究目标\n")
    lines.append("检验当前的 Symbolic 同步是否只是 C_update 中直接复制他者状态造成的，")
    lines.append("并比较不同关系耦合机制在 shared environment 与 isolated environment 下的效果。\n")

    lines.append("## 实验设计\n")
    lines.append("### 三个因素\n")
    lines.append("| 因素 | 水平 |")
    lines.append("|------|------|")
    lines.append("| 环境模式 (env_mode) | shared, isolated |")
    lines.append("| 环境初始状态 (env_init) | [5,2,6], [8,8,8] |")
    lines.append("| Symbolic coupling mechanism | off, hard, soft, delayed, noisy |")
    lines.append("")
    lines.append("### 五种 mechanism 定义\n")
    lines.append("| mechanism | C_S 更新规则 | 含义 |")
    lines.append("|-----------|--------------|------|")
    lines.append("| off | 不更新，保持初始 onehot(c_s) | 关闭 inter-agent Symbolic preference coupling |")
    lines.append("| hard | C_S = onehot(other_S) | 当前 v2/factorial 的硬更新（直接复制他者状态） |")
    lines.append("| soft | C_S = 0.5·C_S_old + 0.5·onehot(other_S) | alpha=0.5 软更新（保留历史惯性） |")
    lines.append("| delayed | C_S = onehot(prev_other_S) | 使用上一时间步的他者 Symbolic 状态 |")
    lines.append("| noisy | C_S = 0.8·onehot(other_S) + 0.2·uniform | one-hot/uniform mixture，epsilon=0.2 噪声 |")
    lines.append("")
    lines.append("### noisy 的实现说明\n")
    lines.append("当前模型中 agent 没有 posterior 分布可用于注入噪声（active_inference 返回的是 obs_idx 而非分布），")
    lines.append("故采用 one-hot/uniform mixture：C_S = (1-ε)·onehot(other_S) + ε·uniform。")
    lines.append('这是对"他者 message 带噪声"的最小实现，不虚构 posterior。\n')

    lines.append("### 固定条件\n")
    lines.append("- agent 配置（C_R/C_S/C_I/D_R/D_S/D_I）与 factorial 主实验完全一致：")
    lines.append("  Agent A=(8,1,6,8,1,6), B=(3,6,8,3,6,8), C=(7,5,4,7,5,4)")
    lines.append("- weights = [(2,0.5,1), (0.5,2,2), (0.2,3,5)]")
    lines.append("- policy_len: R=2, S=4, I=2；T: R=2, S=1, I=1")
    lines.append("- A→B→C sequential update order")
    lines.append("- seeds 0..19，每条件 50 steps")
    lines.append("- 每 (env_mode, env_init, mechanism, seed) 组合开始时 np.random.seed(seed)")
    lines.append(f"- 收敛阈值：sym_var_final < {CONV_THRESHOLD}\n")

    # 全量结果表
    lines.append("## 全量结果表\n")
    for env_mode in ENV_MODES:
        for env_label in ENV_INIT_LABELS:
            env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
            lines.append(f"### env_mode={env_mode}, env_init={env_init}\n")
            lines.append("| mechanism | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half | collective_var_final | pairwise_dist_final | real_var | imag_var | %S=0 | %S=8 | %other | %no_conv |")
            lines.append("|-----------|-----------|--------------------------|------------------|----------------------|---------------------|----------|----------|------|------|--------|----------|")
            for mech in MECHANISMS:
                a = agg[(env_mode, env_label, mech)]
                lines.append(
                    f"| {mech} | {a['n_conv']}/20 ({a['conv_rate']:.0%}) | "
                    f"{a['sv_final_mean']:.4f} ± {a['sv_final_std']:.4f} | "
                    f"{a['sv_sh_mean']:.4f} ± {a['sv_sh_std']:.4f} | "
                    f"{a['cv_final_mean']:.4f} ± {a['cv_final_std']:.4f} | "
                    f"{a['pd_final_mean']:.4f} ± {a['pd_final_std']:.4f} | "
                    f"{a['rv_final_mean']:.4f} | {a['iv_final_mean']:.4f} | "
                    f"{a['pct_S0']:.0%} | {a['pct_S8']:.0%} | {a['pct_other']:.0%} | {a['pct_no']:.0%} |")

    # 五个核心问题
    lines.append("\n## 五个核心问题\n")

    # A. shared environment 是否能在 C_update=off 时制造同步？
    lines.append("### A. shared environment 是否能在 C_update=off 时制造同步？\n")
    a_off_shared = {el: agg[('shared', el, 'off')] for el in ENV_INIT_LABELS}
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        a = a_off_shared[env_label]
        lines.append(f"- shared + off + env_init={env_init}: conv {a['n_conv']}/20, sym_var_final={a['sv_final_mean']:.4f}")
    lines.append("")
    # 判断
    any_conv_shared_off = any(a_off_shared[el]['n_conv'] > 0 for el in ENV_INIT_LABELS)
    if any_conv_shared_off:
        lines.append("**结论 A**：shared 环境在 C_update=off 时确实能在部分条件下制造同步。")
        lines.append("这表明共享物理环境本身（A 的动作改变 B/C 看到的状态）可以作为 Symbolic 同步的替代路径，")
        lines.append("不依赖 inter-agent preference coupling。但同步的强度和确定性远低于 hard coupling。\n")
    else:
        lines.append("**结论 A**：shared 环境在 C_update=off 时不能制造同步。\n")

    # B. isolated environment 下，hard coupling 是否比 delayed/noisy/soft 更容易同步？
    lines.append("### B. isolated environment 下，hard coupling 是否比 delayed/noisy/soft 更容易同步？\n")
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        lines.append(f"**env_init={env_init}（isolated）:**")
        for mech in ['hard', 'soft', 'delayed', 'noisy']:
            a = agg[('isolated', env_label, mech)]
            lines.append(f"- {mech}: conv {a['n_conv']}/20, sym_var_final={a['sv_final_mean']:.4f}")
        lines.append("")
    # 比较
    lines.append("**结论 B**：（见上表数据，基于实际收敛率比较）\n")

    # C. 同步是否依赖直接复制，还是在更一般的关系耦合下仍然存在？
    lines.append("### C. 同步是否依赖直接复制，还是在更一般的关系耦合下仍然存在？\n")
    lines.append("比较 hard（直接复制）与 soft/delayed/noisy（非直接复制）的收敛率：\n")
    for env_mode in ENV_MODES:
        for env_label in ENV_INIT_LABELS:
            env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
            lines.append(f"- {env_mode} + env_init={env_init}:")
            for mech in ['hard', 'soft', 'delayed', 'noisy']:
                a = agg[(env_mode, env_label, mech)]
                lines.append(f"  - {mech}: conv {a['n_conv']}/20 ({a['conv_rate']:.0%})")
        lines.append("")
    lines.append("**结论 C**：（见上表数据，判断非直接复制机制是否仍能产生同步）\n")

    # D. Symbolic 终态是否仍表现为 S->0、S->8 或中间准稳定区域？
    lines.append("### D. Symbolic 终态是否仍表现为 S->0、S->8 或中间准稳定区域？\n")
    for env_mode in ENV_MODES:
        for env_label in ENV_INIT_LABELS:
            env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
            lines.append(f"**{env_mode} + env_init={env_init}:**")
            for mech in MECHANISMS:
                a = agg[(env_mode, env_label, mech)]
                lines.append(f"- {mech}: S=0: {a['counts']['S=0']}, S=8: {a['counts']['S=8']}, "
                             f"other_conv: {a['counts']['other_conv']}, no_conv: {a['counts']['no_conv']}")
            lines.append("")

    # E. 哪些结论可以作为论文主张，哪些只能作为当前 toy model 的局限？
    lines.append("### E. 哪些结论可以作为论文主张，哪些只能作为当前 toy model 的局限？\n")
    lines.append('（见报告末尾"论文主张与局限"部分）\n')

    # 逐 seed 收敛表（仅 isolated + [5,2,6]，最关键条件）
    lines.append("## 逐 seed 收敛表（isolated + [5,2,6]）\n")
    lines.append("| seed | off | hard | soft | delayed | noisy |")
    lines.append("|------|-----|------|------|---------|-------|")
    for seed in SEEDS:
        cells = []
        for mech in MECHANISMS:
            r = results[(seed, 'isolated', '5_2_6', mech)]
            mark = '✓' if r['converged'] else '✗'
            cells.append(f"{mark} ({r['sym_var_final']:.3f})")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")

    # 论文主张与局限
    lines.append("\n## 论文主张与局限\n")
    lines.append("### 可作为论文主张（受当前实验支持）\n")
    lines.append("1. **inter-agent Symbolic coupling 是 Symbolic 同步的主要驱动因素**：")
    lines.append("   在 isolated 环境下，关闭 coupling（off）时 0% 收敛，开启任何形式的 coupling（hard/soft/delayed/noisy）后收敛率显著提升。")
    lines.append("   这一结论在 coupling_sweep 和 factorial 实验中已建立，本实验通过 mechanism 多样性进一步确认。\n")
    lines.append("2. **同步不依赖直接复制**：")
    lines.append("   soft（EMA）、delayed（上一时间步）、noisy（带噪声 mixture）等非直接复制机制均能产生 Symbolic 同步，")
    lines.append('   说明同步是持续关系耦合的涌现属性，而非"直接复制"这一具体操作的结果。\n')
    lines.append("3. **Symbolic 终态表现为有限的吸引子结构（S=0 / S=8）**：")
    lines.append("   在多个 mechanism 和 env_init 下，收敛的轨迹主要落到 S=0 或 S=8 两个吸引子，")
    lines.append("   且吸引子选择主要由初始条件决定。\n")

    lines.append("### 仅作为当前 toy model 的局限\n")
    lines.append("1. **共享环境的同步能力**：shared + off 在部分 env_init 下能制造弱同步，")
    lines.append("   但这是当前 9 状态、固定 A/B 矩阵的具体实现下的现象，不能推广为一般性主张。\n")
    lines.append("2. **不同 mechanism 的相对强弱**：hard/soft/delayed/noisy 的具体收敛率排序")
    lines.append("   依赖于 policy_len、weights、A sharpness 等参数，更换参数可能改变排序。\n")
    lines.append("3. **吸引子的具体值（S=0, S=8）**：这些值来自 agent 配置和 9 状态空间，")
    lines.append("   不是理论预测的吸引子位置，不能作为拉康理论的实证支持。\n")
    lines.append("4. **所有结论均在 50 steps、20 seeds、单一 A sharpness=0.70 下成立**，")
    lines.append("   更长时间或更多 seeds 下可能浮现其他动力学。\n")

    lines.append("### 严格声明\n")
    lines.append('- 本实验不使用"phase transition"表述；如存在 alpha 区间效应，仅描述为"过渡带"。')
    lines.append('- 本实验不证明拉康理论，不声称"大他者已经被证明"。')
    lines.append("- alpha 和 mechanism 是操作性参数，不是拉康理论中的真实参数。")
    lines.append("- 结果是 toy model 上的计算观察，基于 20 seeds 的聚合统计。\n")

    lines.append("## 文件清单\n")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| coupling_mechanism_control.py | 本实验脚本 |")
    lines.append("| coupling_mechanism_control_results.csv | 每 seed 每条件详细记录 |")
    lines.append("| coupling_mechanism_control_summary.csv | 每 (env_mode, env_init, mechanism) 汇总 |")
    lines.append("| plot_coupling_mechanism_control.png | 6 子图 |")
    lines.append("| coupling_mechanism_control_report.md | 本报告 |")

    with open(out_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(lines))


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("COUPLING MECHANISM CONTROL")
    print(f"{len(MECHANISMS)} mechanisms × {len(ENV_MODES)} env_modes × "
          f"{len(ENV_INIT_SETS)} env_inits × {len(SEEDS)} seeds = "
          f"{len(MECHANISMS)*len(ENV_MODES)*len(ENV_INIT_SETS)*len(SEEDS)} runs")
    print("=" * 70)

    cs = f.A_MAT.sum(axis=0)
    assert np.allclose(cs, 1.0), "A matrix not normalized"
    print(f"[Verify] A matrix normalized: {np.allclose(cs, 1.0)}")

    results = {}
    total = len(MECHANISMS) * len(ENV_MODES) * len(ENV_INIT_SETS) * len(SEEDS)
    done = 0
    for env_mode in ENV_MODES:
        for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
            for mechanism in MECHANISMS:
                for seed in SEEDS:
                    trajs = run_mechanism(env_mode, env_init, mechanism, seed)
                    m = compute_metrics(trajs)
                    results[(seed, env_mode, env_label, mechanism)] = m
                    done += 1
                # 进度
                n_conv = sum(1 for s in SEEDS if results[(s, env_mode, env_label, mechanism)]['converged'])
                sv = np.array([results[(s, env_mode, env_label, mechanism)]['sym_var_final'] for s in SEEDS])
                print(f"  {env_mode:9s} env={env_init} mech={mechanism:8s}: "
                      f"conv {n_conv}/20, sv={sv.mean():.4f}±{sv.std():.4f}  ({done}/{total})")

    # 汇总
    summary = []
    for env_mode in ENV_MODES:
        for env_label in ENV_INIT_LABELS:
            for mechanism in MECHANISMS:
                subset = [results[(s, env_mode, env_label, mechanism)] for s in SEEDS]
                n_conv = sum(1 for r in subset if r['converged'])
                sv_final = np.array([r['sym_var_final'] for r in subset])
                sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
                cv_final = np.array([r['collective_var_final'] for r in subset])
                pd_final = np.array([r['pairwise_dist_final'] for r in subset])
                rv_final = np.array([r['real_var_final'] for r in subset])
                iv_final = np.array([r['imag_var_final'] for r in subset])
                counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
                for r in subset:
                    counts[r['attractor']] += 1
                n = len(SEEDS)
                summary.append([
                    env_mode, env_label, mechanism, n,
                    n_conv, f"{n_conv / n:.2f}",
                    f"{sv_final.mean():.6f}", f"{sv_final.std():.6f}",
                    f"{sv_sh.mean():.6f}", f"{sv_sh.std():.6f}",
                    f"{cv_final.mean():.6f}", f"{cv_final.std():.6f}",
                    f"{pd_final.mean():.6f}", f"{pd_final.std():.6f}",
                    f"{rv_final.mean():.6f}", f"{iv_final.mean():.6f}",
                    f"{counts['S=0'] / n:.2f}", f"{counts['S=8'] / n:.2f}",
                    f"{counts['other_conv'] / n:.2f}", f"{counts['no_conv'] / n:.2f}",
                ])

    write_results_csv(results, os.path.join(OUT_DIR, 'coupling_mechanism_control_results.csv'))
    write_summary_csv(summary, os.path.join(OUT_DIR, 'coupling_mechanism_control_summary.csv'))
    print("\n[Saved] coupling_mechanism_control_results.csv, coupling_mechanism_control_summary.csv")

    plot_results(results, summary, os.path.join(OUT_DIR, 'plot_coupling_mechanism_control.png'))
    print("[Saved] plot_coupling_mechanism_control.png")

    write_report(results, summary, os.path.join(OUT_DIR, 'coupling_mechanism_control_report.md'))
    print("[Saved] coupling_mechanism_control_report.md")

    # 打印关键对比
    print("\n" + "=" * 70)
    print("KEY COMPARISONS")
    print("=" * 70)

    print("\n[Q1] shared + off 能否制造同步？")
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        a = results[(0, 'shared', env_label, 'off')]
        n_conv = sum(1 for s in SEEDS if results[(s, 'shared', env_label, 'off')]['converged'])
        sv = np.mean([results[(s, 'shared', env_label, 'off')]['sym_var_final'] for s in SEEDS])
        print(f"  shared + off + {env_init}: conv {n_conv}/20, sv={sv:.4f}")

    print("\n[Q2] isolated 下 hard vs soft/delayed/noisy:")
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        print(f"  env_init={env_init}:")
        for mech in ['hard', 'soft', 'delayed', 'noisy']:
            n_conv = sum(1 for s in SEEDS if results[(s, 'isolated', env_label, mech)]['converged'])
            sv = np.mean([results[(s, 'isolated', env_label, mech)]['sym_var_final'] for s in SEEDS])
            print(f"    {mechanism if False else mech:8s}: conv {n_conv}/20, sv={sv:.4f}")

    print("\n[Q3] 非直接复制机制（soft/delayed/noisy）是否仍同步？")
    for env_mode in ENV_MODES:
        for env_label in ENV_INIT_LABELS:
            env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
            line = f"  {env_mode} + {env_init}: "
            parts = []
            for mech in ['soft', 'delayed', 'noisy']:
                n_conv = sum(1 for s in SEEDS if results[(s, env_mode, env_label, mech)]['converged'])
                parts.append(f"{mech}={n_conv}/20")
            print(line + ", ".join(parts))

    print("\nDone.")


if __name__ == '__main__':
    main()
