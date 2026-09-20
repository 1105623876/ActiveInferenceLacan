"""
Symbolic Coupling Strength Sweep
================================

把 inter-agent Symbolic preference coupling 从硬更新改为软更新（EMA），
扫描 coupling strength alpha ∈ {0.0, 0.1, ..., 1.0}：

    C_S_new = (1 - alpha) * C_S_old + alpha * onehot(other_agent_symbolic_state)

- alpha = 0 : 关闭 inter-agent Symbolic preference coupling（C_S 保持初始固定值）
- alpha = 1 : 等价于 factorial 主实验的硬 change_C（C_S 完全设为 other 的当前 S obs）
- 0 < alpha < 1 : 主体保留历史 Symbolic preference 的惯性，同时按 alpha 比例纳入他者的 Symbolic position

实验设置：
- 只做 isolated 环境（排除 shared physical environment）
- 两组 env_init: [5,2,6] 和 [8,8,8]
- 11 个 alpha × 2 env_init × 20 seeds = 440 次运行
- 50 steps
- 保持 factorial_env_symbolic.py 中的 A/B/D、C_R/C_I、agent configs、weights、policy_len、T、
  A->B->C sequential update order

不修改 deep_experiments_v2.py；复用 factorial_env_symbolic.py 的模型函数。
仅使用 numpy + matplotlib。不引入 identification/LLM/topology 等新概念。
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
OUT_DIR = str(output_dir("coupling_sweep"))

ENV_MODE = 'isolated'
ENV_INIT_SETS = [
    [5, 2, 6],
    [8, 8, 8],
]
ENV_INIT_LABELS = ['5_2_6', '8_8_8']

ALPHAS = [round(0.1 * i, 1) for i in range(11)]  # 0.0, 0.1, ..., 1.0

SEEDS = list(range(20))
T_STEPS = 50
CONV_THRESHOLD = 0.01  # 与 v2 Exp4 / factorial / env_init_sensitivity 一致


# ============================================================
# 软更新版运行函数
# ============================================================
def run_with_alpha(env_init, alpha, seed, T=T_STEPS):
    """
    isolated 环境，软更新 inter-agent Symbolic coupling，指定 alpha 和 env_init。
    复用 f.AGENT_CONFIGS / f.WEIGHTS / f.INIT_OBS / f.A_MAT / f.B_MAT 等。

    alpha=0 : C_S 保持初始 onehot(c_s) 不变（等价于 factorial C_update_off）
    alpha=1 : C_S = onehot(other_S)（等价于 factorial C_update_on 硬更新）
    """
    np.random.seed(seed)

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

    # isolated：每 agent 独立 env 副本，初始状态统一为 env_init
    envs = [
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
    ]

    obs = [list(o) for o in f.INIT_OBS]
    trajs = [[], [], []]

    for j in range(T):
        for i, ag in enumerate(agents):
            next_i = (i + 1) % 3
            # 软更新 C_S
            target = f.onehot(obs[next_i][1], f.N_STATES)
            ag['C_S'] = (1.0 - alpha) * ag['C_S'] + alpha * target
            # 数值稳定：保证非负且归一化（软更新理论上保持，但浮点误差累积时兜底）
            ag['C_S'] = np.maximum(ag['C_S'], 0.0)
            ag['C_S'] = ag['C_S'] / ag['C_S'].sum()

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

    return [np.array(t) for t in trajs]


# ============================================================
# 指标 + attractor 分类
# ============================================================
def classify_attractor(trajs):
    """
    根据三 agent 的最终 Symbolic observation 分类吸引子。
    返回: 'S=0', 'S=8', 'other_conv', 'no_conv'
    """
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
    sym_var = np.var(all_trajs[:, :, 1], axis=0)  # (T,)
    real_var = np.var(all_trajs[:, :, 0], axis=0)
    imag_var = np.var(all_trajs[:, :, 2], axis=0)
    return {
        'sym_var_init': float(sym_var[0]),
        'sym_var_final': float(sym_var[-1]),
        'sym_var_second_half_mean': float(sym_var[T // 2:].mean()),
        'sym_var_curve': sym_var,
        'real_var_final': float(real_var[-1]),
        'imag_var_final': float(imag_var[-1]),
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
def write_timeseries_csv(results, out_path):
    """每 alpha 每 env_init 每 seed 每 step。"""
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'alpha', 'env_init', 'seed', 'step',
            'sym_var', 'real_var', 'imag_var',
            'obs_S_A', 'obs_S_B', 'obs_S_C',
        ])
        for alpha in ALPHAS:
            for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
                for seed in SEEDS:
                    r = results[(alpha, env_label, seed)]
                    trajs = r['_trajs']
                    all_t = np.stack(trajs)
                    for t in range(T_STEPS):
                        w.writerow([
                            alpha, env_label, seed, t,
                            f"{r['sym_var_curve'][t]:.6f}",
                            f"{np.var(all_t[:, t, 0]):.6f}",
                            f"{np.var(all_t[:, t, 2]):.6f}",
                            int(all_t[0, t, 1]), int(all_t[1, t, 1]), int(all_t[2, t, 1]),
                        ])


def write_summary_csv(summary, out_path):
    """每 alpha 每 env_init 汇总。"""
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'alpha', 'env_init', 'n_seeds', 'n_converged', 'convergence_rate',
            'sym_var_final_mean', 'sym_var_final_std',
            'sym_var_second_half_mean', 'sym_var_second_half_std',
            'real_var_final_mean', 'real_var_final_std',
            'imag_var_final_mean', 'imag_var_final_std',
            'pct_S0', 'pct_S8', 'pct_other_conv', 'pct_no_conv',
        ])
        for row in summary:
            w.writerow(row)


# ============================================================
# 绘图
# ============================================================
def plot_sweep(results, summary, out_path):
    """
    2×3 布局：
    上左: alpha vs convergence rate（两条线，两个 env_init）
    上中: alpha vs final sym_var（两条线）
    上右: alpha vs final real_var + imag_var（两条线 × 两个维度，或分两组）
    下左: [5,2,6] attractor proportion 堆叠柱
    下中: [8,8,8] attractor proportion 堆叠柱
    下右: alpha vs sym_var_second_half（两条线）
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    color_526 = '#e74c3c'
    color_888 = '#3498db'

    # ---- 上左: convergence rate ----
    ax = axes[0, 0]
    for env_label, env_init, color in zip(ENV_INIT_LABELS, ENV_INIT_SETS, [color_526, color_888]):
        rates = []
        for alpha in ALPHAS:
            n_conv = sum(1 for s in SEEDS if results[(alpha, env_label, s)]['converged'])
            rates.append(n_conv / len(SEEDS))
        ax.plot(ALPHAS, rates, '-o', color=color, linewidth=2, markersize=6,
                label=f'env_init={env_init}')
    ax.set_xlabel('alpha (coupling strength)')
    ax.set_ylabel('Convergence rate (20 seeds)')
    ax.set_title('Symbolic convergence rate vs alpha')
    ax.set_xticks(ALPHAS)
    ax.set_ylim(-0.05, 1.1)
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 上中: final sym_var ----
    ax = axes[0, 1]
    for env_label, env_init, color in zip(ENV_INIT_LABELS, ENV_INIT_SETS, [color_526, color_888]):
        means = []
        stds = []
        for alpha in ALPHAS:
            sv = np.array([results[(alpha, env_label, s)]['sym_var_final'] for s in SEEDS])
            means.append(sv.mean())
            stds.append(sv.std())
        means = np.array(means); stds = np.array(stds)
        ax.plot(ALPHAS, means, '-o', color=color, linewidth=2, markersize=6,
                label=f'env_init={env_init} (mean)')
        ax.fill_between(ALPHAS, means - stds, means + stds, color=color, alpha=0.15)
    ax.set_xlabel('alpha (coupling strength)')
    ax.set_ylabel('Symbolic variance (final)')
    ax.set_title('Final Symbolic variance vs alpha (mean ± std)')
    ax.set_xticks(ALPHAS)
    ax.legend()
    ax.grid(alpha=0.3)

    # ---- 上右: real_var & imag_var final ----
    ax = axes[0, 2]
    for env_label, env_init, color in zip(ENV_INIT_LABELS, ENV_INIT_SETS, [color_526, color_888]):
        rv = [np.mean([results[(alpha, env_label, s)]['real_var_final'] for s in SEEDS]) for alpha in ALPHAS]
        iv = [np.mean([results[(alpha, env_label, s)]['imag_var_final'] for s in SEEDS]) for alpha in ALPHAS]
        ax.plot(ALPHAS, rv, '-o', color=color, linewidth=2, markersize=5,
                label=f'Real var, env={env_init}')
        ax.plot(ALPHAS, iv, '--s', color=color, linewidth=2, markersize=5,
                label=f'Imag var, env={env_init}')
    ax.set_xlabel('alpha (coupling strength)')
    ax.set_ylabel('Real / Imaginary variance (final)')
    ax.set_title('Real & Imaginary final variance vs alpha')
    ax.set_xticks(ALPHAS)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- 下左 & 下中: attractor proportion 堆叠柱 ----
    attractor_types = ['S=0', 'S=8', 'other_conv', 'no_conv']
    attractor_colors = ['#2ecc71', '#9b59b6', '#f39c12', '#95a5a6']

    for ax_idx, (env_label, env_init) in enumerate(zip(ENV_INIT_LABELS, ENV_INIT_SETS)):
        ax = axes[1, ax_idx]
        proportions = {at: [] for at in attractor_types}
        for alpha in ALPHAS:
            counts = {at: 0 for at in attractor_types}
            for s in SEEDS:
                counts[results[(alpha, env_label, s)]['attractor']] += 1
            for at in attractor_types:
                proportions[at].append(counts[at] / len(SEEDS))

        x = np.arange(len(ALPHAS))
        bottom = np.zeros(len(ALPHAS))
        for at, col in zip(attractor_types, attractor_colors):
            vals = np.array(proportions[at])
            ax.bar(x, vals, bottom=bottom, color=col, alpha=0.8, edgecolor='white',
                   linewidth=0.5, label=at)
            # 在柱内标比例（>10% 才标）
            for xi, v, b in zip(x, vals, bottom):
                if v > 0.1:
                    ax.text(xi, b + v / 2, f'{v:.0f}', ha='center', va='center',
                            fontsize=8, color='white', fontweight='bold')
            bottom += vals
        ax.set_xlabel('alpha (coupling strength)')
        ax.set_ylabel('Proportion (20 seeds)')
        ax.set_title(f'Attractor proportion vs alpha\nenv_init={env_init}')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{a:.1f}' for a in ALPHAS], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(alpha=0.3, axis='y')

    # ---- 下右: sym_var second half ----
    ax = axes[1, 2]
    for env_label, env_init, color in zip(ENV_INIT_LABELS, ENV_INIT_SETS, [color_526, color_888]):
        means = []
        for alpha in ALPHAS:
            sv = np.array([results[(alpha, env_label, s)]['sym_var_second_half_mean'] for s in SEEDS])
            means.append(sv.mean())
        ax.plot(ALPHAS, means, '-o', color=color, linewidth=2, markersize=6,
                label=f'env_init={env_init}')
    ax.set_xlabel('alpha (coupling strength)')
    ax.set_ylabel('Symbolic variance (2nd half mean)')
    ax.set_title('Second-half Symbolic variance vs alpha')
    ax.set_xticks(ALPHAS)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(
        'Symbolic Coupling Strength Sweep\n'
        'isolated env, soft update C_S = (1-α)·C_S_old + α·onehot(other_S), 20 seeds × 50 steps per α',
        fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# 报告
# ============================================================
def write_report(results, summary, out_path):
    lines = []
    lines.append("# Symbolic Coupling Strength Sweep Report\n")
    lines.append("## 研究目的\n")
    lines.append("把 inter-agent Symbolic preference coupling 从硬更新改为软更新（EMA），")
    lines.append("扫描 coupling strength alpha ∈ {0.0, 0.1, ..., 1.0}：\n")
    lines.append("```")
    lines.append("C_S_new = (1 - alpha) * C_S_old + alpha * onehot(other_agent_symbolic_state)")
    lines.append("```\n")
    lines.append("- alpha=0：关闭 inter-agent Symbolic preference coupling（C_S 保持初始固定值）")
    lines.append("- alpha=1：等价于 factorial 主实验的硬 change_C")
    lines.append("- 0<alpha<1：主体保留历史 Symbolic preference 的惯性，同时按 alpha 比例纳入他者的 Symbolic position\n")

    lines.append("## 理论框架（操作性定义）\n")
    lines.append("- alpha 是\"他者对主体欲望/符号偏好的规定强度\"的操作性参数。")
    lines.append("- alpha 越高，表示主体越强地把他者的 Symbolic position 纳入自己的 preference。")
    lines.append("- 如果存在临界区间，可以讨论从个体漂移到共同 Symbolic order 的动力学转变。")
    lines.append("- 如果所有 alpha>0 都收敛，则说明当前模型的关键不是临界强度，而是是否存在持续的他者耦合。")
    lines.append("- alpha 不是拉康理论中的真实参数；结果不证明大他者或欲望理论。\n")

    lines.append("## 实验设置\n")
    lines.append("- isolated 环境（排除 shared physical environment）")
    lines.append("- 两组 env_init: [5,2,6] 和 [8,8,8]")
    lines.append(f"- 11 个 alpha × 2 env_init × 20 seeds = {11*2*20} 次运行")
    lines.append("- 50 steps；收敛阈值 sym_var_final < 0.01")
    lines.append("- 保持 factorial_env_symbolic.py 中的 A/B/D、C_R/C_I、agent configs、weights、policy_len、T、A→B→C 顺序\n")

    lines.append("## 全量结果表\n")
    lines.append("### env_init = [5, 2, 6]\n")
    lines.append("| alpha | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half (mean±std) | real_var_final | imag_var_final | %S=0 | %S=8 | %other_conv | %no_conv |")
    lines.append("|-------|-----------|--------------------------|----------------------------|----------------|----------------|------|------|-------------|----------|")
    for row in summary:
        if row[1] == '5_2_6':
            alpha = row[0]  # already a string like "0.0"
            n_conv = row[3]
            conv_rate = row[4]
            sv_m = row[5]; sv_s = row[6]
            sv_sh_m = row[7]; sv_sh_s = row[8]
            rv_m = row[9]
            iv_m = row[11]
            p0 = row[13]; p8 = row[14]; po = row[15]; pn = row[16]
            lines.append(f"| {alpha} | {n_conv}/20 ({conv_rate}) | {sv_m} ± {sv_s} | {sv_sh_m} ± {sv_sh_s} | {rv_m} | {iv_m} | {p0} | {p8} | {po} | {pn} |")

    lines.append("\n### env_init = [8, 8, 8]\n")
    lines.append("| alpha | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half (mean±std) | real_var_final | imag_var_final | %S=0 | %S=8 | %other_conv | %no_conv |")
    lines.append("|-------|-----------|--------------------------|----------------------------|----------------|----------------|------|------|-------------|----------|")
    for row in summary:
        if row[1] == '8_8_8':
            alpha = row[0]  # already a string like "0.0"
            n_conv = row[3]
            conv_rate = row[4]
            sv_m = row[5]; sv_s = row[6]
            sv_sh_m = row[7]; sv_sh_s = row[8]
            rv_m = row[9]
            iv_m = row[11]
            p0 = row[13]; p8 = row[14]; po = row[15]; pn = row[16]
            lines.append(f"| {alpha} | {n_conv}/20 ({conv_rate}) | {sv_m} ± {sv_s} | {sv_sh_m} ± {sv_sh_s} | {rv_m} | {iv_m} | {p0} | {p8} | {po} | {pn} |")

    # 主分析
    lines.append("\n## 主要分析\n")

    # 1. coupling threshold
    lines.append("### 是否存在明显 coupling threshold？\n")
    # 找每个 env_init 的首次达到 50% / 90% / 100% 收敛的 alpha
    for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        lines.append(f"**env_init={env_init}:**\n")
        first_50 = None; first_90 = None; first_100 = None
        for alpha in ALPHAS:
            n_conv = sum(1 for s in SEEDS if results[(alpha, env_label, s)]['converged'])
            rate = n_conv / len(SEEDS)
            if first_50 is None and rate >= 0.5:
                first_50 = alpha
            if first_90 is None and rate >= 0.9:
                first_90 = alpha
            if first_100 is None and rate >= 1.0:
                first_100 = alpha
        lines.append(f"- 首次 ≥50% 收敛: alpha={first_50}")
        lines.append(f"- 首次 ≥90% 收敛: alpha={first_90}")
        lines.append(f"- 首次 100% 收敛: alpha={first_100}")
        # sym_var_final 跨 alpha 变化
        sv_means = []
        for alpha in ALPHAS:
            sv = np.array([results[(alpha, env_label, s)]['sym_var_final'] for s in SEEDS])
            sv_means.append(sv.mean())
        # 找最大降幅
        drops = [(ALPHAS[i+1] - ALPHAS[i], sv_means[i] - sv_means[i+1]) for i in range(len(ALPHAS)-1)]
        max_drop = max(drops, key=lambda x: x[1])
        lines.append(f"- sym_var_final 最大单步降幅: alpha {max_drop[0]:.1f} -> {max_drop[0]+0.1:.1f}, 降幅 {max_drop[1]:.4f}")
        lines.append("")

    # 2. attractor 稳定性
    lines.append("### 两个 Symbolic attractor 是否稳定存在？\n")
    for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        lines.append(f"**env_init={env_init}:**\n")
        for alpha in ALPHAS:
            counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
            for s in SEEDS:
                counts[results[(alpha, env_label, s)]['attractor']] += 1
            lines.append(f"- alpha={alpha:.1f}: S=0: {counts['S=0']}, S=8: {counts['S=8']}, other_conv: {counts['other_conv']}, no_conv: {counts['no_conv']}")
        lines.append("")

    # 3. 与端点对照
    lines.append("### 与 factorial 主实验端点对照\n")
    lines.append("| 条件 | 本实验对应 | factorial 主实验 | 对照 |")
    lines.append("|------|-----------|------------------|------|")
    # alpha=0, [5,2,6] 对照 factorial isolated+C_update_off+[5,2,6]
    a0_526_conv = sum(1 for s in SEEDS if results[(0.0, '5_2_6', s)]['converged'])
    a0_526_sv = np.mean([results[(0.0, '5_2_6', s)]['sym_var_final'] for s in SEEDS])
    lines.append(f"| alpha=0, [5,2,6] | conv {a0_526_conv}/20, sv={a0_526_sv:.4f} | isolated+C_off+[5,2,6]: conv 0/20, sv=4.7111 | {'✓ 一致' if a0_526_conv == 0 and abs(a0_526_sv - 4.7111) < 0.1 else '✗ 不一致'} |")
    # alpha=1, [5,2,6] 对照 factorial isolated+C_update_on+[5,2,6]
    a1_526_conv = sum(1 for s in SEEDS if results[(1.0, '5_2_6', s)]['converged'])
    a1_526_sv = np.mean([results[(1.0, '5_2_6', s)]['sym_var_final'] for s in SEEDS])
    lines.append(f"| alpha=1, [5,2,6] | conv {a1_526_conv}/20, sv={a1_526_sv:.4f} | isolated+C_on+[5,2,6]: conv 19/20, sv=0.0111 | {'✓ 一致' if a1_526_conv == 19 and abs(a1_526_sv - 0.0111) < 0.01 else '✗ 不一致'} |")

    # 4. 理论讨论
    lines.append("\n## 理论讨论\n")
    lines.append("（基于实际结果，不提前假设 phase transition）\n")
    # 判断"所有 alpha>0 是否都收敛"
    all_pos_conv = True
    for env_label in ENV_INIT_LABELS:
        for alpha in ALPHAS:
            if alpha == 0:
                continue
            n_conv = sum(1 for s in SEEDS if results[(alpha, env_label, s)]['converged'])
            if n_conv < len(SEEDS) * 0.5:
                all_pos_conv = False
                break
    if all_pos_conv:
        lines.append("- **所有 alpha>0 条件下都达到较高收敛率**：当前模型的关键不是临界强度，而是是否存在持续的他者耦合。即使 alpha 很小（如 0.1），只要持续把主体的 Symbolic preference 朝他者位置拉动，就能产生集体 Symbolic 收敛。")
    else:
        lines.append("- **存在 alpha 区间使收敛率显著变化**：可以讨论从个体漂移到共同 Symbolic order 的动力学转变，但这是计算观察而非理论证明。")

    lines.append("\n### alpha 的操作性解读\n")
    lines.append("- alpha 是\"他者对主体欲望/符号偏好的规定强度\"的操作性参数。")
    lines.append("- alpha 越高，主体越强地把他者的 Symbolic position 纳入自己的 preference；alpha 越低，主体越保留自身历史 preference 的惯性。")
    lines.append("- alpha=0 时主体完全不受他者影响，三 agent 各自按初始 C_S 漂移；alpha=1 时主体每步完全采用他者的当前 Symbolic position 作为 preference。")
    lines.append("- 本实验不把 alpha 说成拉康理论中的真实参数，结果不证明大他者或欲望理论。\n")

    lines.append("## 声明\n")
    lines.append("1. 本实验固定 isolated 环境，仅扫描 alpha 和 env_init。")
    lines.append("2. 不修改 deep_experiments_v2.py；复用 factorial_env_symbolic.py 的模型函数。")
    lines.append("3. 不引入 identification/LLM/topology 等新概念。")
    lines.append("4. 结果不构成拉康理论的证明，是 toy model 上的计算观察。")
    lines.append("5. 基于 20 seeds 的聚合统计，非单 seed 轨迹。")
    lines.append("6. 报告不提前假设 phase transition，只报告实际结果。\n")

    lines.append("## 文件清单\n")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| symbolic_coupling_sweep.py | 本实验脚本 |")
    lines.append("| coupling_sweep_summary.csv | 每 alpha 每 env_init 汇总 |")
    lines.append("| coupling_sweep_timeseries.csv | 每 alpha 每 env_init 每 seed 每 step |")
    lines.append("| plot_coupling_sweep.png | 6 子图：收敛率/sym_var/real_imag_var/attractor比例/sym_var_2nd_half |")
    lines.append("| coupling_sweep_report.md | 本报告 |")

    with open(out_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(lines))


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("SYMBOLIC COUPLING STRENGTH SWEEP")
    print(f"isolated env, {len(ALPHAS)} alphas × {len(ENV_INIT_SETS)} env_inits × {len(SEEDS)} seeds")
    print("=" * 70)

    cs = f.A_MAT.sum(axis=0)
    assert np.allclose(cs, 1.0), "A matrix not normalized"
    print(f"[Verify] A matrix normalized: {np.allclose(cs, 1.0)}")

    results = {}
    total = len(ALPHAS) * len(ENV_INIT_SETS) * len(SEEDS)
    done = 0
    for alpha in ALPHAS:
        for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
            for seed in SEEDS:
                trajs = run_with_alpha(env_init, alpha, seed)
                m = compute_metrics(trajs)
                m['_trajs'] = trajs
                results[(alpha, env_label, seed)] = m
                done += 1
            # 进度
            n_conv = sum(1 for s in SEEDS if results[(alpha, env_label, s)]['converged'])
            sv = np.array([results[(alpha, env_label, s)]['sym_var_final'] for s in SEEDS])
            print(f"  alpha={alpha:.1f} env={env_init}: conv {n_conv}/20, sym_var_final={sv.mean():.4f}±{sv.std():.4f}  ({done}/{total})")

    # 汇总
    summary = []
    for alpha in ALPHAS:
        for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
            subset = [results[(alpha, env_label, s)] for s in SEEDS]
            n_conv = sum(1 for r in subset if r['converged'])
            sv_final = np.array([r['sym_var_final'] for r in subset])
            sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
            rv_final = np.array([r['real_var_final'] for r in subset])
            iv_final = np.array([r['imag_var_final'] for r in subset])
            # attractor 比例
            counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
            for r in subset:
                counts[r['attractor']] += 1
            n = len(SEEDS)
            summary.append([
                f"{alpha:.1f}", env_label, n, n_conv, f"{n_conv / n:.2f}",
                f"{sv_final.mean():.6f}", f"{sv_final.std():.6f}",
                f"{sv_sh.mean():.6f}", f"{sv_sh.std():.6f}",
                f"{rv_final.mean():.6f}", f"{rv_final.std():.6f}",
                f"{iv_final.mean():.6f}", f"{iv_final.std():.6f}",
                f"{counts['S=0'] / n:.2f}", f"{counts['S=8'] / n:.2f}",
                f"{counts['other_conv'] / n:.2f}", f"{counts['no_conv'] / n:.2f}",
            ])

    write_summary_csv(summary, os.path.join(OUT_DIR, 'coupling_sweep_summary.csv'))
    write_timeseries_csv(results, os.path.join(OUT_DIR, 'coupling_sweep_timeseries.csv'))
    print("\n[Saved] coupling_sweep_summary.csv, coupling_sweep_timeseries.csv")

    plot_sweep(results, summary, os.path.join(OUT_DIR, 'plot_coupling_sweep.png'))
    print("[Saved] plot_coupling_sweep.png")

    write_report(results, summary, os.path.join(OUT_DIR, 'coupling_sweep_report.md'))
    print("[Saved] coupling_sweep_report.md")

    # 打印汇总
    print("\n" + "=" * 70)
    print("SUMMARY: convergence rate & sym_var_final (mean)")
    print("=" * 70)
    print(f"{'alpha':<7}{'[5,2,6] conv / sv':<28}{'[8,8,8] conv / sv':<28}")
    print("-" * 63)
    for alpha in ALPHAS:
        row_526 = [r for r in summary if r[0] == f"{alpha:.1f}" and r[1] == '5_2_6'][0]
        row_888 = [r for r in summary if r[0] == f"{alpha:.1f}" and r[1] == '8_8_8'][0]
        print(f"{alpha:<7.1f}{row_526[3]}/20, sv={row_526[5]:<14}    {row_888[3]}/20, sv={row_888[5]}")

    # 端点 sanity check
    print("\n" + "=" * 70)
    print("SANITY CHECK (端点对照 factorial 主实验)")
    print("=" * 70)
    a0 = sum(1 for s in SEEDS if results[(0.0, '5_2_6', s)]['converged'])
    a0_sv = np.mean([results[(0.0, '5_2_6', s)]['sym_var_final'] for s in SEEDS])
    a1 = sum(1 for s in SEEDS if results[(1.0, '5_2_6', s)]['converged'])
    a1_sv = np.mean([results[(1.0, '5_2_6', s)]['sym_var_final'] for s in SEEDS])
    print(f"alpha=0, [5,2,6]: conv {a0}/20, sv={a0_sv:.4f}  (factorial C_off: 0/20, 4.7111)")
    print(f"alpha=1, [5,2,6]: conv {a1}/20, sv={a1_sv:.4f}  (factorial C_on:  19/20, 0.0111)")

    print("\nDone.")


if __name__ == '__main__':
    main()
