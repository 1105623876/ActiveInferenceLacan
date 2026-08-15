"""
Environment Initial-State Sensitivity Experiment
================================================

固定 isolated + C_update_on 条件（与 v2 Exp4 / factorial 主实验最强收敛条件一致），
固定 agent 配置（C_R/C_S/C_I/D_R/D_S/D_I/weights/policy_len/T）和 seed 集合（0..19），
只改变环境初始状态 [env_r0, env_s0, env_i0]，看 Symbolic 收敛是否对初态敏感、
以及是否总是落到同一个终态。

测试的环境初始状态：
  [5, 2, 6]   —— factorial 主实验用的初态（中性）
  [0, 0, 0]   —— 全低端
  [8, 8, 8]   —— 全高端
  [0, 2, 8]   —— 跨度大、不对称
  [8, 6, 0]   —— 跨度大、反向不对称

每组 20 seeds，50 steps，记录：
  - Symbolic 收敛率（sym_var_final < 0.01）
  - 最终 Symbolic 方差（mean ± std）
  - 三个 agent 的最终状态
  - 是否总是落到同一个终态（unique 终态数量 + 具体终态）

不修改 deep_experiments_v2.py；复用 factorial_env_symbolic.py 的模型函数
（该模块顶层只有函数/常量定义，import 不会触发实验执行）。
仅使用 numpy + matplotlib。
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import factorial_env_symbolic as f

# ============================================================
# 配置
# ============================================================
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# 固定 isolated + C_update_on
ENV_MODE = 'isolated'
C_UPDATE = True

# 环境初始状态扫描集
ENV_INIT_SETS = [
    [5, 2, 6],
    [0, 0, 0],
    [8, 8, 8],
    [0, 2, 8],
    [8, 6, 0],
]
ENV_INIT_LABELS = ['5_2_6', '0_0_0', '8_8_8', '0_2_8', '8_6_0']

SEEDS = list(range(20))
T_STEPS = 50
CONV_THRESHOLD = 0.01  # 与 v2 Exp4 / factorial 主实验一致


# ============================================================
# 单条件运行（复用 factorial_env_symbolic.run_condition 的逻辑，
#            但允许自定义环境初始状态）
# ============================================================
def run_with_env_init(env_init, seed, T=T_STEPS):
    """
    isolated + C_update_on，指定环境初始状态。
    复用 f.AGENT_CONFIGS / f.WEIGHTS / f.INIT_OBS / f.A_MAT / f.B_MAT 等。
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

    # isolated：每个 agent 独立 env 副本，但初始状态统一为 env_init
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
            # C_update_on
            ag['C_S'] = f.change_C(obs[next_i][1])

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


def compute_metrics(trajs, T=T_STEPS):
    all_trajs = np.stack(trajs)  # (3, T, 3)
    sym_var = np.var(all_trajs[:, :, 1], axis=0)
    return {
        'sym_var_init': float(sym_var[0]),
        'sym_var_final': float(sym_var[-1]),
        'sym_var_second_half_mean': float(sym_var[T // 2:].mean()),
        'sym_var_curve': sym_var,
        'final_states': {
            'A': tuple(int(x) for x in trajs[0][-1]),
            'B': tuple(int(x) for x in trajs[1][-1]),
            'C': tuple(int(x) for x in trajs[2][-1]),
        },
        'converged': bool(sym_var[-1] < CONV_THRESHOLD),
    }


# ============================================================
# CSV
# ============================================================
def write_results_csv(results, out_path):
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'env_init_label', 'env_init', 'seed',
            'sym_var_init', 'sym_var_final', 'sym_var_second_half_mean',
            'final_R_A', 'final_S_A', 'final_I_A',
            'final_R_B', 'final_S_B', 'final_I_B',
            'final_R_C', 'final_S_C', 'final_I_C',
            'converged',
        ])
        for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
            for seed in SEEDS:
                r = results[(env_label, seed)]
                fs = r['final_states']
                w.writerow([
                    env_label, f"{env_init[0]}_{env_init[1]}_{env_init[2]}", seed,
                    f"{r['sym_var_init']:.6f}", f"{r['sym_var_final']:.6f}",
                    f"{r['sym_var_second_half_mean']:.6f}",
                    fs['A'][0], fs['A'][1], fs['A'][2],
                    fs['B'][0], fs['B'][1], fs['B'][2],
                    fs['C'][0], fs['C'][1], fs['C'][2],
                    int(r['converged']),
                ])


def write_summary_csv(summary, out_path):
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'env_init_label', 'env_init', 'n_seeds', 'n_converged', 'convergence_rate',
            'sym_var_final_mean', 'sym_var_final_std',
            'sym_var_second_half_mean', 'sym_var_second_half_std',
            'n_unique_final_states', 'unique_final_states',
        ])
        for row in summary:
            w.writerow(row)


# ============================================================
# 绘图
# ============================================================
def plot_sensitivity(results, out_path, T=T_STEPS):
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ENV_INIT_SETS)))

    # 上图：每条件 20 seeds 的 sym_var 曲线（均值 + 单 seed）
    ax = axes[0]
    for idx, (env_label, env_init) in enumerate(zip(ENV_INIT_LABELS, ENV_INIT_SETS)):
        all_curves = []
        for seed in SEEDS:
            r = results[(env_label, seed)]
            all_curves.append(r['sym_var_curve'])
            ax.plot(r['sym_var_curve'], color=colors[idx], alpha=0.15, linewidth=0.7)
        mean_curve = np.mean(all_curves, axis=0)
        ax.plot(mean_curve, color=colors[idx], linewidth=2.5,
                label=f'env_init={env_init} (mean)')
    ax.axhline(CONV_THRESHOLD, color='red', linestyle=':', alpha=0.5,
               label=f'threshold={CONV_THRESHOLD}')
    ax.set_xlabel('Step')
    ax.set_ylabel('Symbolic observation variance')
    ax.set_title('Symbolic variance over time (isolated + C_update_on, 20 seeds per env_init)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.1, max(2.5, max(r['sym_var_init'] for r in [results[(l, 0)] for l in ENV_INIT_LABELS]) * 1.2))

    # 下图：收敛率柱状图
    ax2 = axes[1]
    conv_rates = []
    sv_finals = {label: [] for label in ENV_INIT_LABELS}
    for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        n_conv = sum(1 for s in SEEDS if results[(env_label, s)]['converged'])
        conv_rates.append(n_conv / len(SEEDS))
        for s in SEEDS:
            sv_finals[env_label].append(results[(env_label, s)]['sym_var_final'])

    x_pos = np.arange(len(ENV_INIT_SETS))
    bars = ax2.bar(x_pos, conv_rates, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{ei}\n({lbl})" for ei, lbl in zip(ENV_INIT_SETS, ENV_INIT_LABELS)])
    ax2.set_ylabel('Convergence rate (20 seeds)')
    ax2.set_title('Convergence rate by environment initial state')
    ax2.set_ylim(0, 1.15)
    for bar, v in zip(bars, conv_rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                 f'{v:.0%}', ha='center', fontsize=10)
    ax2.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Environment Initial-State Sensitivity\n'
        'isolated + C_update_on, 20 seeds, 50 steps per env_init',
        fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# 报告
# ============================================================
def write_report(results, summary, out_path):
    lines = []
    lines.append("# Environment Initial-State Sensitivity Report\n")
    lines.append("## 实验设置\n")
    lines.append("固定 **isolated + C_update_on** 条件（factorial 主实验中的最强收敛条件），")
    lines.append("固定 agent 配置（C_R/C_S/C_I/D_R/D_S/D_I/weights/policy_len/T）和 seed 集合（0..19），")
    lines.append("只改变环境初始状态 `[env_r0, env_s0, env_i0]`，观察：\n")
    lines.append("- Symbolic 收敛率是否对初态敏感；")
    lines.append("- 最终 Symbolic 方差；")
    lines.append("- 三个 agent 的最终状态；")
    lines.append("- 是否总是落到同一个终态。\n")
    lines.append("### 测试的环境初始状态\n")
    lines.append("| 标签 | env_init | 含义 |")
    lines.append("|------|----------|------|")
    for lbl, ei in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        lines.append(f"| {lbl} | {ei} | |")
    lines.append("\n### 公平性\n")
    lines.append("- agent 配置完全沿用 factorial 主实验（与 v2 `run_triadic_isolated` 一致）；")
    lines.append("- 每 seed 每 env_init 重新初始化所有环境、agent 参数和 observation；")
    lines.append("- 收敛阈值 `sym_var_final < 0.01`（与 v2 Exp4 / factorial 主实验一致）；")
    lines.append("- 20 seeds (0..19)，50 steps。\n")

    lines.append("## 汇总表\n")
    lines.append("| env_init | 收敛率 | sym_var_final (mean±std) | sym_var_2nd_half (mean±std) | unique 终态数 | 终态分布 |")
    lines.append("|----------|--------|--------------------------|----------------------------|---------------|----------|")
    for row in summary:
        label = row[0]
        env_init_str = row[1]
        n_seeds = row[2]
        n_conv = row[3]
        conv_rate = row[4]
        sv_mean = row[5]
        sv_std = row[6]
        sv_sh_mean = row[7]
        sv_sh_std = row[8]
        n_unique = row[9]
        unique_states = row[10]
        lines.append(f"| {env_init_str} | {n_conv}/{n_seeds} ({conv_rate}) | {sv_mean} ± {sv_std} | {sv_sh_mean} ± {sv_sh_std} | {n_unique} | {unique_states} |")

    lines.append("\n## 逐 seed 终态表\n")
    lines.append("| seed | " + " | ".join(ENV_INIT_LABELS) + " |")
    lines.append("|------|" + "|".join(["---"] * len(ENV_INIT_LABELS)) + "|")
    for seed in SEEDS:
        cells = []
        for env_label in ENV_INIT_LABELS:
            r = results[(env_label, seed)]
            fs = r['final_states']
            mark = '✓' if r['converged'] else '✗'
            cells.append(f"{mark} A={fs['A']} B={fs['B']} C={fs['C']}")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")

    lines.append("\n## 主要观察\n")
    lines.append("（由脚本生成后人工核对）\n")

    # 自动判断"是否总是落到同一个终态"
    # 收集每个 env_init 下所有 seed 的 (A,B,C) 三元组
    final_state_sets = {}
    for env_label in ENV_INIT_LABELS:
        states = []
        for s in SEEDS:
            fs = results[(env_label, s)]['final_states']
            states.append((fs['A'], fs['B'], fs['C']))
        unique = list(set(states))
        final_state_sets[env_label] = unique

    lines.append("### 是否总是落到同一个终态？\n")
    for env_label in ENV_INIT_LABELS:
        unique = final_state_sets[env_label]
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        lines.append(f"- **env_init={env_init} ({env_label})**: {len(unique)} 个 unique 终态")
        for st in unique:
            count = sum(1 for s in SEEDS
                        if (results[(env_label, s)]['final_states']['A'],
                            results[(env_label, s)]['final_states']['B'],
                            results[(env_label, s)]['final_states']['C']) == st)
            lines.append(f"  - {st[0]} / {st[1]} / {st[2]}  出现 {count}/{len(SEEDS)} seeds")

    lines.append("\n## 声明\n")
    lines.append("1. 本实验固定 isolated + C_update_on，仅扫描环境初始状态。")
    lines.append("2. 不修改 deep_experiments_v2.py；复用 factorial_env_symbolic.py 的模型函数。")
    lines.append("3. 结果不构成拉康理论的证明，是 toy model 上的计算观察。")
    lines.append("4. 基于 20 seeds 的聚合统计，非单 seed 轨迹。\n")

    lines.append("## 文件清单\n")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| env_init_sensitivity.py | 本实验脚本 |")
    lines.append("| env_init_results.csv | 每 seed 每 env_init 详细记录 |")
    lines.append("| env_init_summary.csv | 5 组 env_init 汇总 |")
    lines.append("| plot_env_init_sensitivity.png | sym_var 曲线 + 收敛率图 |")
    lines.append("| env_init_report.md | 本报告 |")

    with open(out_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(lines))


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("ENV INIT SENSITIVITY (isolated + C_update_on, 5 env_inits × 20 seeds)")
    print("=" * 70)

    cs = f.A_MAT.sum(axis=0)
    assert np.allclose(cs, 1.0), "A matrix not normalized"
    print(f"[Verify] A matrix normalized: {np.allclose(cs, 1.0)}")

    results = {}
    for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        for seed in SEEDS:
            trajs = run_with_env_init(env_init, seed)
            results[(env_label, seed)] = compute_metrics(trajs)
        # 进度
        n_conv = sum(1 for s in SEEDS if results[(env_label, s)]['converged'])
        sv = np.array([results[(env_label, s)]['sym_var_final'] for s in SEEDS])
        print(f"  env_init={env_init} ({env_label}): conv {n_conv}/20, "
              f"sym_var_final={sv.mean():.4f}±{sv.std():.4f}")

    # 汇总
    summary = []
    for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        subset = [results[(env_label, s)] for s in SEEDS]
        n_conv = sum(1 for r in subset if r['converged'])
        sv_final = np.array([r['sym_var_final'] for r in subset])
        sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
        # unique 终态
        states = []
        for r in subset:
            fs = r['final_states']
            states.append((fs['A'], fs['B'], fs['C']))
        unique = list(set(states))
        unique_str = " ; ".join([f"{u[0]}/{u[1]}/{u[2]}" for u in unique])
        summary.append([
            env_label,
            f"{env_init[0]}_{env_init[1]}_{env_init[2]}",
            len(SEEDS), n_conv, f"{n_conv / len(SEEDS):.2f}",
            f"{sv_final.mean():.6f}", f"{sv_final.std():.6f}",
            f"{sv_sh.mean():.6f}", f"{sv_sh.std():.6f}",
            len(unique), unique_str,
        ])

    write_results_csv(results, os.path.join(OUT_DIR, 'env_init_results.csv'))
    write_summary_csv(summary, os.path.join(OUT_DIR, 'env_init_summary.csv'))
    print("\n[Saved] env_init_results.csv, env_init_summary.csv")

    plot_sensitivity(results, os.path.join(OUT_DIR, 'plot_env_init_sensitivity.png'))
    print("[Saved] plot_env_init_sensitivity.png")

    write_report(results, summary, os.path.join(OUT_DIR, 'env_init_report.md'))
    print("[Saved] env_init_report.md")

    # 打印汇总
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'env_init':<14}{'conv_rate':<12}{'sym_var_final (mean±std)':<28}{'unique_states':<15}")
    print("-" * 70)
    for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        subset = [results[(env_label, s)] for s in SEEDS]
        n_conv = sum(1 for r in subset if r['converged'])
        sv = np.array([r['sym_var_final'] for r in subset])
        states = set((r['final_states']['A'], r['final_states']['B'], r['final_states']['C']) for r in subset)
        print(f"{str(env_init):<14}{n_conv}/20        {sv.mean():.4f} ± {sv.std():.4f}            {len(states)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
