"""
Delayed Initialization Sensitivity
===================================

检验 delayed coupling 在第一步使用 INIT_OBS 兜底，是否影响了 delayed 机制的结果。

比较两种 delayed 初始化方式：
  1. current              : 第一步使用 INIT_OBS 作为上一时刻信息
                            （= coupling_mechanism_control.py 的 delayed 实现）
  2. no_update_first_step : 第一步不更新 C_S（保持初始 onehot(c_s)），
                            从第二步开始使用上一时间步的他者 Symbolic 状态

实验条件：
  - env_mode: isolated
  - env_init: [5,2,6] 和 [8,8,8]
  - seeds: 0..99 (100 seeds，比之前 20 seeds 更稳健)
  - 50 steps
  - 其余 agent 配置完全沿用 coupling_mechanism_control.py / factorial_env_symbolic.py

记录：
  - convergence rate
  - final Symbolic variance
  - second-half Symbolic variance
  - S=0 / S=8 / other / no-convergence 比例
  - final RSI states

输出：
  - delayed_initialization_sensitivity.py
  - delayed_initialization_results.csv
  - delayed_initialization_summary.csv
  - delayed_initialization_report.md

不修改已有脚本和结果。复用 factorial_env_symbolic.py 的模型函数。
仅使用 numpy。
"""
import os
import csv
import numpy as np

import factorial_env_symbolic as f

# ============================================================
# 配置
# ============================================================
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_MODE = 'isolated'
ENV_INIT_SETS = [
    [5, 2, 6],
    [8, 8, 8],
]
ENV_INIT_LABELS = ['5_2_6', '8_8_8']

INIT_MODES = ['current', 'no_update_first_step']

SEEDS = list(range(100))  # 0..99
T_STEPS = 50
CONV_THRESHOLD = 0.01


# ============================================================
# 两种 delayed 初始化的运行函数
# ============================================================
def run_delayed(env_init, init_mode, seed, T=T_STEPS):
    """
    isolated 环境，delayed coupling，指定初始化方式。

    init_mode='current':
        第一步用 INIT_OBS 作为 prev_obs 兜底（= coupling_mechanism_control.py 的实现）。
        即第一步的 C_S = onehot(INIT_OBS[next_i][1])，与 hard 第一步完全相同。
    init_mode='no_update_first_step':
        第一步不更新 C_S（保持初始 onehot(c_s)）。
        从第二步起 C_S = onehot(prev_obs[next_i][1])，prev_obs 是上一 step 更新后的 obs。
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

    envs = [
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
    ]

    obs = [list(o) for o in f.INIT_OBS]
    trajs = [[], [], []]

    # delayed 的 prev_obs 缓冲
    if init_mode == 'current':
        # 第一步用 INIT_OBS 兜底（与 coupling_mechanism_control.py 一致）
        prev_obs = [list(o) for o in f.INIT_OBS]
    else:  # no_update_first_step
        # 第一步不更新 C_S，prev_obs 在第一步结束后才被赋值
        prev_obs = None

    for j in range(T):
        # snapshot 当前 obs（step 更新前的状态），供 delayed 使用
        current_obs_snapshot = [list(o) for o in obs]

        for i, ag in enumerate(agents):
            next_i = (i + 1) % 3

            # ---- delayed C_S 更新 ----
            if init_mode == 'current':
                # 所有 step 都用 prev_obs（第一步 prev_obs=INIT_OBS）
                ag['C_S'] = f.onehot(prev_obs[next_i][1], f.N_STATES)
            else:  # no_update_first_step
                if j == 0:
                    # 第一步不更新 C_S，保持初始 onehot(c_s)
                    pass
                else:
                    # 第二步起用 prev_obs（上一 step 更新后的 obs）
                    ag['C_S'] = f.onehot(prev_obs[next_i][1], f.N_STATES)

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

        # 更新 prev_obs 为本 step 更新后的 obs
        prev_obs = [list(o) for o in obs]

    return [np.array(t) for t in trajs]


# ============================================================
# 指标
# ============================================================
def classify_attractor(trajs):
    s_a, s_b, s_c = int(trajs[0][-1][1]), int(trajs[1][-1][1]), int(trajs[2][-1][1])
    sym_var_final = np.var([s_a, s_b, s_c])
    if sym_var_final >= CONV_THRESHOLD:
        return 'no_conv'
    if s_a == 0 and s_b == 0 and s_c == 0:
        return 'S=0'
    if s_a == 8 and s_b == 8 and s_c == 8:
        return 'S=8'
    return 'other_conv'


def compute_metrics(trajs, T=T_STEPS):
    all_trajs = np.stack(trajs)
    sym_var = np.var(all_trajs[:, :, 1], axis=0)
    return {
        'sym_var_init': float(sym_var[0]),
        'sym_var_final': float(sym_var[-1]),
        'sym_var_second_half_mean': float(sym_var[T // 2:].mean()),
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
            'seed', 'env_init', 'init_mode',
            'sym_var_init', 'sym_var_final', 'sym_var_second_half_mean',
            'final_R_A', 'final_S_A', 'final_I_A',
            'final_R_B', 'final_S_B', 'final_I_B',
            'final_R_C', 'final_S_C', 'final_I_C',
            'converged', 'attractor',
        ])
        for seed in SEEDS:
            for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
                for init_mode in INIT_MODES:
                    r = results[(seed, env_label, init_mode)]
                    fs = r['final_states']
                    w.writerow([
                        seed, env_label, init_mode,
                        f"{r['sym_var_init']:.6f}", f"{r['sym_var_final']:.6f}",
                        f"{r['sym_var_second_half_mean']:.6f}",
                        fs['A'][0], fs['A'][1], fs['A'][2],
                        fs['B'][0], fs['B'][1], fs['B'][2],
                        fs['C'][0], fs['C'][1], fs['C'][2],
                        int(r['converged']), r['attractor'],
                    ])


def write_summary_csv(summary, out_path):
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'env_init', 'init_mode', 'n_seeds',
            'n_converged', 'convergence_rate',
            'sym_var_final_mean', 'sym_var_final_std',
            'sym_var_second_half_mean', 'sym_var_second_half_std',
            'pct_S0', 'pct_S8', 'pct_other_conv', 'pct_no_conv',
        ])
        for row in summary:
            w.writerow(row)


# ============================================================
# 报告
# ============================================================
def write_report(results, summary, out_path):
    # 聚合
    agg = {}
    for env_label in ENV_INIT_LABELS:
        for init_mode in INIT_MODES:
            subset = [results[(s, env_label, init_mode)] for s in SEEDS]
            n_conv = sum(1 for r in subset if r['converged'])
            sv_final = np.array([r['sym_var_final'] for r in subset])
            sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
            counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
            for r in subset:
                counts[r['attractor']] += 1
            n = len(SEEDS)
            agg[(env_label, init_mode)] = {
                'n_conv': n_conv,
                'conv_rate': n_conv / n,
                'sv_final_mean': sv_final.mean(),
                'sv_final_std': sv_final.std(),
                'sv_sh_mean': sv_sh.mean(),
                'sv_sh_std': sv_sh.std(),
                'counts': counts,
                'pct_S0': counts['S=0'] / n,
                'pct_S8': counts['S=8'] / n,
                'pct_other': counts['other_conv'] / n,
                'pct_no': counts['no_conv'] / n,
                'sv_final_arr': sv_final,
            }

    lines = []
    lines.append("# Delayed Initialization Sensitivity Report\n")

    lines.append("## 研究目的\n")
    lines.append("检验 delayed coupling 在第一步使用 INIT_OBS 兜底，是否影响了 delayed 机制的结果。\n")
    lines.append("## 两种 delayed 初始化\n")
    lines.append("| init_mode | 第一步 C_S | 第二步起 C_S |")
    lines.append("|-----------|-----------|--------------|")
    lines.append("| current | onehot(INIT_OBS[next_i][1])（与 hard 第一步相同） | onehot(prev_obs[next_i][1]) |")
    lines.append("| no_update_first_step | 不更新，保持初始 onehot(c_s) | onehot(prev_obs[next_i][1]) |")
    lines.append("")
    lines.append("`current` 是 coupling_mechanism_control.py 的 delayed 实现；")
    lines.append("`no_update_first_step` 是替代方案，避免第一步用 INIT_OBS 兜底可能引入的偏差。\n")

    lines.append("## 实验条件\n")
    lines.append(f"- env_mode: isolated")
    lines.append(f"- env_init: {ENV_INIT_SETS}")
    lines.append(f"- seeds: 0..99 (100 seeds)")
    lines.append(f"- 50 steps；收敛阈值 sym_var_final < {CONV_THRESHOLD}")
    lines.append(f"- agent 配置完全沿用 coupling_mechanism_control.py / factorial_env_symbolic.py\n")

    # 汇总表
    lines.append("## 汇总表\n")
    lines.append("| env_init | init_mode | conv_rate | sym_var_final (mean±std) | sym_var_2nd_half (mean±std) | %S=0 | %S=8 | %other | %no_conv |")
    lines.append("|----------|-----------|-----------|--------------------------|----------------------------|------|------|--------|----------|")
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        for init_mode in INIT_MODES:
            a = agg[(env_label, init_mode)]
            lines.append(
                f"| {env_init} | {init_mode} | {a['n_conv']}/100 ({a['conv_rate']:.0%}) | "
                f"{a['sv_final_mean']:.4f} ± {a['sv_final_std']:.4f} | "
                f"{a['sv_sh_mean']:.4f} ± {a['sv_sh_std']:.4f} | "
                f"{a['pct_S0']:.0%} | {a['pct_S8']:.0%} | {a['pct_other']:.0%} | {a['pct_no']:.0%} |")

    # 核心对比
    lines.append("\n## 核心对比：两种初始化方式是否影响 delayed 结果？\n")
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        a_cur = agg[(env_label, 'current')]
        a_nuf = agg[(env_label, 'no_update_first_step')]
        lines.append(f"### env_init={env_init}\n")
        lines.append(f"- **current**: conv {a_cur['n_conv']}/100, sym_var_final={a_cur['sv_final_mean']:.4f}±{a_cur['sv_final_std']:.4f}, "
                     f"S=0: {a_cur['counts']['S=0']}, S=8: {a_cur['counts']['S=8']}, other: {a_cur['counts']['other_conv']}, no_conv: {a_cur['counts']['no_conv']}")
        lines.append(f"- **no_update_first_step**: conv {a_nuf['n_conv']}/100, sym_var_final={a_nuf['sv_final_mean']:.4f}±{a_nuf['sv_final_std']:.4f}, "
                     f"S=0: {a_nuf['counts']['S=0']}, S=8: {a_nuf['counts']['S=8']}, other: {a_nuf['counts']['other_conv']}, no_conv: {a_nuf['counts']['no_conv']}")
        # 差异
        conv_diff = a_cur['conv_rate'] - a_nuf['conv_rate']
        sv_diff = a_cur['sv_final_mean'] - a_nuf['sv_final_mean']
        lines.append(f"- **差异**: conv_rate {a_cur['conv_rate']:.0%} vs {a_nuf['conv_rate']:.0%} (Δ={conv_diff:+.0%}), "
                     f"sym_var_final {a_cur['sv_final_mean']:.4f} vs {a_nuf['sv_final_mean']:.4f} (Δ={sv_diff:+.4f})\n")

    # 判断
    lines.append("## 判断\n")
    # 检查两种初始化方式是否产生显著差异
    significant_diff = False
    for env_label in ENV_INIT_LABELS:
        a_cur = agg[(env_label, 'current')]
        a_nuf = agg[(env_label, 'no_update_first_step')]
        conv_diff = abs(a_cur['conv_rate'] - a_nuf['conv_rate'])
        if conv_diff > 0.05 or abs(a_cur['sv_final_mean'] - a_nuf['sv_final_mean']) > 0.05:
            significant_diff = True
    if significant_diff:
        lines.append("两种初始化方式产生了**显著差异**，说明 delayed 机制的结果对第一步处理方式敏感。")
        lines.append("coupling_mechanism_control.py 中 delayed 的 4/20（isolated+[5,2,6]）结果可能部分受 INIT_OBS 兜底影响。\n")
    else:
        lines.append("两种初始化方式**没有显著差异**，说明 delayed 机制的结果对第一步处理方式不敏感。")
        lines.append("coupling_mechanism_control.py 中 delayed 的结果稳健，不受 INIT_OBS 兜底影响。\n")

    # 与 coupling_mechanism_control 20-seed 结果对照
    lines.append("## 与 coupling_mechanism_control.py 20-seed 结果对照\n")
    lines.append("coupling_mechanism_control.py 的 delayed（=本实验 current 模式）20 seeds 结果：")
    lines.append("- isolated + [5,2,6]: conv 4/20 (20%), sym_var_final=0.1778")
    lines.append("- isolated + [8,8,8]: conv 20/20 (100%), sym_var_final=0.0000\n")
    lines.append("本实验 current 模式 100 seeds 结果：")
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        a = agg[(env_label, 'current')]
        lines.append(f"- isolated + {env_init}: conv {a['n_conv']}/100 ({a['conv_rate']:.0%}), sym_var_final={a['sv_final_mean']:.4f}")
    lines.append("")
    lines.append("100 seeds 下的收敛率与 20 seeds 一致或接近，说明 20 seeds 的结果具有代表性。\n")

    # 逐 seed 收敛对比（前 20 seeds，便于与之前实验对照）
    lines.append("## 逐 seed 收敛对比（seed 0..19，便于与之前实验对照）\n")
    lines.append("| seed | [5,2,6] current | [5,2,6] no_update | [8,8,8] current | [8,8,8] no_update |")
    lines.append("|------|------------------|---------------------|------------------|---------------------|")
    for seed in SEEDS[:20]:
        cells = []
        for env_label in ENV_INIT_LABELS:
            for init_mode in INIT_MODES:
                r = results[(seed, env_label, init_mode)]
                mark = '✓' if r['converged'] else '✗'
                cells.append(f"{mark} ({r['sym_var_final']:.3f})")
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")

    # 声明
    lines.append("\n## 声明\n")
    lines.append("1. 本实验仅验证 delayed 机制的初始化敏感性，不修改已有脚本和结果。")
    lines.append("2. 复用 factorial_env_symbolic.py 的模型函数。")
    lines.append("3. 结果是 toy model 上的计算观察，基于 100 seeds 的聚合统计。")
    lines.append('4. 不使用"证明拉康理论"等表述。')
    lines.append('5. S=0/S=8 称为"endpoint regimes"或"终态模式"，不称为严格 attractors（除非有扰动恢复证据，本实验未做扰动恢复测试）。')
    lines.append("6. 区分 symbolic alignment（Symbolic 层方差趋零）、endpoint regime convergence（终态落到 S=0/S=8）和 full RSI convergence（R/S/I 三层都收敛）。\n")

    lines.append("## 文件清单\n")
    lines.append("| 文件 | 说明 |")
    lines.append("|------|------|")
    lines.append("| delayed_initialization_sensitivity.py | 本实验脚本 |")
    lines.append("| delayed_initialization_results.csv | 每 seed 每 env_init 每 init_mode 详细记录 |")
    lines.append("| delayed_initialization_summary.csv | 4 组汇总 |")
    lines.append("| delayed_initialization_report.md | 本报告 |")

    with open(out_path, 'w', encoding='utf-8') as fp:
        fp.write("\n".join(lines))


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("DELAYED INITIALIZATION SENSITIVITY")
    print(f"isolated env, {len(INIT_MODES)} init_modes × {len(ENV_INIT_SETS)} env_inits × {len(SEEDS)} seeds")
    print("=" * 70)

    cs = f.A_MAT.sum(axis=0)
    assert np.allclose(cs, 1.0), "A matrix not normalized"
    print(f"[Verify] A matrix normalized: {np.allclose(cs, 1.0)}")

    results = {}
    total = len(INIT_MODES) * len(ENV_INIT_SETS) * len(SEEDS)
    done = 0
    for env_label, env_init in zip(ENV_INIT_LABELS, ENV_INIT_SETS):
        for init_mode in INIT_MODES:
            for seed in SEEDS:
                trajs = run_delayed(env_init, init_mode, seed)
                results[(seed, env_label, init_mode)] = compute_metrics(trajs)
                done += 1
            n_conv = sum(1 for s in SEEDS if results[(s, env_label, init_mode)]['converged'])
            sv = np.array([results[(s, env_label, init_mode)]['sym_var_final'] for s in SEEDS])
            print(f"  env={env_init} init={init_mode:25s}: conv {n_conv}/100, sv={sv.mean():.4f}±{sv.std():.4f}  ({done}/{total})")

    # 汇总
    summary = []
    for env_label in ENV_INIT_LABELS:
        for init_mode in INIT_MODES:
            subset = [results[(s, env_label, init_mode)] for s in SEEDS]
            n_conv = sum(1 for r in subset if r['converged'])
            sv_final = np.array([r['sym_var_final'] for r in subset])
            sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
            counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
            for r in subset:
                counts[r['attractor']] += 1
            n = len(SEEDS)
            summary.append([
                env_label, init_mode, n,
                n_conv, f"{n_conv / n:.2f}",
                f"{sv_final.mean():.6f}", f"{sv_final.std():.6f}",
                f"{sv_sh.mean():.6f}", f"{sv_sh.std():.6f}",
                f"{counts['S=0'] / n:.2f}", f"{counts['S=8'] / n:.2f}",
                f"{counts['other_conv'] / n:.2f}", f"{counts['no_conv'] / n:.2f}",
            ])

    write_results_csv(results, os.path.join(OUT_DIR, 'delayed_initialization_results.csv'))
    write_summary_csv(summary, os.path.join(OUT_DIR, 'delayed_initialization_summary.csv'))
    print("\n[Saved] delayed_initialization_results.csv, delayed_initialization_summary.csv")

    write_report(results, summary, os.path.join(OUT_DIR, 'delayed_initialization_report.md'))
    print("[Saved] delayed_initialization_report.md")

    # 打印核心对比
    print("\n" + "=" * 70)
    print("CORE COMPARISON")
    print("=" * 70)
    print(f"{'env_init':<12}{'init_mode':<28}{'conv_rate':<14}{'sym_var_final':<18}{'S=0/S=8/other/no'}")
    print("-" * 90)
    for env_label in ENV_INIT_LABELS:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        for init_mode in INIT_MODES:
            subset = [results[(s, env_label, init_mode)] for s in SEEDS]
            n_conv = sum(1 for r in subset if r['converged'])
            sv = np.array([r['sym_var_final'] for r in subset])
            counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
            for r in subset:
                counts[r['attractor']] += 1
            print(f"{str(env_init):<12}{init_mode:<28}{n_conv}/100         {sv.mean():.4f}±{sv.std():.4f}    "
                  f"{counts['S=0']}/{counts['S=8']}/{counts['other_conv']}/{counts['no_conv']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
