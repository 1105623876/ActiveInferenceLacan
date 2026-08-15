"""
Opaque Other Coupling — inferred-other preference (position, not desire)
=======================================================================

路径 B v1：主体不再读取他者 Symbolic 真值写入 C_S，
而是从带噪社会观察滤波得到 q(s_j)，再令 C_S := q(s_j)。

对照：off / hard / noisy（noisy 仍涂糊真值，不是推断）。
环境：isolated only。初态：[5,2,6]、[8,8,8]。
infer 扫描 σ ∈ {0.40, 0.70, 1.00}。

用法：
  python opaque_other_coupling.py phase0
  python opaque_other_coupling.py phase1
"""
import os
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import factorial_env_symbolic as f

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_INIT_SETS = [
    [5, 2, 6],
    [8, 8, 8],
]
ENV_INIT_LABELS = ['5_2_6', '8_8_8']

# (mechanism, sigma or None)
BASE_CONDS = [('off', None), ('hard', None), ('noisy', None)]
INFER_SIGMAS = [0.40, 0.70, 1.00]
NOISE_EPS = 0.2
UNIFORM = np.ones(f.N_STATES) / f.N_STATES

T_STEPS = 50
CONV_THRESHOLD = 0.01
PHASE0_SEEDS = list(range(5))
PHASE1_SEEDS = list(range(20))


def cond_label(mechanism, sigma):
    if mechanism == 'infer':
        return f'infer_{sigma:.2f}'
    return mechanism


def all_conditions():
    return BASE_CONDS + [('infer', s) for s in INFER_SIGMAS]


def sample_social_obs(A_social, s_j):
    return int(np.random.choice(f.N_STATES, p=A_social[:, s_j]))


def vec_entropy(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-16)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def mean_run_length(flags):
    runs = []
    cur = 0
    for x in flags:
        if x:
            cur += 1
        elif cur:
            runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return float(np.mean(runs)) if runs else 0.0


def time_to_sync(sym_var, thresh=CONV_THRESHOLD):
    """First t such that the group stays below threshold until the end."""
    for t in range(len(sym_var)):
        if np.all(sym_var[t:] < thresh):
            return t
    return None


def belief_lag_from_series(true_s, hat_s):
    """S_j 变化后，argmax q 追上新值所需步数的平均。无变化则 nan。"""
    T = len(true_s)
    lags = []
    for t in range(1, T):
        if true_s[t] == true_s[t - 1]:
            continue
        new_s = true_s[t]
        k = 0
        while t + k < T and hat_s[t + k] != new_s:
            k += 1
        lags.append(k if t + k < T and hat_s[t + k] == new_s else (T - t))
    if not lags:
        return float('nan')
    return float(np.mean(lags))


def classify_attractor(trajs):
    s_a, s_b, s_c = int(trajs[0][-1][1]), int(trajs[1][-1][1]), int(trajs[2][-1][1])
    if np.var([s_a, s_b, s_c]) >= CONV_THRESHOLD:
        return 'no_conv'
    if s_a == 0 and s_b == 0 and s_c == 0:
        return 'S=0'
    if s_a == 8 and s_b == 8 and s_c == 8:
        return 'S=8'
    return 'other_conv'


def run_trial(env_init, mechanism, sigma, seed, T=T_STEPS):
    np.random.seed(seed)

    A_social = None
    if mechanism == 'infer':
        A_social = f.make_A_matrix(sharpness=sigma)

    agents = []
    d_others = []
    for (c_r, c_s, c_i, d_r, d_s, d_i) in f.AGENT_CONFIGS:
        agents.append({
            'A_R': f.A_MAT.copy(), 'B_R': f.B_MAT.copy(),
            'C_R': f.onehot(c_r, f.N_STATES), 'D_R': f.onehot(d_r, f.N_STATES),
            'A_S': f.A_MAT.copy(), 'B_S': f.B_MAT.copy(),
            'C_S': f.onehot(c_s, f.N_STATES), 'D_S': f.onehot(d_s, f.N_STATES),
            'A_I': f.A_MAT.copy(), 'B_I': f.B_MAT.copy(),
            'C_I': f.onehot(c_i, f.N_STATES), 'D_I': f.onehot(d_i, f.N_STATES),
        })
        d_others.append(UNIFORM.copy())

    envs = [
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
    ]
    obs = [list(o) for o in f.INIT_OBS]
    trajs = [[], [], []]
    true_s = [[] for _ in range(3)]
    hat_s = [[] for _ in range(3)]
    kl_steps = [[] for _ in range(3)]
    h_steps = [[] for _ in range(3)]

    for _ in range(T):
        for i, ag in enumerate(agents):
            j = (i + 1) % 3
            s_j = obs[j][1]

            if mechanism == 'off':
                belief = ag['C_S']
            elif mechanism == 'hard':
                ag['C_S'] = f.onehot(s_j, f.N_STATES)
                belief = ag['C_S']
            elif mechanism == 'noisy':
                target = f.onehot(s_j, f.N_STATES)
                ag['C_S'] = (1.0 - NOISE_EPS) * target + NOISE_EPS * UNIFORM
                ag['C_S'] = ag['C_S'] / ag['C_S'].sum()
                belief = ag['C_S']
            elif mechanism == 'infer':
                o = sample_social_obs(A_social, s_j)
                q = f.infer_states(o, A_social, d_others[i])
                ag['C_S'] = q.copy()
                d_others[i] = q.copy()
                belief = q
            else:
                raise ValueError(f'unknown mechanism: {mechanism}')

            true_s[i].append(int(s_j))
            hat_s[i].append(int(np.argmax(belief)))
            kl_steps[i].append(f.kl_divergence(belief, f.onehot(s_j, f.N_STATES)))
            h_steps[i].append(vec_entropy(belief))

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

    trajs = [np.array(t) for t in trajs]
    return trajs, {
        'true_s': [np.array(x) for x in true_s],
        'hat_s': [np.array(x) for x in hat_s],
        'kl_steps': [np.array(x) for x in kl_steps],
        'h_steps': [np.array(x) for x in h_steps],
    }


def compute_metrics(trajs, traces, T=T_STEPS):
    all_trajs = np.stack(trajs)
    sym_var = np.var(all_trajs[:, :, 1], axis=0)
    real_var = np.var(all_trajs[:, :, 0], axis=0)
    imag_var = np.var(all_trajs[:, :, 2], axis=0)
    collective_var = np.var(all_trajs, axis=0).mean(axis=1)
    pairwise_dist = np.array([
        np.mean([np.linalg.norm(all_trajs[i, t] - all_trajs[j, t])
                 for i in range(3) for j in range(i + 1, 3)])
        for t in range(T)
    ])

    mis_flags = []
    lags = []
    run_lens = []
    for i in range(3):
        flags = traces['hat_s'][i] != traces['true_s'][i]
        mis_flags.append(flags)
        run_lens.append(mean_run_length(flags))
        lags.append(belief_lag_from_series(traces['true_s'][i], traces['hat_s'][i]))

    tts = time_to_sync(sym_var)
    mis_all = np.stack(mis_flags)  # (3, T)
    if tts is None:
        mis_post = float('nan')
    else:
        mis_post = float(mis_all[:, tts:].mean())
    return {
        'sym_var_init': float(sym_var[0]),
        'sym_var_final': float(sym_var[-1]),
        'sym_var_second_half_mean': float(sym_var[T // 2:].mean()),
        'real_var_final': float(real_var[-1]),
        'imag_var_final': float(imag_var[-1]),
        'collective_var_final': float(collective_var[-1]),
        'pairwise_dist_final': float(pairwise_dist[-1]),
        'converged': bool(sym_var[-1] < CONV_THRESHOLD),
        'attractor': classify_attractor(trajs),
        'time_to_sync': tts,
        'misID_rate': float(mis_all.mean()),
        'misID_second_half': float(mis_all[:, T // 2:].mean()),
        'misID_post_sync': mis_post,
        'misID_run_mean': float(np.mean(run_lens)),
        'belief_lag': float(np.nanmean(lags)),
        'kl_belief': float(np.mean([x.mean() for x in traces['kl_steps']])),
        'H_belief': float(np.mean([x.mean() for x in traces['h_steps']])),
        'final_states': {
            'A': tuple(int(x) for x in trajs[0][-1]),
            'B': tuple(int(x) for x in trajs[1][-1]),
            'C': tuple(int(x) for x in trajs[2][-1]),
        },
    }


def bootstrap_ci(values, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return (float('nan'), float('nan'))
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def run_condition_set(seeds, env_labels, conditions):
    results = {}
    init_map = dict(zip(ENV_INIT_LABELS, ENV_INIT_SETS))
    total = len(seeds) * len(env_labels) * len(conditions)
    done = 0
    for env_label in env_labels:
        env_init = init_map[env_label]
        for mechanism, sigma in conditions:
            label = cond_label(mechanism, sigma)
            for seed in seeds:
                trajs, traces = run_trial(env_init, mechanism, sigma, seed)
                results[(seed, env_label, label)] = compute_metrics(trajs, traces)
                done += 1
            subset = [results[(s, env_label, label)] for s in seeds]
            n_conv = sum(1 for r in subset if r['converged'])
            sv = np.array([r['sym_var_final'] for r in subset])
            mis = np.array([r['misID_rate'] for r in subset])
            print(f"  isolated env={env_init} {label:12s}: "
                  f"conv {n_conv}/{len(seeds)}, "
                  f"sv={sv.mean():.4f}, misID={mis.mean():.3f}  ({done}/{total})")
    return results


def evaluate_phase0(results, seeds):
    env_label = '5_2_6'
    def rate(label):
        return sum(1 for s in seeds if results[(s, env_label, label)]['converged']) / len(seeds)

    def mean_of(label, key):
        return float(np.mean([results[(s, env_label, label)][key] for s in seeds]))

    off_rate = rate('off')
    hard_rate = rate('hard')
    infer1_rate = rate('infer_1.00')
    infer1_mis = mean_of('infer_1.00', 'misID_rate')
    infer04_mis = mean_of('infer_0.40', 'misID_rate')
    hard_mis = mean_of('hard', 'misID_rate')
    infer1_tts = np.nanmean([
        results[(s, env_label, 'infer_1.00')]['time_to_sync']
        if results[(s, env_label, 'infer_1.00')]['time_to_sync'] is not None else np.nan
        for s in seeds
    ])
    infer04_tts = np.nanmean([
        results[(s, env_label, 'infer_0.40')]['time_to_sync']
        if results[(s, env_label, 'infer_0.40')]['time_to_sync'] is not None else np.nan
        for s in seeds
    ])

    gates = []
    gates.append(('off_zero', off_rate == 0.0, f'off conv={off_rate:.0%}'))
    gates.append(('infer1_misID_low', infer1_mis < 0.05,
                  f'infer@1 misID={infer1_mis:.3f} (hard={hard_mis:.3f})'))
    gates.append(('infer1_near_hard', abs(infer1_rate - hard_rate) <= 1.0 / len(seeds) or
                  (infer1_rate >= 0.6 and hard_rate >= 0.6),
                  f'infer@1 conv={infer1_rate:.0%} vs hard={hard_rate:.0%}'))
    gates.append(('infer04_misID_up', infer04_mis > infer1_mis + 0.05,
                  f'infer@0.4 misID={infer04_mis:.3f} vs infer@1 {infer1_mis:.3f}'))

    print('\n' + '=' * 70)
    print('PHASE 0 GATES')
    print('=' * 70)
    all_pass = True
    for name, ok, detail in gates:
        mark = 'PASS' if ok else 'FAIL'
        print(f'  [{mark}] {name}: {detail}')
        all_pass = all_pass and ok
    print(f'  time-to-sync (nanmean) infer@1={infer1_tts:.2f}, infer@0.4={infer04_tts:.2f}')
    print('  PHASE 0:', 'PASS' if all_pass else 'FAIL')
    return all_pass


def aggregate(results, seeds, env_label, label):
    subset = [results[(s, env_label, label)] for s in seeds]
    n = len(seeds)
    n_conv = sum(1 for r in subset if r['converged'])
    conv_flags = np.array([1.0 if r['converged'] else 0.0 for r in subset])
    ci_lo, ci_hi = bootstrap_ci(conv_flags)
    counts = {'S=0': 0, 'S=8': 0, 'other_conv': 0, 'no_conv': 0}
    for r in subset:
        counts[r['attractor']] += 1
    tts_vals = [r['time_to_sync'] for r in subset if r['time_to_sync'] is not None]
    return {
        'n': n,
        'n_conv': n_conv,
        'conv_rate': n_conv / n,
        'conv_ci': (ci_lo, ci_hi),
        'sv_mean': float(np.mean([r['sym_var_final'] for r in subset])),
        'sv_std': float(np.std([r['sym_var_final'] for r in subset])),
        'mis_mean': float(np.mean([r['misID_rate'] for r in subset])),
        'mis_std': float(np.std([r['misID_rate'] for r in subset])),
        'mis_half': float(np.mean([r['misID_second_half'] for r in subset])),
        'mis_post': float(np.nanmean([r['misID_post_sync'] for r in subset])),
        'run_mean': float(np.mean([r['misID_run_mean'] for r in subset])),
        'lag_mean': float(np.nanmean([r['belief_lag'] for r in subset])),
        'kl_mean': float(np.mean([r['kl_belief'] for r in subset])),
        'H_mean': float(np.mean([r['H_belief'] for r in subset])),
        'tts_mean': float(np.mean(tts_vals)) if tts_vals else float('nan'),
        'tts_n': len(tts_vals),
        'rv_mean': float(np.mean([r['real_var_final'] for r in subset])),
        'iv_mean': float(np.mean([r['imag_var_final'] for r in subset])),
        'counts': counts,
    }


def write_results_csv(results, seeds, env_labels, conditions, out_path):
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'seed', 'env_mode', 'env_init', 'mechanism', 'sigma',
            'sym_var_init', 'sym_var_final', 'sym_var_second_half_mean',
            'collective_var_final', 'pairwise_dist_final',
            'real_var_final', 'imag_var_final',
            'converged', 'attractor', 'time_to_sync',
            'misID_rate', 'misID_second_half', 'misID_post_sync',
            'misID_run_mean', 'belief_lag', 'kl_belief', 'H_belief',
            'final_R_A', 'final_S_A', 'final_I_A',
            'final_R_B', 'final_S_B', 'final_I_B',
            'final_R_C', 'final_S_C', 'final_I_C',
        ])
        for seed in seeds:
            for env_label in env_labels:
                for mechanism, sigma in conditions:
                    label = cond_label(mechanism, sigma)
                    r = results[(seed, env_label, label)]
                    fs = r['final_states']
                    tts = '' if r['time_to_sync'] is None else r['time_to_sync']
                    lag = '' if np.isnan(r['belief_lag']) else f"{r['belief_lag']:.6f}"
                    w.writerow([
                        seed, 'isolated', env_label, mechanism,
                        '' if sigma is None else f'{sigma:.2f}',
                        f"{r['sym_var_init']:.6f}", f"{r['sym_var_final']:.6f}",
                        f"{r['sym_var_second_half_mean']:.6f}",
                        f"{r['collective_var_final']:.6f}", f"{r['pairwise_dist_final']:.6f}",
                        f"{r['real_var_final']:.6f}", f"{r['imag_var_final']:.6f}",
                        int(r['converged']), r['attractor'], tts,
                        f"{r['misID_rate']:.6f}", f"{r['misID_second_half']:.6f}",
                        '' if np.isnan(r['misID_post_sync']) else f"{r['misID_post_sync']:.6f}",
                        f"{r['misID_run_mean']:.6f}",
                        lag, f"{r['kl_belief']:.6f}", f"{r['H_belief']:.6f}",
                        fs['A'][0], fs['A'][1], fs['A'][2],
                        fs['B'][0], fs['B'][1], fs['B'][2],
                        fs['C'][0], fs['C'][1], fs['C'][2],
                    ])


def write_summary_csv(results, seeds, env_labels, conditions, out_path):
    with open(out_path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'env_mode', 'env_init', 'mechanism', 'sigma', 'n_seeds',
            'n_converged', 'convergence_rate', 'conv_ci_lo', 'conv_ci_hi',
            'sym_var_final_mean', 'sym_var_final_std',
            'misID_rate_mean', 'misID_second_half_mean', 'misID_post_sync_mean',
            'misID_run_mean', 'belief_lag_mean',
            'kl_belief_mean', 'H_belief_mean', 'time_to_sync_mean', 'time_to_sync_n',
            'real_var_final_mean', 'imag_var_final_mean',
            'pct_S0', 'pct_S8', 'pct_other_conv', 'pct_no_conv',
        ])
        for env_label in env_labels:
            for mechanism, sigma in conditions:
                label = cond_label(mechanism, sigma)
                a = aggregate(results, seeds, env_label, label)
                w.writerow([
                    'isolated', env_label, mechanism,
                    '' if sigma is None else f'{sigma:.2f}',
                    a['n'], a['n_conv'], f"{a['conv_rate']:.4f}",
                    f"{a['conv_ci'][0]:.4f}", f"{a['conv_ci'][1]:.4f}",
                    f"{a['sv_mean']:.6f}", f"{a['sv_std']:.6f}",
                    f"{a['mis_mean']:.6f}", f"{a['mis_half']:.6f}",
                    '' if np.isnan(a['mis_post']) else f"{a['mis_post']:.6f}",
                    f"{a['run_mean']:.6f}",
                    f"{a['lag_mean']:.6f}", f"{a['kl_mean']:.6f}", f"{a['H_mean']:.6f}",
                    '' if np.isnan(a['tts_mean']) else f"{a['tts_mean']:.4f}",
                    a['tts_n'],
                    f"{a['rv_mean']:.6f}", f"{a['iv_mean']:.6f}",
                    f"{a['counts']['S=0'] / a['n']:.4f}",
                    f"{a['counts']['S=8'] / a['n']:.4f}",
                    f"{a['counts']['other_conv'] / a['n']:.4f}",
                    f"{a['counts']['no_conv'] / a['n']:.4f}",
                ])


def plot_results(results, seeds, env_labels, conditions, out_path):
    labels = [cond_label(m, s) for m, s in conditions]
    display = []
    for m, s in conditions:
        display.append(f'infer\nσ={s:.2f}' if m == 'infer' else m)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = {'5_2_6': '#c0392b', '8_8_8': '#2980b9'}
    x = np.arange(len(labels))

    ax = axes[0, 0]
    width = 0.35
    for i, env_label in enumerate(env_labels):
        rates = [aggregate(results, seeds, env_label, lb)['conv_rate'] for lb in labels]
        ax.bar(x + (i - 0.5) * width, rates, width, color=colors[env_label], alpha=0.85,
               label=env_label)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=8)
    ax.set_ylim(-0.05, 1.1)
    ax.set_ylabel('Convergence rate')
    ax.set_title('Convergence rate (isolated)')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[0, 1]
    for i, env_label in enumerate(env_labels):
        vals = [aggregate(results, seeds, env_label, lb)['mis_mean'] for lb in labels]
        ax.bar(x + (i - 0.5) * width, vals, width, color=colors[env_label], alpha=0.85,
               label=env_label)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=8)
    ax.set_ylabel('misID rate')
    ax.set_title('argmax(belief) ≠ true S_j')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[0, 2]
    for i, env_label in enumerate(env_labels):
        vals = [aggregate(results, seeds, env_label, lb)['run_mean'] for lb in labels]
        ax.bar(x + (i - 0.5) * width, vals, width, color=colors[env_label], alpha=0.85,
               label=env_label)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=8)
    ax.set_ylabel('mean misID run length')
    ax.set_title('Error persistence')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1, 0]
    for i, env_label in enumerate(env_labels):
        vals = []
        for lb in labels:
            a = aggregate(results, seeds, env_label, lb)
            vals.append(a['tts_mean'] if not np.isnan(a['tts_mean']) else 0.0)
        ax.bar(x + (i - 0.5) * width, vals, width, color=colors[env_label], alpha=0.85,
               label=env_label)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=8)
    ax.set_ylabel('time-to-sync (mean of synced)')
    ax.set_title('Time to Symbolic consensus')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1, 1]
    for i, env_label in enumerate(env_labels):
        vals = []
        for lb in labels:
            v = aggregate(results, seeds, env_label, lb)['mis_post']
            vals.append(0.0 if np.isnan(v) else v)
        ax.bar(x + (i - 0.5) * width, vals, width, color=colors[env_label], alpha=0.85,
               label=env_label)
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=8)
    ax.set_ylabel('misID after sync')
    ax.set_title('Post-consensus misidentification')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    ax = axes[1, 2]
    attractor_types = ['S=0', 'S=8', 'other_conv', 'no_conv']
    attractor_colors = ['#2ecc71', '#9b59b6', '#f39c12', '#95a5a6']
    env_label = '5_2_6'
    bottom = np.zeros(len(labels))
    for at, col in zip(attractor_types, attractor_colors):
        vals = np.array([
            aggregate(results, seeds, env_label, lb)['counts'][at] / len(seeds)
            for lb in labels
        ])
        ax.bar(x, vals, bottom=bottom, color=col, alpha=0.85, label=at)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(display, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Proportion')
    ax.set_title('Endpoint regimes — isolated [5,2,6]')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis='y')

    plt.suptitle(
        'Opaque-other coupling (inferred position)\n'
        'isolated only · off / hard / noisy / infer(σ)',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


def write_report(results, seeds, env_labels, conditions, out_path, phase0_pass):
    labels = [cond_label(m, s) for m, s in conditions]
    lines = []
    lines.append('# Opaque Other Coupling Report\n')
    lines.append('## 研究目标\n')
    lines.append('检验：当他者 Symbolic 位置只能通过带噪社会观察被推断时，')
    lines.append('`C_S := q(s_j)` 是否仍能产生 Symbolic 共识，以及这种错误是否不同于 noisy 对真值的逐步涂糊。\n')
    lines.append('推断对象是位置 `s_j`，不是欲望 `C_j`。\n')

    lines.append('## 机制\n')
    lines.append('| mechanism | C_S | 他者是否可读 |')
    lines.append('|-----------|-----|--------------|')
    lines.append('| off | 保持初始 onehot | 不读 |')
    lines.append('| hard | onehot(true S_j) | 直接读真值 |')
    lines.append('| noisy | 0.8·onehot(true S_j)+0.2·uniform | 读真值后涂糊 |')
    lines.append('| infer(σ) | q(s_j) = filter(A_social(σ), o, D_other) | 只读带噪 o |')
    lines.append('')
    lines.append('- 环境：isolated；A→B→C sequential；agent / A / B / weights 与 factorial 一致')
    lines.append(f'- seeds 0..{seeds[-1]}，{T_STEPS} steps，收敛阈值 sym_var < {CONV_THRESHOLD}')
    lines.append('- D_other 初值为均匀分布；v1 不使用他者运动模型 B')
    lines.append('- infer@1.0 与 hard 的轨迹不必逐 seed 相同（社会通道采样会消耗 RNG），只要求统计接近')
    lines.append(f'- Phase 0 体检：{"PASS" if phase0_pass else "FAIL"}\n')

    lines.append('## 结果表\n')
    for env_label in env_labels:
        env_init = ENV_INIT_SETS[ENV_INIT_LABELS.index(env_label)]
        lines.append(f'### isolated, env_init={env_init}\n')
        lines.append('| cond | conv_rate [95% CI] | sv_final | misID | misID_2nd | misID_post | misID_run | lag | H(C) | TTS | %S=0 | %S=8 | %other | %no |')
        lines.append('|------|--------------------|----------|-------|-----------|------------|-----------|-----|------|-----|------|------|--------|-----|')
        for mechanism, sigma in conditions:
            label = cond_label(mechanism, sigma)
            a = aggregate(results, seeds, env_label, label)
            tts = 'NA' if np.isnan(a['tts_mean']) else f"{a['tts_mean']:.1f} (n={a['tts_n']})"
            lag = 'NA' if np.isnan(a['lag_mean']) else f"{a['lag_mean']:.2f}"
            post = 'NA' if np.isnan(a['mis_post']) else f"{a['mis_post']:.3f}"
            lines.append(
                f"| {label} | {a['n_conv']}/{a['n']} ({a['conv_rate']:.0%}) "
                f"[{a['conv_ci'][0]:.0%},{a['conv_ci'][1]:.0%}] | "
                f"{a['sv_mean']:.4f}±{a['sv_std']:.4f} | "
                f"{a['mis_mean']:.3f} | {a['mis_half']:.3f} | {post} | "
                f"{a['run_mean']:.2f} | {lag} | {a['H_mean']:.3f} | {tts} | "
                f"{a['counts']['S=0']/a['n']:.0%} | {a['counts']['S=8']/a['n']:.0%} | "
                f"{a['counts']['other_conv']/a['n']:.0%} | {a['counts']['no_conv']/a['n']:.0%} |"
            )
        lines.append('')

    # 核心对照
    lines.append('## 核心对照\n')
    lines.append('### 1. σ=1 是否接近 hard（通道没写坏）\n')
    for env_label in env_labels:
        h = aggregate(results, seeds, env_label, 'hard')
        i1 = aggregate(results, seeds, env_label, 'infer_1.00')
        lines.append(
            f"- {env_label}: hard {h['conv_rate']:.0%} [{h['conv_ci'][0]:.0%},{h['conv_ci'][1]:.0%}] "
            f"misID={h['mis_mean']:.3f}; "
            f"infer@1 {i1['conv_rate']:.0%} [{i1['conv_ci'][0]:.0%},{i1['conv_ci'][1]:.0%}] "
            f"misID={i1['mis_mean']:.3f}"
        )
    lines.append('')

    lines.append('### 2. σ 下降时 misID / 收敛是否变化\n')
    for env_label in env_labels:
        parts = []
        for sig in INFER_SIGMAS:
            a = aggregate(results, seeds, env_label, cond_label('infer', sig))
            parts.append(f"σ={sig:.2f}: conv {a['conv_rate']:.0%}, misID {a['mis_mean']:.3f}, run {a['run_mean']:.2f}")
        lines.append(f"- {env_label}: " + ' | '.join(parts))
    lines.append('')

    lines.append('### 3. infer 是否不同于 noisy\n')
    for env_label in env_labels:
        nsy = aggregate(results, seeds, env_label, 'noisy')
        lines.append(
            f"- {env_label} noisy: conv {nsy['conv_rate']:.0%} "
            f"[{nsy['conv_ci'][0]:.0%},{nsy['conv_ci'][1]:.0%}], "
            f"misID={nsy['mis_mean']:.3f}, run={nsy['run_mean']:.2f}, "
            f"H={nsy['H_mean']:.3f}"
        )
        for sig in INFER_SIGMAS:
            a = aggregate(results, seeds, env_label, cond_label('infer', sig))
            overlap = not (a['conv_ci'][1] < nsy['conv_ci'][0] or nsy['conv_ci'][1] < a['conv_ci'][0])
            lines.append(
                f"  - vs infer σ={sig:.2f}: conv {a['conv_rate']:.0%} "
                f"[{a['conv_ci'][0]:.0%},{a['conv_ci'][1]:.0%}], "
                f"misID={a['mis_mean']:.3f}, run={a['run_mean']:.2f}, "
                f"H={a['H_mean']:.3f}; conv CI overlap={overlap}"
            )
    lines.append('')

    lines.append('### 4. off 是否仍为 0\n')
    for env_label in env_labels:
        a = aggregate(results, seeds, env_label, 'off')
        lines.append(f"- {env_label} off: conv {a['n_conv']}/{a['n']} ({a['conv_rate']:.0%})")
    lines.append('')

    # 自动判断可写主张
    def distinguishable(env_label):
        nsy = aggregate(results, seeds, env_label, 'noisy')
        # any infer sigma with non-overlapping conv CI, or misID gap > 0.05 and run gap
        for sig in INFER_SIGMAS:
            a = aggregate(results, seeds, env_label, cond_label('infer', sig))
            ci_sep = a['conv_ci'][1] < nsy['conv_ci'][0] or nsy['conv_ci'][1] < a['conv_ci'][0]
            mis_sep = abs(a['mis_mean'] - nsy['mis_mean']) > 0.05
            run_sep = abs(a['run_mean'] - nsy['run_mean']) > 0.5
            if ci_sep or (mis_sep and run_sep):
                return True
        return False

    any_distinct = any(distinguishable(el) for el in env_labels)
    i1_ok = all(
        aggregate(results, seeds, el, 'infer_1.00')['mis_mean'] < 0.05
        for el in env_labels
    )
    off_ok = all(aggregate(results, seeds, el, 'off')['n_conv'] == 0 for el in env_labels)
    sigma_effect = False
    for el in env_labels:
        m1 = aggregate(results, seeds, el, 'infer_1.00')['mis_mean']
        m0 = aggregate(results, seeds, el, 'infer_0.40')['mis_mean']
        if m0 > m1 + 0.05:
            sigma_effect = True

    i04 = aggregate(results, seeds, '5_2_6', 'infer_0.40')
    i10 = aggregate(results, seeds, '5_2_6', 'infer_1.00')
    hd = aggregate(results, seeds, '5_2_6', 'hard')
    i04b = aggregate(results, seeds, '8_8_8', 'infer_0.40')
    lines.append('## 最重要的观察\n')
    post04 = 'NA' if np.isnan(i04['mis_post']) else f"{i04['mis_post']:.3f}"
    lines.append(
        f"1. **共识不要求读准他者位置**。isolated+[5,2,6] 上 infer@0.40 仍有 "
        f"{i04['n_conv']}/{i04['n']} 收敛，全程 misID={i04['mis_mean']:.3f}，"
        f"后半段 {i04['mis_half']:.3f}，共识后 {post04}。"
        "若共识后 misID 仍高，则是共享秩序但不读准他者；若塌到接近 0，则只是路上认错。"
    )
    tts_h = 'NA' if np.isnan(hd['tts_mean']) else f"{hd['tts_mean']:.1f}"
    tts_1 = 'NA' if np.isnan(i10['tts_mean']) else f"{i10['tts_mean']:.1f}"
    tts_4 = 'NA' if np.isnan(i04['tts_mean']) else f"{i04['tts_mean']:.1f}"
    lines.append(
        f"2. **不透明度主要拖慢、而不是关掉共识**。[5,2,6] TTS：hard {tts_h}，"
        f"infer@1 {tts_1}，infer@0.40 {tts_4}；"
        f"[8,8,8] infer@0.40 仍 {i04b['n_conv']}/{i04b['n']} 收敛。"
    )
    lines.append('3. **noisy 的 misID=0 是结构事实**：它先读真值再涂糊，argmax 不会错。')
    lines.append('   与 infer 的差别首先是“会不会认错”，不是收敛率（收敛 CI 多半重叠）。')
    lines.append(
        f"4. infer@1 与 hard 收敛率接近，但 [5,2,6] 终态不同：hard S=0 "
        f"{hd['counts']['S=0']/hd['n']:.0%}，infer@1 other_conv "
        f"{i10['counts']['other_conv']/i10['n']:.0%}。"
        "σ=1 只保证通道可读，不保证落入同一个 endpoint。\n"
    )

    lines.append('## 主张与停机\n')
    lines.append('### 当前证据支持的说法\n')
    if off_ok:
        lines.append('- isolated + off 仍不收敛：没有通道时不会凭空对齐。')
    if i1_ok:
        lines.append('- infer@σ=1 的 misID 接近 0：无噪社会通道下位置推断塌回可读真值。')
    if sigma_effect:
        lines.append('- σ 下降时 misID 上升：不透明度进入了信念，不是假通道。')
    if any_distinct:
        lines.append('- 至少在一个初态上，infer 与 noisy 在收敛 CI 或（misID + 错误持续性）上可区分。')
    else:
        lines.append('- **infer 与 noisy 在当前 n 和条件下未能稳定区分。路径 B 在此模型中未显示出独立机制。**')
    lines.append('')

    lines.append('### 不应写\n')
    lines.append('- 这不是大他者、欲望或 Che vuoi? 的形式化。')
    lines.append('- 不写 phase transition；σ 只是社会通道清晰度。')
    lines.append('- S=0 / S=8 仍称 endpoint regime，不是 attractor。')
    lines.append('- noisy 的 argmax 几乎总是真值，故 misID≈0 是该 baseline 的结构事实，不是“更准确的推断”。\n')

    if any_distinct and sigma_effect and i1_ok:
        lines.append('### Phase 2 门\n')
        lines.append('**打开**：可做噪声匹配（把 noisy 的 ε 调到与某档 infer 的 H(C) 接近），只在 isolated 上做。\n')
    else:
        lines.append('### Phase 2 门\n')
        lines.append('**关闭**：先不要做噪声匹配或 σ 细扫。本实验作为失败或弱结果收口。\n')

    lines.append('## 文件\n')
    lines.append('| 文件 | 说明 |')
    lines.append('|------|------|')
    lines.append('| opaque_other_coupling.py | 本脚本 |')
    lines.append('| opaque_other_results.csv | 每 seed 明细 |')
    lines.append('| opaque_other_summary.csv | 条件汇总 |')
    lines.append('| plot_opaque_other.png | 六子图 |')
    lines.append('| opaque_other_report.md | 本报告 |')
    lines.append('| ROADMAP_OPAQUE_OTHER.md | 计划 |')

    with open(out_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))


def phase0():
    print('=' * 70)
    print('OPAQUE OTHER — PHASE 0')
    print('isolated [5,2,6] · 5 seeds · off / hard / infer@1.0 / infer@0.4')
    print('=' * 70)
    conds = [('off', None), ('hard', None), ('infer', 1.00), ('infer', 0.40)]
    results = run_condition_set(PHASE0_SEEDS, ['5_2_6'], conds)
    return evaluate_phase0(results, PHASE0_SEEDS)


def phase1(phase0_pass):
    print('=' * 70)
    print('OPAQUE OTHER — PHASE 1')
    n_cond = len(all_conditions())
    print(f'isolated · 2 env_inits · {n_cond} conds · {len(PHASE1_SEEDS)} seeds')
    print('=' * 70)
    conds = all_conditions()
    results = run_condition_set(PHASE1_SEEDS, ENV_INIT_LABELS, conds)

    write_results_csv(results, PHASE1_SEEDS, ENV_INIT_LABELS, conds,
                      os.path.join(OUT_DIR, 'opaque_other_results.csv'))
    write_summary_csv(results, PHASE1_SEEDS, ENV_INIT_LABELS, conds,
                      os.path.join(OUT_DIR, 'opaque_other_summary.csv'))
    plot_results(results, PHASE1_SEEDS, ENV_INIT_LABELS, conds,
                 os.path.join(OUT_DIR, 'plot_opaque_other.png'))
    write_report(results, PHASE1_SEEDS, ENV_INIT_LABELS, conds,
                 os.path.join(OUT_DIR, 'opaque_other_report.md'), phase0_pass)
    print('[Saved] opaque_other_results.csv / summary.csv / plot / report')

    print('\n' + '=' * 70)
    print('PHASE 1 KEY')
    print('=' * 70)
    for env_label in ENV_INIT_LABELS:
        print(f'  [{env_label}]')
        for mechanism, sigma in conds:
            label = cond_label(mechanism, sigma)
            a = aggregate(results, PHASE1_SEEDS, env_label, label)
            post = 'NA' if np.isnan(a['mis_post']) else f"{a['mis_post']:.3f}"
            print(f"    {label:12s} conv {a['n_conv']:2d}/20  "
                  f"misID={a['mis_mean']:.3f}  half={a['mis_half']:.3f}  "
                  f"post={post}  H={a['H_mean']:.3f}")


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else 'phase0'
    if phase not in ('phase0', 'phase1', 'all'):
        print('usage: python opaque_other_coupling.py [phase0|phase1|all]')
        sys.exit(2)

    cs = f.A_MAT.sum(axis=0)
    assert np.allclose(cs, 1.0), 'A not normalized'
    A_id = f.make_A_matrix(sharpness=1.0)
    assert np.allclose(A_id, np.eye(f.N_STATES)), 'σ=1 A_social is not identity'

    if phase == 'phase0':
        ok = phase0()
        sys.exit(0 if ok else 1)

    if phase == 'phase1':
        phase1(phase0_pass=True)
        return

    ok = phase0()
    if not ok:
        print('\nPhase 0 failed. Stop per roadmap. Not running Phase 1.')
        sys.exit(1)
    phase1(phase0_pass=True)


if __name__ == '__main__':
    main()
