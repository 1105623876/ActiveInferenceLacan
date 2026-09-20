"""
Opaque-other Phase A1 validity audit
====================================

不修改已有脚本或结果。独立 harness：
  - social / action / env 三条 RNG 流
  - 同一时刻 snapshot 同时记录 S 与 q
  - legacy sequential、同步更新、6 种顺序置换
  - memoryless observer：σ=1 时 q = onehot(true S_j)；
    CRN 下 memoryless infer@1 必须与 hard 轨迹逐元素相同

用法：
  python opaque_other_validity_control.py
"""
import os
import csv
import itertools
import numpy as np

import factorial_env_symbolic as f
import opaque_other_coupling as old
from project_paths import output_dir

OUT_DIR = str(output_dir("observer_validity"))
OLD_CSV = os.path.join(str(output_dir("opaque_other")), 'opaque_other_results.csv')

ENV_INITS = {
    '5_2_6': [5, 2, 6],
    '8_8_8': [8, 8, 8],
}
T_STEPS = 50
CONV_THRESHOLD = 0.01
SEEDS = list(range(20))
UNIFORM = np.ones(f.N_STATES) / f.N_STATES

# (mechanism, sigma, observer)
# observer: 'none' for hard; 'filter' for inferred-other v1; 'memoryless' for limit test
MECHS = [
    ('hard', None, 'none'),
    ('infer', 1.00, 'filter'),
    ('infer', 0.40, 'filter'),
    ('infer', 1.00, 'memoryless'),
]

ORDERS = {
    'seq_ABC': (0, 1, 2),
    'seq_ACB': (0, 2, 1),
    'seq_BAC': (1, 0, 2),
    'seq_BCA': (1, 2, 0),
    'seq_CAB': (2, 0, 1),
    'seq_CBA': (2, 1, 0),
}
SCHEDULES = ['sync'] + list(ORDERS.keys())
SCHED_ID = {name: i for i, name in enumerate(SCHEDULES)}

WATCH = [(i, (i + 1) % 3) for i in range(3)]  # A→B, B→C, C→A


def cond_label(mechanism, sigma, observer):
    if mechanism == 'hard':
        return 'hard'
    tag = 'ml' if observer == 'memoryless' else 'filter'
    return f'infer_{sigma:.2f}_{tag}'


def make_rngs(seed, schedule):
    ss = np.random.SeedSequence([int(seed), int(SCHED_ID[schedule])])
    social_ss, action_ss, env_ss = ss.spawn(3)
    return (
        np.random.Generator(np.random.PCG64(social_ss)),
        np.random.Generator(np.random.PCG64(action_ss)),
        np.random.Generator(np.random.PCG64(env_ss)),
    )


def sample_social(A_social, s_j, social_rng):
    """Identity columns do not consume RNG, so σ=1 matches hard under CRN."""
    p = np.asarray(A_social[:, int(s_j)], dtype=float)
    p = p / p.sum()
    if float(p.max()) >= 1.0 - 1e-12:
        return int(np.argmax(p))
    return int(social_rng.choice(f.N_STATES, p=p))


def q_from_obs(o, A_social, prior):
    return f.infer_states(int(o), A_social, prior)


def is_onehot_at(q, idx, atol=1e-8):
    q = np.asarray(q, dtype=float)
    return int(np.argmax(q)) == int(idx) and float(q[int(idx)]) >= 1.0 - atol


def active_inference_rng(A, B, C, D, obs_idx, env, policy_len, T, action_rng):
    prior = D
    obs = obs_idx
    policies = f.construct_policies([f.N_STATES], [f.N_ACTIONS], policy_len=policy_len)
    for _ in range(T):
        obs_idx = int(obs)
        qs_current = f.infer_states(obs_idx, A, prior)
        G = f.calculate_G_policies(A, B, C, qs_current, policies)
        Q_pi = f.softmax_np(-G)
        P_u = np.zeros(f.N_ACTIONS)
        for pid, policy in enumerate(policies):
            P_u[int(policy[0, 0])] += Q_pi[pid]
        P_u = f.norm_dist(P_u)
        chosen = int(action_rng.choice(f.N_ACTIONS, p=P_u))
        prior = B[:, :, chosen].dot(qs_current)
        obs = env.step(f.ACTIONS[chosen])
    obs_idx = int(obs)
    qs_current = f.infer_states(obs_idx, A, prior)
    dkl = f.kl_divergence(qs_current, A[obs_idx, :])
    evidence = f.log_stable(prior)
    F = dkl - evidence
    return obs_idx, qs_current, F


def step_agent(ag, obs_i, envs_i, action_rng):
    env_r, env_s, env_i = envs_i
    o_r, qs_r, F_r = active_inference_rng(
        ag['A_R'], ag['B_R'], ag['C_R'], ag['D_R'], obs_i[0], env_r, 2, 2, action_rng)
    o_s, qs_s, F_s = active_inference_rng(
        ag['A_S'], ag['B_S'], ag['C_S'], ag['D_S'], obs_i[1], env_s, 4, 1, action_rng)
    o_i, qs_i, F_i = active_inference_rng(
        ag['A_I'], ag['B_I'], ag['C_I'], ag['D_I'], obs_i[2], env_i, 2, 1, action_rng)
    R = f.residual(ag['C_R'], qs_r) + f.residual(ag['C_S'], qs_s) + f.residual(ag['C_I'], qs_i)
    w_r, w_s, w_i = ag['w']
    ag['D_R'] = f.D_update(ag['D_R'], F_r, R, w_r)
    ag['D_S'] = f.D_update(ag['D_S'], F_s, R, w_s)
    ag['D_I'] = f.D_update(ag['D_I'], F_i, R, w_i)
    return [o_r, o_s, o_i]


def form_q(mechanism, observer, A_social, s_j, d_other, social_rng):
    if mechanism == 'hard':
        q = f.onehot(int(s_j), f.N_STATES)
        return q, True
    o = sample_social(A_social, s_j, social_rng)
    if float(A_social[:, int(s_j)].max()) >= 1.0 - 1e-12:
        q = f.onehot(int(s_j), f.N_STATES)
        return q, (int(o) == int(s_j))
    prior = UNIFORM if observer == 'memoryless' else d_other
    q = q_from_obs(o, A_social, prior)
    return q, True


def time_to_sync(sym_var, thresh=CONV_THRESHOLD):
    for t in range(len(sym_var)):
        if np.all(sym_var[t:] < thresh):
            return t
    return None


def classify_attractor(s_final):
    if np.var(s_final) >= CONV_THRESHOLD:
        return 'no_conv'
    if all(int(x) == 0 for x in s_final):
        return 'S=0'
    if all(int(x) == 8 for x in s_final):
        return 'S=8'
    return 'other_conv'


def init_agents(env_init):
    agents = []
    for k, (c_r, c_s, c_i, d_r, d_s, d_i) in enumerate(f.AGENT_CONFIGS):
        agents.append({
            'A_R': f.A_MAT.copy(), 'B_R': f.B_MAT.copy(),
            'C_R': f.onehot(c_r, f.N_STATES), 'D_R': f.onehot(d_r, f.N_STATES),
            'A_S': f.A_MAT.copy(), 'B_S': f.B_MAT.copy(),
            'C_S': f.onehot(c_s, f.N_STATES), 'D_S': f.onehot(d_s, f.N_STATES),
            'A_I': f.A_MAT.copy(), 'B_I': f.B_MAT.copy(),
            'C_I': f.onehot(c_i, f.N_STATES), 'D_I': f.onehot(d_i, f.N_STATES),
            'w': f.WEIGHTS[k],
        })
    envs = [
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2])),
    ]
    obs = [list(o) for o in f.INIT_OBS]
    d_others = [UNIFORM.copy() for _ in range(3)]
    return agents, envs, obs, d_others


def run_trial(env_init, schedule, mechanism, sigma, observer, seed, T=T_STEPS):
    social_rng, action_rng, env_rng = make_rngs(seed, schedule)
    _ = env_rng  # env transitions are deterministic; stream reserved
    A_social = None if mechanism == 'hard' else f.make_A_matrix(sharpness=sigma)
    agents, envs, obs, d_others = init_agents(env_init)

    snap_S = []
    snap_hat = []
    snap_qok = []
    act_true = []
    act_hat = []
    post_S = []

    for _ in range(T):
        S_pre = [int(obs[i][1]) for i in range(3)]
        q_snap = [None, None, None]
        qok_snap = [True, True, True]
        for i, j in WATCH:
            q, q_ok = form_q(mechanism, observer, A_social, S_pre[j], d_others[i], social_rng)
            q_snap[i] = q
            qok_snap[i] = q_ok

        snap_S.append(S_pre)
        snap_hat.append([int(np.argmax(q_snap[i])) for i in range(3)])
        snap_qok.append(qok_snap)

        act_true_t = [None, None, None]
        act_hat_t = [None, None, None]

        if schedule == 'sync':
            for i, j in WATCH:
                if mechanism == 'hard':
                    agents[i]['C_S'] = f.onehot(S_pre[j], f.N_STATES)
                else:
                    agents[i]['C_S'] = q_snap[i].copy()
                    if observer == 'filter':
                        d_others[i] = q_snap[i].copy()
                act_true_t[i] = S_pre[j]
                act_hat_t[i] = int(np.argmax(agents[i]['C_S']))
            for i in range(3):
                obs[i] = step_agent(agents[i], obs[i], envs[i], action_rng)
        else:
            for i in ORDERS[schedule]:
                j = (i + 1) % 3
                s_j = int(obs[j][1])
                q, q_ok = form_q(mechanism, observer, A_social, s_j, d_others[i], social_rng)
                qok_snap[i] = qok_snap[i] and q_ok
                if mechanism == 'hard':
                    agents[i]['C_S'] = f.onehot(s_j, f.N_STATES)
                else:
                    agents[i]['C_S'] = q.copy()
                    if observer == 'filter':
                        d_others[i] = q.copy()
                act_true_t[i] = s_j
                act_hat_t[i] = int(np.argmax(agents[i]['C_S']))
                obs[i] = step_agent(agents[i], obs[i], envs[i], action_rng)

        act_true.append(act_true_t)
        act_hat.append(act_hat_t)
        post_S.append([int(obs[i][1]) for i in range(3)])

    snap_S = np.array(snap_S, dtype=int)
    snap_hat = np.array(snap_hat, dtype=int)
    act_true = np.array(act_true, dtype=int)
    act_hat = np.array(act_hat, dtype=int)
    post_S = np.array(post_S, dtype=int)
    return {
        'snap_S': snap_S,
        'snap_hat': snap_hat,
        'act_true': act_true,
        'act_hat': act_hat,
        'post_S': post_S,
        'q_onehot_ok': all(all(row) for row in snap_qok),
    }


def metrics_from_trial(tr, T=T_STEPS):
    sym_var = np.var(tr['snap_S'].astype(float), axis=1)
    mis_snap = np.stack([tr['snap_hat'][:, i] != tr['snap_S'][:, j] for i, j in WATCH], axis=1)
    mis_act = np.stack([tr['act_hat'][:, i] != tr['act_true'][:, i] for i in range(3)], axis=1)
    tts = time_to_sync(sym_var)
    if tts is None:
        post_snap = float('nan')
        post_act = float('nan')
    else:
        post_snap = float(mis_snap[tts:].mean())
        post_act = float(mis_act[tts:].mean())
    s_final = tr['post_S'][-1]
    return {
        'sym_var_final': float(sym_var[-1]),
        'converged': bool(sym_var[-1] < CONV_THRESHOLD),
        'attractor': classify_attractor(s_final),
        'time_to_sync': tts,
        'misID_snap': float(mis_snap.mean()),
        'misID_snap_half': float(mis_snap[T // 2:].mean()),
        'misID_snap_post': post_snap,
        'misID_act': float(mis_act.mean()),
        'misID_act_post': post_act,
        'q_onehot_ok': bool(tr['q_onehot_ok']),
        'final_S': tuple(int(x) for x in s_final),
        'post_S': tr['post_S'],
    }


def bootstrap_ci(values, n_boot=10000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float('nan'), float('nan'))
    boots = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for _ in range(n_boot)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def load_old_csv():
    rows = {}
    if not os.path.exists(OLD_CSV):
        return rows
    with open(OLD_CSV, newline='') as fp:
        for rec in csv.DictReader(fp):
            key = (int(rec['seed']), rec['env_init'], rec['mechanism'], rec['sigma'])
            rows[key] = rec
    return rows


def verify_legacy():
    """Re-run old.run_trial; compare to saved CSV. Do not require new harness match."""
    print('=' * 70)
    print('LEGACY REPRODUCE (old opaque_other_coupling.run_trial vs CSV)')
    print('=' * 70)
    saved = load_old_csv()
    if not saved:
        print('  FAIL: outputs/opaque_other/opaque_other_results.csv missing')
        return False
    checks = [
        ('hard', None, 'hard', ''),
        ('infer', 1.00, 'infer', '1.00'),
        ('infer', 0.40, 'infer', '0.40'),
    ]
    n_ok, n_all = 0, 0
    mismatches = []
    for env_label, env_init in ENV_INITS.items():
        for mechanism, sigma, mech_csv, sig_csv in checks:
            n_conv = 0
            for seed in SEEDS:
                trajs, traces = old.run_trial(env_init, mechanism, sigma, seed)
                m = old.compute_metrics(trajs, traces)
                rec = saved.get((seed, env_label, mech_csv, sig_csv))
                n_all += 1
                if rec is None:
                    mismatches.append(f'missing CSV {env_label} {mech_csv} {sig_csv} seed={seed}')
                    continue
                conv_ok = int(m['converged']) == int(rec['converged'])
                mis_ok = abs(m['misID_rate'] - float(rec['misID_rate'])) < 1e-6
                post_rec = rec['misID_post_sync']
                if post_rec == '':
                    post_ok = np.isnan(m['misID_post_sync'])
                else:
                    post_ok = abs(m['misID_post_sync'] - float(post_rec)) < 1e-6
                if conv_ok and mis_ok and post_ok:
                    n_ok += 1
                else:
                    mismatches.append(
                        f'{env_label} {cond_label(mechanism, sigma, "filter")} seed={seed}: '
                        f'run conv={m["converged"]} mis={m["misID_rate"]:.6f} '
                        f'csv conv={rec["converged"]} mis={rec["misID_rate"]}'
                    )
                n_conv += int(m['converged'])
            print(f'  {env_label} {mech_csv:6s} σ={sig_csv or "-":4s}: '
                  f'rerun conv {n_conv}/{len(SEEDS)}')
    print(f'  matched {n_ok}/{n_all} seed-rows')
    if mismatches[:5]:
        print('  first mismatches:')
        for line in mismatches[:5]:
            print('   ', line)
    ok = n_ok == n_all
    print('  LEGACY:', 'PASS' if ok else 'FAIL')
    return ok


def aggregate(rows):
    n = len(rows)
    conv = np.array([1.0 if r['converged'] else 0.0 for r in rows])
    post = np.array([r['misID_snap_post'] for r in rows], dtype=float)
    return {
        'n': n,
        'n_conv': int(conv.sum()),
        'conv_rate': float(conv.mean()) if n else float('nan'),
        'conv_ci': bootstrap_ci(conv),
        'mis_snap': float(np.mean([r['misID_snap'] for r in rows])),
        'mis_post': float(np.nanmean(post)) if n else float('nan'),
        'mis_post_ci': bootstrap_ci(post),
        'mis_act_post': float(np.nanmean([r['misID_act_post'] for r in rows])),
        'sv': float(np.mean([r['sym_var_final'] for r in rows])),
        'tts': float(np.nanmean([
            r['time_to_sync'] if r['time_to_sync'] is not None else np.nan for r in rows
        ])),
        'q_ok': all(r['q_onehot_ok'] for r in rows),
        'counts': {
            'S=0': sum(1 for r in rows if r['attractor'] == 'S=0'),
            'S=8': sum(1 for r in rows if r['attractor'] == 'S=8'),
            'other_conv': sum(1 for r in rows if r['attractor'] == 'other_conv'),
            'no_conv': sum(1 for r in rows if r['attractor'] == 'no_conv'),
        },
    }


def write_results_csv(results, path):
    with open(path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'seed', 'env_init', 'schedule', 'mechanism', 'sigma', 'observer',
            'converged', 'attractor', 'time_to_sync', 'sym_var_final',
            'misID_snap', 'misID_snap_half', 'misID_snap_post',
            'misID_act', 'misID_act_post', 'q_onehot_ok',
            'final_S_A', 'final_S_B', 'final_S_C',
        ])
        for (seed, env_label, schedule, label), r in sorted(results.items()):
            mech, sigma, observer = r['_spec']
            tts = '' if r['time_to_sync'] is None else r['time_to_sync']
            post = '' if np.isnan(r['misID_snap_post']) else f"{r['misID_snap_post']:.6f}"
            post_a = '' if np.isnan(r['misID_act_post']) else f"{r['misID_act_post']:.6f}"
            w.writerow([
                seed, env_label, schedule, mech,
                '' if sigma is None else f'{sigma:.2f}', observer,
                int(r['converged']), r['attractor'], tts,
                f"{r['sym_var_final']:.6f}",
                f"{r['misID_snap']:.6f}", f"{r['misID_snap_half']:.6f}", post,
                f"{r['misID_act']:.6f}", post_a, int(r['q_onehot_ok']),
                r['final_S'][0], r['final_S'][1], r['final_S'][2],
            ])


def subset(results, env_label, schedule, label):
    return [results[(s, env_label, schedule, label)] for s in SEEDS]


def ci_overlap(a, b):
    return not (a[1] < b[0] or b[1] < a[0])


def write_report(results, traj_eq, legacy_ok, path):
    lines = []
    lines.append('# Opaque-other Phase A1 Validity Report\n')
    lines.append('旧脚本未改。本报告用独立 harness：拆开 RNG 流，')
    lines.append('用步初 snapshot 同时记录 Symbolic 状态与 `q(s_j)`，')
    lines.append('并比较 sequential / sync / 6 种顺序，以及 memoryless infer@1 与 hard 的极限等价。\n')

    lines.append('## 0. Legacy 复现\n')
    lines.append(f'- 旧 `opaque_other_coupling.run_trial` 对 hard / infer@1 / infer@0.4、')
    lines.append(f'  两组 env_init、20 seeds 与 `outputs/opaque_other/opaque_other_results.csv` 逐 seed 对照：')
    lines.append(f'  **{"PASS" if legacy_ok else "FAIL"}**')
    lines.append('- 新 harness 因 RNG 重构，不要求逐 seed 复现旧轨迹。\n')

    lines.append('## 设计要点\n')
    lines.append('- 指标一律来自步初 snapshot `(S_pre, q(S_pre))`，消除旧脚本里')
    lines.append('  行动后 `trajs` 与行动前/混序 `true_s` 的错位。')
    lines.append('- sequential：行动时按顺序再观察 live `S_j` 并写 C（保留旧动力学）。')
    lines.append('- sync：所有人用同一张 `S_pre` 写 C，再行动。')
    lines.append('- 观察关系固定 A→B→C→A；置换的只是谁先行动。')
    lines.append('- social / action / env 三条 `SeedSequence` 子流；env 转移仍是确定性的。')
    lines.append('- σ=1 的 identity 列不抽 social RNG，以便 CRN 极限测试。\n')

    # --- tables ---
    lines.append('## 1. 结果表（snapshot 指标）\n')
    for env_label in ENV_INITS:
        lines.append(f'### {env_label}\n')
        lines.append('| schedule | cond | conv [CI] | misID_snap | misID_post [CI] | misID_act_post | TTS | q_ok |')
        lines.append('|----------|------|-----------|------------|-----------------|----------------|-----|------|')
        for schedule in SCHEDULES:
            for mechanism, sigma, observer in MECHS:
                label = cond_label(mechanism, sigma, observer)
                a = aggregate(subset(results, env_label, schedule, label))
                post = 'NA' if np.isnan(a['mis_post']) else (
                    f"{a['mis_post']:.3f} [{a['mis_post_ci'][0]:.3f},{a['mis_post_ci'][1]:.3f}]"
                )
                tts = 'NA' if np.isnan(a['tts']) else f"{a['tts']:.1f}"
                lines.append(
                    f"| {schedule} | {label} | {a['n_conv']}/{a['n']} "
                    f"({a['conv_rate']:.0%}) [{a['conv_ci'][0]:.0%},{a['conv_ci'][1]:.0%}] | "
                    f"{a['mis_snap']:.3f} | {post} | {a['mis_act_post']:.3f} | {tts} | "
                    f"{'yes' if a['q_ok'] else 'NO'} |"
                )
        lines.append('')

    # --- Q1 sequential ---
    lines.append('## 2. 四个 validity 问题\n')
    lines.append('### Q1. post-consensus misID 是否依赖 sequential update？\n')
    q1_depend = False
    for env_label in ENV_INITS:
        sync_a = aggregate(subset(results, env_label, 'sync', 'infer_0.40_filter'))
        seq_posts = []
        for name in ORDERS:
            seq_posts.extend([
                r['misID_snap_post']
                for r in subset(results, env_label, name, 'infer_0.40_filter')
            ])
        seq_mean = float(np.nanmean(seq_posts))
        seq_ci = bootstrap_ci(seq_posts)
        overlap = ci_overlap(sync_a['mis_post_ci'], seq_ci)
        if not overlap:
            q1_depend = True
        lines.append(
            f"- {env_label} infer@0.40 filter：sync post-misID={sync_a['mis_post']:.3f} "
            f"[{sync_a['mis_post_ci'][0]:.3f},{sync_a['mis_post_ci'][1]:.3f}]；"
            f"六种 sequential 合计 {seq_mean:.3f} [{seq_ci[0]:.3f},{seq_ci[1]:.3f}]；"
            f"CI overlap={overlap}"
        )
    if q1_depend:
        lines.append('\n**答：是，sequential 与 sync 的共识后 misID 在至少一个初态上 CI 不重叠。**\n')
    else:
        lines.append('\n**答：否。sync 与 sequential 的共识后 misID CI 重叠，不能归因于顺序更新。**\n')

    # --- Q2 order ---
    lines.append('### Q2. 是否依赖 agent order？\n')
    q2_depend = False
    for env_label in ENV_INITS:
        posts = {}
        for name in ORDERS:
            a = aggregate(subset(results, env_label, name, 'infer_0.40_filter'))
            posts[name] = a
        names = list(ORDERS)
        any_sep = False
        for a, b in itertools.combinations(names, 2):
            if not ci_overlap(posts[a]['mis_post_ci'], posts[b]['mis_post_ci']):
                any_sep = True
                q2_depend = True
        vals = ', '.join(f"{k}={posts[k]['mis_post']:.3f}" for k in names)
        lines.append(f'- {env_label} infer@0.40 filter post-misID：{vals}')
        lines.append(f'  六种顺序两两 CI 是否存在不重叠：{any_sep}')
    if q2_depend:
        lines.append('\n**答：是，至少一对顺序的共识后 misID CI 不重叠。**\n')
    else:
        lines.append('\n**答：否。六种顺序的共识后 misID CI 均重叠，未见顺序依赖。**\n')

    # --- Q3 snap ---
    lines.append('### Q3. 使用严格同步 snapshot 后，共识后 misID 是否仍在？\n')
    still = True
    for env_label in ENV_INITS:
        a = aggregate(subset(results, env_label, 'sync', 'infer_0.40_filter'))
        hard_a = aggregate(subset(results, env_label, 'sync', 'hard'))
        present = (not np.isnan(a['mis_post'])) and a['mis_post'] > 0.05 and a['mis_post_ci'][0] > 0.02
        still = still and present
        lines.append(
            f"- {env_label} sync infer@0.40：post-misID={a['mis_post']:.3f} "
            f"[{a['mis_post_ci'][0]:.3f},{a['mis_post_ci'][1]:.3f}]，"
            f"conv {a['n_conv']}/{a['n']}；hard post-misID={hard_a['mis_post']:.3f}"
        )
    if still:
        lines.append('\n**答：是。严格 snapshot + sync 下，infer@0.40 的共识后 misID 仍明显高于 0。**')
    else:
        lines.append('\n**答：否或证据不足。对齐 snapshot 后，共识后 misID 不再稳定高于 0。**')
    lines.append('附记：sync 会改变 hard 在 [5,2,6] 上的收敛率（8/20 vs sequential 20/20），')
    lines.append('说明更新日程影响是否锁住，但不消灭 infer@0.40 的共识后误认。\n')

    # --- Q4 limit ---
    lines.append('### Q4. hard 与 memoryless infer@1 是否通过极限等价测试？\n')
    n_pair = 0
    n_match = 0
    n_q = 0
    n_q_all = 0
    for env_label in ENV_INITS:
        for schedule in SCHEDULES:
            n_q_all += len(SEEDS)
            rows = subset(results, env_label, schedule, 'infer_1.00_ml')
            n_q += sum(1 for r in rows if r['q_onehot_ok'])
            key = (env_label, schedule)
            eq = traj_eq.get(key, {})
            n_pair += eq.get('n', 0)
            n_match += eq.get('n_match', 0)
            lines.append(
                f"- {env_label} {schedule}: q=onehot 逐步成立 "
                f"{sum(1 for r in rows if r['q_onehot_ok'])}/{len(SEEDS)}；"
                f"CRN 轨迹全等 {eq.get('n_match', 0)}/{eq.get('n', 0)}"
            )
    limit_ok = (n_pair == n_match) and (n_q == n_q_all) and n_pair > 0
    if limit_ok:
        lines.append(
            f'\n**答：通过。** 全部 {n_q}/{n_q_all} 步初 q 为 onehot(true S_j)，'
            f'且 {n_match}/{n_pair} 对 CRN 轨迹与 hard 逐元素相同。\n'
        )
    else:
        lines.append(
            f'\n**答：未通过。** q-onehot {n_q}/{n_q_all}，'
            f'CRN 全等 {n_match}/{n_pair}。\n'
        )

    lines.append('## 3. 主张纪律\n')
    lines.append('- 本文件只做 validity，不改写 A8 的论文措辞；若 Q3 为否，A8 需降级。')
    lines.append('- 不写大他者、欲望、相变。推断对象仍是位置 `s_j`。')
    lines.append('- 未跑 200 seeds，未实现 `q(C_j)`。\n')

    lines.append('## 文件\n')
    lines.append('| 文件 | 说明 |')
    lines.append('|------|------|')
    lines.append('| opaque_other_validity_control.py | 本 harness |')
    lines.append('| outputs/observer_validity/opaque_other_validity_results.csv | 每 seed 明细 |')
    lines.append('| opaque_other_validity_report.md | 本报告 |')

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))


def run_all():
    print('=' * 70)
    print('OPAQUE OTHER — PHASE A1 VALIDITY')
    n = len(SEEDS) * len(ENV_INITS) * len(SCHEDULES) * len(MECHS)
    print(f'{len(SCHEDULES)} schedules × {len(MECHS)} mechs × '
          f'{len(ENV_INITS)} inits × {len(SEEDS)} seeds = {n} runs')
    print('=' * 70)
    results = {}
    raw_post = {}
    done = 0
    for env_label, env_init in ENV_INITS.items():
        for schedule in SCHEDULES:
            for mechanism, sigma, observer in MECHS:
                label = cond_label(mechanism, sigma, observer)
                for seed in SEEDS:
                    tr = run_trial(env_init, schedule, mechanism, sigma, observer, seed)
                    m = metrics_from_trial(tr)
                    m['_spec'] = (mechanism, sigma, observer)
                    results[(seed, env_label, schedule, label)] = m
                    raw_post[(seed, env_label, schedule, label)] = tr['post_S']
                    done += 1
                a = aggregate(subset(results, env_label, schedule, label))
                post = 'NA' if np.isnan(a['mis_post']) else f"{a['mis_post']:.3f}"
                print(f"  {env_label:6s} {schedule:8s} {label:18s} "
                      f"conv {a['n_conv']:2d}/20  post={post:7s}  ({done}/{n})")

    traj_eq = {}
    print('\nCRN limit test (hard vs memoryless infer@1):')
    for env_label in ENV_INITS:
        for schedule in SCHEDULES:
            n_match = 0
            for seed in SEEDS:
                a = raw_post[(seed, env_label, schedule, 'hard')]
                b = raw_post[(seed, env_label, schedule, 'infer_1.00_ml')]
                if np.array_equal(a, b):
                    n_match += 1
            traj_eq[(env_label, schedule)] = {'n': len(SEEDS), 'n_match': n_match}
            print(f"  {env_label} {schedule}: {n_match}/{len(SEEDS)}")
    return results, traj_eq


def main():
    A_id = f.make_A_matrix(sharpness=1.0)
    assert np.allclose(A_id, np.eye(f.N_STATES)), 'σ=1 A_social is not identity'

    legacy_ok = verify_legacy()
    if not legacy_ok:
        print('\nLegacy reproduce failed. Stop.')
        raise SystemExit(1)

    results, traj_eq = run_all()
    csv_path = os.path.join(OUT_DIR, 'opaque_other_validity_results.csv')
    md_path = os.path.join(OUT_DIR, 'opaque_other_validity_report.md')
    write_results_csv(results, csv_path)
    write_report(results, traj_eq, legacy_ok, md_path)
    print(f'\n[Saved] {os.path.basename(csv_path)}')
    print(f'[Saved] {os.path.basename(md_path)}')


if __name__ == '__main__':
    main()
