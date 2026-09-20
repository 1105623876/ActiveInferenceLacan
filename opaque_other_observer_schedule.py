"""
Observer × update-schedule factorial
====================================

独立脚本，不改旧实验。

因子
  observer : hard | smooth | memoryless | static | leaky
  schedule : sync | seq_ABC
  env      : [5,2,6] 主环境；[8,8,8] 容易对照
  n        : 200 seeds / 格

锁定假设
  - σ = 0.4（所有推断观察者）
  - leaky：D ← (1-λ)q + λ·uniform，λ=0.20
  - smooth：读真值后涂糊，ε 使 H(C) 等于 memoryless@0.4 的期望信念熵
    （解析标定，不消耗实验 RNG）
  - 指标用步初 snapshot (S, q)；诊断不读机制 RNG
  - CRN 按 (seed, t, agent, slot) 预生成，观察者/日程共享同一张表
  - 主终点：convergence（snapshot sym_var[-1] < 0.01）
  - post-consensus misID：仅收敛轨迹的次要终点
  - 比例用 Wilson CI；同 seed 用 McNemar exact + Holm
  - 预先检验 schedule × observer interaction（seed 块置换）
  - 不因 CI 重叠宣称“无差异”

用法
  python opaque_other_observer_schedule.py smoke
  python opaque_other_observer_schedule.py
"""
import os
import sys
import csv
import math
import numpy as np

import factorial_env_symbolic as f

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

ENV_INITS = {
    '5_2_6': [5, 2, 6],
    '8_8_8': [8, 8, 8],
}
PRIMARY_ENV = '5_2_6'
SCHEDULES = ['sync', 'seq_ABC']
SEQ_ORDER = (0, 1, 2)
OBSERVERS = ['hard', 'smooth', 'memoryless', 'static', 'leaky']
SIGMA = 0.40
LEAK = 0.20
T_STEPS = 50
CONV_THRESHOLD = 0.01
FULL_SEEDS = list(range(200))
SMOKE_SEEDS = list(range(3))
UNIFORM = np.ones(f.N_STATES) / f.N_STATES
WATCH = [(i, (i + 1) % 3) for i in range(3)]
N_ACTION_U = 4  # R,R,S,I
Z_WILSON = 1.959963984540054

POLICY_CACHE = {
    2: f.construct_policies([f.N_STATES], [f.N_ACTIONS], policy_len=2),
    4: f.construct_policies([f.N_STATES], [f.N_ACTIONS], policy_len=4),
}


# ---------------------------------------------------------------------------
# CRN
# ---------------------------------------------------------------------------
def make_crn(seed, T=T_STEPS):
    ss = np.random.SeedSequence([int(seed), 20260815])
    g_snap, g_act, g_action = [
        np.random.Generator(np.random.PCG64(s)) for s in ss.spawn(3)
    ]
    return {
        'social_snap': g_snap.random((T, 3)),
        'social_act': g_act.random((T, 3)),
        'action_u': g_action.random((T, 3, N_ACTION_U)),
    }


def cat_from_u(p, u):
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    cdf = np.cumsum(p)
    u = float(min(max(u, 0.0), 1.0 - 1e-16))
    return int(np.searchsorted(cdf, u, side='right'))


# ---------------------------------------------------------------------------
# Entropy-matched smoothing
# ---------------------------------------------------------------------------
def vec_entropy(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-16)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def mixture_entropy(eps, n=f.N_STATES):
    cs = (1.0 - eps) + eps / n
    co = eps / n
    return vec_entropy(np.array([cs] + [co] * (n - 1)))


def expected_memoryless_H(sigma=SIGMA):
    A = f.make_A_matrix(sharpness=sigma)
    acc = 0.0
    for s in range(f.N_STATES):
        for o in range(f.N_STATES):
            q = f.infer_states(o, A, UNIFORM)
            acc += float(A[o, s]) * vec_entropy(q)
    return acc / f.N_STATES


def match_eps(h_target, n=f.N_STATES):
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if mixture_entropy(mid, n) < h_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


H_TARGET = expected_memoryless_H(SIGMA)
SMOOTH_EPS = match_eps(H_TARGET)
A_SOCIAL = f.make_A_matrix(sharpness=SIGMA)


def smooth_from_true(s_j, eps=SMOOTH_EPS):
    c = eps * UNIFORM.copy()
    c[int(s_j)] += 1.0 - eps
    return c / c.sum()


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------
def kl_safe(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & (a > 1e-15)
    if not mask.any():
        return 0.0
    val = np.sum(a[mask] * (np.log(a[mask]) - np.log(np.maximum(b[mask], 1e-15))))
    if not np.isscalar(val) and np.size(val) == 0:
        return 0.0
    val = float(val)
    if not np.isfinite(val):
        return 0.0
    return max(val, 0.0)


def calculate_G_safe(A, B, C, qs_current, policies):
    G = np.zeros(len(policies))
    H_A = f.entropy(A)
    for pid, policy in enumerate(policies):
        G_pi = 0.0
        qs_prev = qs_current
        for t in range(policy.shape[0]):
            qs_prev = f.get_expected_states(B, qs_prev, policy[t, 0])
            qo = f.get_expected_observations(A, qs_prev)
            G_pi += float(H_A.dot(qs_prev)) + kl_safe(qo, C)
        G[pid] = G_pi
    return G


def active_inference_u(A, B, C, D, obs_idx, env, policy_len, n_inner, us):
    prior = D
    obs = obs_idx
    policies = POLICY_CACHE[policy_len]
    for k in range(n_inner):
        obs_idx = int(obs)
        qs = f.infer_states(obs_idx, A, prior)
        G = calculate_G_safe(A, B, C, qs, policies)
        Q_pi = f.softmax_np(-G)
        P_u = np.zeros(f.N_ACTIONS)
        for pid, policy in enumerate(policies):
            P_u[int(policy[0, 0])] += Q_pi[pid]
        P_u = f.norm_dist(P_u)
        chosen = cat_from_u(P_u, us[k])
        prior = B[:, :, chosen].dot(qs)
        obs = env.step(f.ACTIONS[chosen])
    obs_idx = int(obs)
    qs = f.infer_states(obs_idx, A, prior)
    dkl = f.kl_divergence(qs, A[obs_idx, :])
    F = dkl - f.log_stable(prior)
    return obs_idx, qs, F


def step_agent(ag, obs_i, envs_i, u4):
    env_r, env_s, env_i = envs_i
    o_r, qs_r, F_r = active_inference_u(
        ag['A_R'], ag['B_R'], ag['C_R'], ag['D_R'], obs_i[0], env_r, 2, 2, u4[0:2])
    o_s, qs_s, F_s = active_inference_u(
        ag['A_S'], ag['B_S'], ag['C_S'], ag['D_S'], obs_i[1], env_s, 4, 1, u4[2:3])
    o_i, qs_i, F_i = active_inference_u(
        ag['A_I'], ag['B_I'], ag['C_I'], ag['D_I'], obs_i[2], env_i, 2, 1, u4[3:4])
    R = f.residual(ag['C_R'], qs_r) + f.residual(ag['C_S'], qs_s) + f.residual(ag['C_I'], qs_i)
    w_r, w_s, w_i = ag['w']
    ag['D_R'] = f.D_update(ag['D_R'], F_r, R, w_r)
    ag['D_S'] = f.D_update(ag['D_S'], F_s, R, w_s)
    ag['D_I'] = f.D_update(ag['D_I'], F_i, R, w_i)
    return [o_r, o_s, o_i]


def infer_q(s_j, d_other, u):
    o = cat_from_u(A_SOCIAL[:, int(s_j)], u)
    return f.infer_states(o, A_SOCIAL, d_other)


def snapshot_q(observer, s_j, d_other, u):
    if observer == 'hard':
        return f.onehot(int(s_j), f.N_STATES)
    if observer == 'smooth':
        return smooth_from_true(s_j)
    return infer_q(s_j, d_other, u)


def _norm_pref(c):
    c = np.maximum(np.asarray(c, dtype=float), 0.0)
    z = c.sum()
    if z <= 0 or not np.isfinite(z):
        return UNIFORM.copy()
    return c / z


def write_C_and_memory(observer, q, s_j, d_other):
    if observer == 'hard':
        return f.onehot(int(s_j), f.N_STATES), d_other
    if observer == 'smooth':
        return smooth_from_true(s_j), d_other
    if observer == 'memoryless':
        return _norm_pref(q), UNIFORM.copy()
    if observer == 'static':
        qn = _norm_pref(q)
        return qn, qn.copy()
    qn = _norm_pref(q)
    leaked = _norm_pref((1.0 - LEAK) * qn + LEAK * UNIFORM)
    return qn, leaked


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
        (f.Agent(env_init[0]), f.Agent(env_init[1]), f.Agent(env_init[2]))
        for _ in range(3)
    ]
    obs = [list(o) for o in f.INIT_OBS]
    d_others = [UNIFORM.copy() for _ in range(3)]
    return agents, envs, obs, d_others


def run_trial(env_init, schedule, observer, crn, T=T_STEPS):
    agents, envs, obs, d_others = init_agents(env_init)
    snap_S = np.zeros((T, 3), dtype=int)
    snap_q = np.zeros((T, 3, f.N_STATES))
    snap_true_j = np.zeros((T, 3), dtype=int)

    for t in range(T):
        S_pre = [int(obs[i][1]) for i in range(3)]
        q_snap = [None, None, None]
        for i, j in WATCH:
            q_snap[i] = snapshot_q(observer, S_pre[j], d_others[i], crn['social_snap'][t, i])
            snap_S[t, i] = S_pre[i]
            snap_q[t, i] = q_snap[i]
            snap_true_j[t, i] = S_pre[j]

        if schedule == 'sync':
            for i, j in WATCH:
                C, d_others[i] = write_C_and_memory(observer, q_snap[i], S_pre[j], d_others[i])
                agents[i]['C_S'] = C
            for i in range(3):
                obs[i] = step_agent(agents[i], obs[i], envs[i], crn['action_u'][t, i])
        else:
            for i in SEQ_ORDER:
                j = (i + 1) % 3
                s_j = int(obs[j][1])
                q_act = snapshot_q(observer, s_j, d_others[i], crn['social_act'][t, i])
                C, d_others[i] = write_C_and_memory(observer, q_act, s_j, d_others[i])
                agents[i]['C_S'] = C
                obs[i] = step_agent(agents[i], obs[i], envs[i], crn['action_u'][t, i])

    return snap_S, snap_q, snap_true_j


# ---------------------------------------------------------------------------
# Metrics (no RNG)
# ---------------------------------------------------------------------------
def time_to_sync(sym_var, thresh=CONV_THRESHOLD):
    for t in range(len(sym_var)):
        if np.all(sym_var[t:] < thresh):
            return t
    return None


def classify_attractor(s_final):
    if np.var(s_final.astype(float)) >= CONV_THRESHOLD:
        return 'no_conv'
    if np.all(s_final == 0):
        return 'S=0'
    if np.all(s_final == 8):
        return 'S=8'
    return 'other_conv'


def acf(x, lag):
    x = np.asarray(x, dtype=float)
    if len(x) <= lag:
        return float('nan')
    a, b = x[:-lag], x[lag:]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def spectral_peak(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0.0
    tot = float(spec.sum())
    if tot <= 0:
        return 0.0, 0.0
    k = int(np.argmax(spec))
    freq = float(np.fft.rfftfreq(len(x))[k])
    return freq, float(spec[k] / tot)


def turning_rate(x):
    d = np.diff(np.asarray(x, dtype=float))
    if len(d) < 2:
        return float('nan')
    return float(np.mean((d[1:] * d[:-1]) < 0))


def period2_score(x):
    x = np.asarray(x, dtype=int)
    if len(x) < 3:
        return float('nan')
    return float(np.mean((x[2:] == x[:-2]) & (x[1:-1] != x[:-2])))


def compute_metrics(snap_S, snap_q, snap_true_j, T=T_STEPS):
    sym_var = np.var(snap_S.astype(float), axis=1)
    tts = time_to_sync(sym_var)
    converged = bool(sym_var[-1] < CONV_THRESHOLD)
    hat = np.argmax(snap_q, axis=2)
    mis = hat != snap_true_j
    if (not converged) or tts is None:
        mis_post = float('nan')
    else:
        mis_post = float(mis[tts:].mean())

    half = slice(T // 2, T)
    nlls, briers = [], []
    for t in range(T // 2, T):
        for i in range(3):
            q = np.maximum(snap_q[t, i], 1e-16)
            q = q / q.sum()
            sj = int(snap_true_j[t, i])
            nlls.append(-math.log(q[sj]))
            e = f.onehot(sj, f.N_STATES)
            briers.append(float(np.sum((q - e) ** 2)))

    switch, acf1, acf2, acf5, turn, p2, freq, share = [], [], [], [], [], [], [], []
    mis_acf1 = []
    for i in range(3):
        s = snap_S[:, i]
        switch.append(float(np.mean(s[1:] != s[:-1])))
        acf1.append(acf(s, 1))
        acf2.append(acf(s, 2))
        acf5.append(acf(s, 5))
        turn.append(turning_rate(s))
        p2.append(period2_score(s))
        fr, sh = spectral_peak(s)
        freq.append(fr)
        share.append(sh)
        mis_acf1.append(acf(mis[:, i].astype(float), 1))

    def nanmean(xs):
        arr = np.asarray(xs, dtype=float)
        arr = arr[~np.isnan(arr)]
        return float(arr.mean()) if arr.size else float('nan')

    return {
        'converged': converged,
        'attractor': classify_attractor(snap_S[-1]),
        'time_to_sync': tts,
        'sym_var_final': float(sym_var[-1]),
        'misID_snap': float(mis.mean()),
        'misID_half': float(mis[half].mean()),
        'misID_post': mis_post,
        'nll_half': float(np.mean(nlls)),
        'brier_half': float(np.mean(briers)),
        'switch_rate': nanmean(switch),
        'acf1': nanmean(acf1),
        'acf2': nanmean(acf2),
        'acf5': nanmean(acf5),
        'mis_acf1': nanmean(mis_acf1),
        'turning_rate': nanmean(turn),
        'period2_score': nanmean(p2),
        'spec_peak_freq': nanmean(freq),
        'spec_peak_share': nanmean(share),
        'final_S': tuple(int(x) for x in snap_S[-1]),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson_ci(x, n, z=Z_WILSON):
    if n <= 0:
        return (float('nan'), float('nan'))
    p = x / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / den
    return (max(0.0, center - half), min(1.0, center + half))


def binom_pmf_half(n, k):
    if k < 0 or k > n:
        return 0.0
    k = min(k, n - k)
    logc = 0.0
    for i in range(k):
        logc += math.log(n - i) - math.log(i + 1)
    return math.exp(logc - n * math.log(2.0))


def mcnemar_exact(n01, n10):
    n = n01 + n10
    if n == 0:
        return 1.0
    p_obs = binom_pmf_half(n, n10)
    s = 0.0
    for i in range(n + 1):
        pi = binom_pmf_half(n, i)
        if pi <= p_obs + 1e-15:
            s += pi
    return float(min(1.0, s))


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def paired_counts(a, b):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n11 = int(np.sum((a == 1) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    return n11, n00, n10, n01


def interaction_perm_p(conv, n_perm=5000, rng_seed=42):
    """
    conv[seed, observer, schedule] in {0,1}, schedule 0=sync 1=seq.
    H0: schedule effect exchangeable across observers.
    stat = variance of observer-wise mean(sync-seq).
    """
    d = conv[:, :, 0].astype(float) - conv[:, :, 1].astype(float)
    obs_mean = d.mean(axis=0)
    stat = float(np.var(obs_mean, ddof=0))
    rng = np.random.RandomState(rng_seed)
    n_seed, n_obs = d.shape
    exceed = 0
    for _ in range(n_perm):
        dp = np.empty_like(d)
        for s in range(n_seed):
            dp[s] = rng.permutation(d[s])
        if float(np.var(dp.mean(axis=0), ddof=0)) >= stat - 1e-15:
            exceed += 1
    return stat, (exceed + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# I/O and report
# ---------------------------------------------------------------------------
def write_results_csv(rows, path):
    fields = [
        'seed', 'env_init', 'schedule', 'observer',
        'converged', 'attractor', 'time_to_sync', 'sym_var_final',
        'misID_snap', 'misID_half', 'misID_post',
        'nll_half', 'brier_half', 'switch_rate',
        'acf1', 'acf2', 'acf5', 'mis_acf1',
        'turning_rate', 'period2_score', 'spec_peak_freq', 'spec_peak_share',
        'final_S_A', 'final_S_B', 'final_S_C',
    ]
    with open(path, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rec = {k: r.get(k, '') for k in fields}
            rec['converged'] = int(r['converged'])
            rec['time_to_sync'] = '' if r['time_to_sync'] is None else r['time_to_sync']
            rec['misID_post'] = '' if np.isnan(r['misID_post']) else f"{r['misID_post']:.6f}"
            rec['final_S_A'], rec['final_S_B'], rec['final_S_C'] = r['final_S']
            for k in fields:
                if isinstance(rec.get(k), float):
                    rec[k] = f"{rec[k]:.6f}"
            w.writerow(rec)


def cell_rows(rows, env, sched, obs):
    return [r for r in rows if r['env_init'] == env and r['schedule'] == sched and r['observer'] == obs]


def conv_vec(rows, env, sched, obs, seeds):
    m = {(r['seed']): int(r['converged']) for r in cell_rows(rows, env, sched, obs)}
    return np.array([m[s] for s in seeds], dtype=int)


def write_summary_csv(rows, seeds, path):
    with open(path, 'w', newline='') as fp:
        w = csv.writer(fp)
        w.writerow([
            'env_init', 'schedule', 'observer', 'n', 'n_conv', 'conv_rate',
            'wilson_lo', 'wilson_hi', 'n_post', 'misID_post_mean',
            'nll_half_mean', 'brier_half_mean', 'switch_rate_mean',
            'acf1_mean', 'mis_acf1_mean', 'turning_rate_mean',
            'period2_mean', 'spec_peak_share_mean',
        ])
        for env in ENV_INITS:
            for sched in SCHEDULES:
                for obs in OBSERVERS:
                    sub = cell_rows(rows, env, sched, obs)
                    n = len(sub)
                    n_conv = sum(int(r['converged']) for r in sub)
                    lo, hi = wilson_ci(n_conv, n)
                    posts = [r['misID_post'] for r in sub if r['converged'] and not np.isnan(r['misID_post'])]
                    w.writerow([
                        env, sched, obs, n, n_conv,
                        f"{n_conv / n:.4f}" if n else '',
                        f"{lo:.4f}", f"{hi:.4f}",
                        len(posts),
                        f"{float(np.mean(posts)):.4f}" if posts else '',
                        f"{np.mean([r['nll_half'] for r in sub]):.4f}",
                        f"{np.mean([r['brier_half'] for r in sub]):.4f}",
                        f"{np.mean([r['switch_rate'] for r in sub]):.4f}",
                        f"{np.mean([r['acf1'] for r in sub]):.4f}",
                        f"{float(np.nanmean([r['mis_acf1'] for r in sub])) if sub else float('nan'):.4f}",
                        f"{np.mean([r['turning_rate'] for r in sub]):.4f}",
                        f"{np.mean([r['period2_score'] for r in sub]):.4f}",
                        f"{np.mean([r['spec_peak_share'] for r in sub]):.4f}",
                    ])


def write_report(rows, seeds, path):
    n = len(seeds)
    lines = []
    lines.append('# Observer × schedule factorial report\n')
    lines.append('主终点是 convergence。Wilson CI 只作区间描述。')
    lines.append('同 seed 比较用 McNemar exact，族内 Holm 校正。')
    lines.append('预先检验 schedule × observer interaction。')
    lines.append('不把 CI 重叠写成“无差异”；不拒绝 H0 只写“未拒绝”。\n')
    lines.append('## 设计\n')
    lines.append(f'- observers: {", ".join(OBSERVERS)}')
    lines.append(f'- schedules: {", ".join(SCHEDULES)}')
    lines.append(f'- env: [5,2,6] 主分析；[8,8,8] 对照')
    lines.append(f'- n={n} seeds / 格，{T_STEPS} steps，CRN 按 seed/t/agent 预生成')
    lines.append(f'- σ={SIGMA}, leak λ={LEAK}, smooth ε={SMOOTH_EPS:.4f} '
                 f'(匹配 memoryless E[H]={H_TARGET:.4f})')
    lines.append('- snapshot 对齐；(NLL/Brier/ACF/振荡) 不消耗机制 RNG')
    lines.append('- post-consensus misID 仅在收敛轨迹上报告\n')

    lines.append('## 1. Convergence（Wilson 95% CI）\n')
    for env in ENV_INITS:
        tag = '主环境' if env == PRIMARY_ENV else '容易对照'
        lines.append(f'### {env}（{tag}）\n')
        lines.append('| observer | sync | seq_ABC |')
        lines.append('|----------|------|---------|')
        for obs in OBSERVERS:
            cells = []
            for sched in SCHEDULES:
                sub = cell_rows(rows, env, sched, obs)
                x = sum(int(r['converged']) for r in sub)
                lo, hi = wilson_ci(x, len(sub))
                cells.append(f'{x}/{len(sub)} ({x/len(sub):.1%}) [{lo:.1%},{hi:.1%}]')
            lines.append(f'| {obs} | {cells[0]} | {cells[1]} |')
        lines.append('')

    # Interaction on primary env
    conv = np.zeros((n, len(OBSERVERS), 2), dtype=int)
    seed_index = {s: i for i, s in enumerate(seeds)}
    for r in rows:
        if r['env_init'] != PRIMARY_ENV:
            continue
        conv[seed_index[r['seed']], OBSERVERS.index(r['observer']), SCHEDULES.index(r['schedule'])] = int(r['converged'])
    stat, p_int = interaction_perm_p(conv)
    lines.append(f'## 2. Interaction（仅 {PRIMARY_ENV}）\n')
    lines.append('H0：各 observer 的 schedule 效应（sync−seq）可交换。')
    lines.append(f'统计量 = Var_o(mean_s[conv_sync−conv_seq]) = {stat:.6f}')
    lines.append(f'块置换 p = {p_int:.4f}（5000 perms，按 seed 置换 observer 标签）')
    if p_int < 0.05:
        lines.append('**预先检验：拒绝可加性，schedule 效应随 observer 变化。随后看 simple effects。**\n')
    else:
        lines.append('**预先检验：未拒绝可加性。仍报告预先指定的 simple effects，但不把未拒绝写成无交互。**\n')

    # Planned McNemar family
    tests = []
    for obs in OBSERVERS:
        a = conv_vec(rows, PRIMARY_ENV, 'sync', obs, seeds)
        b = conv_vec(rows, PRIMARY_ENV, 'seq_ABC', obs, seeds)
        n11, n00, n10, n01 = paired_counts(a, b)
        tests.append({
            'name': f'schedule | {obs} (sync vs seq)',
            'n11': n11, 'n00': n00, 'n10': n10, 'n01': n01,
            'p': mcnemar_exact(n01, n10),
        })
    for sched in SCHEDULES:
        for obs in OBSERVERS:
            if obs == 'hard':
                continue
            a = conv_vec(rows, PRIMARY_ENV, sched, obs, seeds)
            b = conv_vec(rows, PRIMARY_ENV, sched, 'hard', seeds)
            n11, n00, n10, n01 = paired_counts(a, b)
            tests.append({
                'name': f'{obs} vs hard | {sched}',
                'n11': n11, 'n00': n00, 'n10': n10, 'n01': n01,
                'p': mcnemar_exact(n01, n10),
            })
    for sched in SCHEDULES:
        for obs in ('memoryless', 'static', 'leaky'):
            a = conv_vec(rows, PRIMARY_ENV, sched, obs, seeds)
            b = conv_vec(rows, PRIMARY_ENV, sched, 'smooth', seeds)
            n11, n00, n10, n01 = paired_counts(a, b)
            tests.append({
                'name': f'{obs} vs smooth | {sched}',
                'n11': n11, 'n00': n00, 'n10': n10, 'n01': n01,
                'p': mcnemar_exact(n01, n10),
            })
    adj = holm([t['p'] for t in tests])
    for t, p_h in zip(tests, adj):
        t['p_holm'] = float(p_h)

    lines.append(f'## 3. 同 seed McNemar exact + Holm（{PRIMARY_ENV}，{len(tests)} 个预先比较）\n')
    lines.append('| 比较 | n11 | n00 | n10 | n01 | p_raw | p_Holm | 结论 |')
    lines.append('|------|-----|-----|-----|-----|-------|--------|------|')
    for t in tests:
        dec = '拒绝 H0' if t['p_holm'] < 0.05 else '未拒绝 H0'
        lines.append(
            f"| {t['name']} | {t['n11']} | {t['n00']} | {t['n10']} | {t['n01']} | "
            f"{t['p']:.4f} | {t['p_holm']:.4f} | {dec} |"
        )
    lines.append('')
    lines.append('n10 = 左列成功且右列失败；n01 相反。未拒绝 ≠ 证明等价。\n')

    lines.append('## 4. 次要终点（全轨迹，不筛收敛）\n')
    lines.append(f'### {PRIMARY_ENV}\n')
    lines.append('| observer | sched | NLL½ | Brier½ | switch | acf1 | mis_acf1 | turn | p2 | spec_share |')
    lines.append('|----------|-------|------|--------|--------|------|----------|------|----|------------|')
    for obs in OBSERVERS:
        for sched in SCHEDULES:
            sub = cell_rows(rows, PRIMARY_ENV, sched, obs)
            lines.append(
                f"| {obs} | {sched} | "
                f"{np.mean([r['nll_half'] for r in sub]):.3f} | "
                f"{np.mean([r['brier_half'] for r in sub]):.3f} | "
                f"{np.mean([r['switch_rate'] for r in sub]):.3f} | "
                f"{np.mean([r['acf1'] for r in sub]):.3f} | "
                f"{float(np.nanmean([r['mis_acf1'] for r in sub])) if sub else float('nan'):.3f} | "
                f"{np.mean([r['turning_rate'] for r in sub]):.3f} | "
                f"{np.mean([r['period2_score'] for r in sub]):.3f} | "
                f"{np.mean([r['spec_peak_share'] for r in sub]):.3f} |"
            )
    lines.append('')

    lines.append('## 5. 条件性次要终点：post-consensus misID（仅收敛）\n')
    lines.append('| env | observer | sched | n_conv | post-misID mean |')
    lines.append('|-----|----------|-------|--------|-----------------|')
    for env in ENV_INITS:
        for obs in OBSERVERS:
            for sched in SCHEDULES:
                sub = [r for r in cell_rows(rows, env, sched, obs) if r['converged'] and not np.isnan(r['misID_post'])]
                if not sub:
                    val = 'NA'
                else:
                    val = f"{np.mean([r['misID_post'] for r in sub]):.3f}"
                n_c = sum(int(r['converged']) for r in cell_rows(rows, env, sched, obs))
                lines.append(f'| {env} | {obs} | {sched} | {n_c}/{n} | {val} |')
    lines.append('')
    lines.append('此终点以收敛为条件，不作主推断。\n')

    lines.append('## 6. 不能写的话\n')
    lines.append('- 不用 Wilson / bootstrap CI 重叠宣布 observer 或 schedule“无差异”。')
    lines.append('- 未拒绝 Holm 校正后的 H0，只写未拒绝。')
    lines.append('- 不把 leaky/static 写成欲望或大他者。')
    lines.append('- 不写 phase transition。\n')

    lines.append('## 文件\n')
    lines.append('| 文件 | 说明 |')
    lines.append('|------|------|')
    lines.append('| opaque_other_observer_schedule.py | 本脚本 |')
    lines.append('| opaque_other_observer_schedule_results.csv | 每 seed 每格 |')
    lines.append('| opaque_other_observer_schedule_summary.csv | 格子汇总 |')
    lines.append('| opaque_other_observer_schedule_report.md | 本报告 |')

    with open(path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines))


def run_grid(seeds, resume_path=None):
    print(f'H_target={H_TARGET:.4f}  smooth_eps={SMOOTH_EPS:.4f}  leak={LEAK}  n={len(seeds)}',
          flush=True)
    total = len(seeds) * len(ENV_INITS) * len(SCHEDULES) * len(OBSERVERS)
    done = 0
    rows = []
    seen = set()
    if resume_path and os.path.exists(resume_path):
        with open(resume_path, newline='') as fp:
            for rec in csv.DictReader(fp):
                key = (int(rec['seed']), rec['env_init'], rec['schedule'], rec['observer'])
                if key[0] in seeds:
                    seen.add(key)
        print(f'resume: {len(seen)} rows already on disk', flush=True)
    tmp_path = resume_path or os.path.join(OUT_DIR, '_obs_sched_partial.csv')
    write_header = not (resume_path and os.path.exists(resume_path) and seen)
    # append new rows as we go
    out_fp = open(tmp_path, 'a', newline='')
    # header if new
    fields = [
        'seed', 'env_init', 'schedule', 'observer',
        'converged', 'attractor', 'time_to_sync', 'sym_var_final',
        'misID_snap', 'misID_half', 'misID_post',
        'nll_half', 'brier_half', 'switch_rate',
        'acf1', 'acf2', 'acf5', 'mis_acf1',
        'turning_rate', 'period2_score', 'spec_peak_freq', 'spec_peak_share',
        'final_S_A', 'final_S_B', 'final_S_C',
    ]
    writer = csv.DictWriter(out_fp, fieldnames=fields)
    if write_header and not seen:
        writer.writeheader()
        out_fp.flush()

    def flush_row(m):
        rec = {k: m.get(k, '') for k in fields}
        rec['converged'] = int(m['converged'])
        rec['time_to_sync'] = '' if m['time_to_sync'] is None else m['time_to_sync']
        rec['misID_post'] = '' if np.isnan(m['misID_post']) else f"{m['misID_post']:.6f}"
        rec['final_S_A'], rec['final_S_B'], rec['final_S_C'] = m['final_S']
        for k in fields:
            if isinstance(rec.get(k), float):
                rec[k] = f"{rec[k]:.6f}"
        writer.writerow(rec)
        out_fp.flush()

    try:
        for env_label, env_init in ENV_INITS.items():
            for schedule in SCHEDULES:
                for observer in OBSERVERS:
                    n_conv = 0
                    n_cell = 0
                    for seed in seeds:
                        key = (seed, env_label, schedule, observer)
                        if key in seen:
                            done += 1
                            continue
                        crn = make_crn(seed)
                        try:
                            snap_S, snap_q, snap_j = run_trial(env_init, schedule, observer, crn)
                            m = compute_metrics(snap_S, snap_q, snap_j)
                        except Exception as exc:
                            print(f'  ERROR {key}: {type(exc).__name__}: {exc}', flush=True)
                            continue
                        m.update({
                            'seed': seed,
                            'env_init': env_label,
                            'schedule': schedule,
                            'observer': observer,
                        })
                        rows.append(m)
                        flush_row(m)
                        n_conv += int(m['converged'])
                        n_cell += 1
                        done += 1
                    print(f'  {env_label:6s} {schedule:8s} {observer:11s} '
                          f'new {n_cell} conv_new {n_conv}  ({done}/{total})', flush=True)
    finally:
        out_fp.close()

    # reload all rows for this seed set (resume + new)
    all_rows = []
    with open(tmp_path, newline='') as fp:
        for rec in csv.DictReader(fp):
            seed = int(rec['seed'])
            if seed not in seeds:
                continue
            m = {
                'seed': seed,
                'env_init': rec['env_init'],
                'schedule': rec['schedule'],
                'observer': rec['observer'],
                'converged': rec['converged'] in ('1', 'True', 'true'),
                'attractor': rec['attractor'],
                'time_to_sync': None if rec['time_to_sync'] == '' else int(float(rec['time_to_sync'])),
                'sym_var_final': float(rec['sym_var_final']),
                'misID_snap': float(rec['misID_snap']),
                'misID_half': float(rec['misID_half']),
                'misID_post': float('nan') if rec['misID_post'] == '' else float(rec['misID_post']),
                'nll_half': float(rec['nll_half']),
                'brier_half': float(rec['brier_half']),
                'switch_rate': float(rec['switch_rate']),
                'acf1': float(rec['acf1']),
                'acf2': float(rec['acf2']),
                'acf5': float(rec['acf5']),
                'mis_acf1': float(rec['mis_acf1']) if rec['mis_acf1'] not in ('', 'nan') else float('nan'),
                'turning_rate': float(rec['turning_rate']),
                'period2_score': float(rec['period2_score']),
                'spec_peak_freq': float(rec['spec_peak_freq']),
                'spec_peak_share': float(rec['spec_peak_share']),
                'final_S': (int(rec['final_S_A']), int(rec['final_S_B']), int(rec['final_S_C'])),
            }
            all_rows.append(m)
    return all_rows


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'full'
    if mode not in ('smoke', 'full'):
        print('usage: python opaque_other_observer_schedule.py [smoke|full]')
        raise SystemExit(2)
    seeds = SMOKE_SEEDS if mode == 'smoke' else FULL_SEEDS
    print('=' * 70, flush=True)
    print(f'OBSERVER × SCHEDULE  ({mode}, {len(seeds)} seeds)', flush=True)
    print('=' * 70, flush=True)
    tag = '_smoke' if mode == 'smoke' else ''
    res = os.path.join(OUT_DIR, f'opaque_other_observer_schedule_results{tag}.csv')
    rows = run_grid(seeds, resume_path=None if mode == 'smoke' else res)
    summ = os.path.join(OUT_DIR, f'opaque_other_observer_schedule_summary{tag}.csv')
    rep = os.path.join(OUT_DIR, f'opaque_other_observer_schedule_report{tag}.md')
    write_results_csv(rows, res)
    write_summary_csv(rows, seeds, summ)
    write_report(rows, seeds, rep)
    print(f'[Saved] {os.path.basename(res)}', flush=True)
    print(f'[Saved] {os.path.basename(summ)}', flush=True)
    print(f'[Saved] {os.path.basename(rep)}', flush=True)


if __name__ == '__main__':
    main()
