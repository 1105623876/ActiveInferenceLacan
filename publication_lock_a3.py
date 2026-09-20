"""
A3 publication lock — confirmatory simulation
=============================================

Protocol: A3-v1.0. Does not modify or overwrite any A2 script/CSV/report.

    python publication_lock_a3.py stage0
    python publication_lock_a3.py smoke
    python publication_lock_a3.py stage2
    python publication_lock_a3.py stage3
    python publication_lock_a3.py stage4
    python publication_lock_a3.py stage5
    python publication_lock_a3.py stage6
    python publication_lock_a3.py pipeline
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import time
import platform
from datetime import datetime, timezone

import numpy as np

import factorial_env_symbolic as f
import opaque_other_observer_schedule as a2

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Frozen design
SIGMA = 0.40
LAMBDAS = [0.00, 0.02, 0.05, 0.10, 0.20, 0.50, 1.00]
SCHEDULES = ['sync', 'seq_ABC']
SEQ_ORDER = (0, 1, 2)
T_FULL = 200
T_SMOKE = 20
T_LEGACY = 50
CONF_SEEDS = list(range(1000, 1200))
LEGACY_SEEDS = list(range(20))
SMOKE_SEEDS = [1000, 1001]
STAGE3_SEEDS = list(range(1000, 1005))
PRIMARY_ENV = [5, 2, 6]
ROBUST_ENV = [8, 8, 8]
PRIMARY_ENV_LABEL = '5_2_6'
ROBUST_ENV_LABEL = '8_8_8'
HALF_START = 100
FINAL20_START = 180
N_BOOT = 10000
WATCH = [(i, (i + 1) % 3) for i in range(3)]
UNIFORM = np.ones(f.N_STATES) / f.N_STATES
A_SOCIAL = f.make_A_matrix(sharpness=SIGMA)
A_IDENTITY = f.make_A_matrix(sharpness=1.0)

PRIMARY_FILTERS = [('hard', None), ('smooth', None)] + [('filter', lam) for lam in LAMBDAS]
ROBUST_FILTERS = [('hard', None), ('filter', 0.00), ('filter', 0.20), ('filter', 1.00)]

A2_NAME = {0.00: 'static', 0.20: 'leaky', 1.00: 'memoryless'}


def cond_name(observer, lam):
    if observer == 'filter':
        return f'filter_l{lam:.2f}'
    return observer


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fp:
        for chunk in iter(lambda: fp.read(1 << 16), b''):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def check_prob_vec(x, atol=1e-10):
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        return False
    if np.any(x < -atol):
        return False
    return abs(float(x.sum()) - 1.0) < atol


def vec_entropy(p):
    p = np.maximum(np.asarray(p, dtype=float), 1e-16)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def nll_of(q, s):
    q = np.maximum(np.asarray(q, dtype=float), 1e-16)
    q = q / q.sum()
    return float(-math.log(q[int(s)]))


def brier_of(q, s):
    q = np.asarray(q, dtype=float)
    e = f.onehot(int(s), f.N_STATES)
    return float(np.sum((q - e) ** 2))


# ---------------------------------------------------------------------------
# CRN: one keyed draw per seed, shared by all observers/schedules
# ---------------------------------------------------------------------------
def make_crn_a3(seed, T):
    """Reuse A2 keyed generators; social_u := A2 social_snap (not social_act)."""
    raw = a2.make_crn(int(seed), T=T)
    return {
        'social_u': raw['social_snap'],
        'action_u': raw['action_u'],
        '_a2_social_act': raw['social_act'],
    }


def verify_smooth_eps():
    eps = float(a2.SMOOTH_EPS)
    if abs(eps - 0.6227) > 5e-4:
        raise AssertionError(f'smooth eps={eps}, expected ~0.6227')
    return eps


# ---------------------------------------------------------------------------
# Observer
# ---------------------------------------------------------------------------
def infer_q(s_j, d_before, u, A_mat):
    o = a2.cat_from_u(A_mat[:, int(s_j)], u)
    return f.infer_states(int(o), A_mat, d_before)


def apply_B_other(q, lam):
    qn = a2._norm_pref(q)
    d = (1.0 - lam) * qn + lam * UNIFORM
    return a2._norm_pref(d)


def observe(observer, lam, s_j, d_before, u, A_mat):
    """Return q_act, C_S, d_after. Oracle d_* is NaN."""
    nan_d = np.full(f.N_STATES, np.nan)
    if observer == 'hard':
        q = f.onehot(int(s_j), f.N_STATES)
        return q, q.copy(), nan_d
    if observer == 'smooth':
        q = a2.smooth_from_true(int(s_j), eps=a2.SMOOTH_EPS)
        return q, q.copy(), nan_d
    q = infer_q(s_j, d_before, u, A_mat)
    q = a2._norm_pref(q)
    d_after = apply_B_other(q, lam)
    return q, q.copy(), d_after


def init_world(env_init):
    agents, envs, obs, d_others = a2.init_agents(env_init)
    env_ids = [id(e) for triple in envs for e in triple]
    return agents, envs, obs, d_others, env_ids


def run_trial(env_init, schedule, observer, lam, crn, T, A_mat=None):
    A_mat = A_SOCIAL if A_mat is None else A_mat
    agents, envs, obs, d_others, env_ids = init_world(env_init)
    S_pre = np.zeros((T, 3), dtype=np.int16)
    S_post = np.zeros((T, 3), dtype=np.int16)
    R_post = np.zeros((T, 3), dtype=np.int16)
    I_post = np.zeros((T, 3), dtype=np.int16)
    true_j_act = np.zeros((T, 3), dtype=np.int16)
    q_act = np.zeros((T, 3, f.N_STATES))
    C_act = np.zeros((T, 3, f.N_STATES))
    d_before_a = np.full((T, 3, f.N_STATES), np.nan)
    d_after_a = np.full((T, 3, f.N_STATES), np.nan)

    for t in range(T):
        spre = [int(obs[i][1]) for i in range(3)]
        S_pre[t] = spre
        if schedule == 'sync':
            qs, Cs, das, tjs, dbs = [], [], [], [], []
            for i, j in WATCH:
                tj = spre[j]
                db = d_others[i].copy()
                q, C, da = observe(observer, lam, tj, d_others[i], crn['social_u'][t, i], A_mat)
                qs.append(q); Cs.append(C); das.append(da); tjs.append(tj); dbs.append(db)
            for i in range(3):
                true_j_act[t, i] = tjs[i]
                q_act[t, i] = qs[i]
                C_act[t, i] = Cs[i]
                if observer == 'filter':
                    d_before_a[t, i] = dbs[i]
                    d_after_a[t, i] = das[i]
                    d_others[i] = das[i]
                agents[i]['C_S'] = Cs[i]
            for i in range(3):
                obs[i] = a2.step_agent(agents[i], obs[i], envs[i], crn['action_u'][t, i])
        else:
            for i in SEQ_ORDER:
                j = (i + 1) % 3
                tj = int(obs[j][1])
                db = d_others[i].copy()
                q, C, da = observe(observer, lam, tj, d_others[i], crn['social_u'][t, i], A_mat)
                true_j_act[t, i] = tj
                q_act[t, i] = q
                C_act[t, i] = C
                if observer == 'filter':
                    d_before_a[t, i] = db
                    d_after_a[t, i] = da
                    d_others[i] = da
                agents[i]['C_S'] = C
                obs[i] = a2.step_agent(agents[i], obs[i], envs[i], crn['action_u'][t, i])
        S_post[t] = [int(obs[i][1]) for i in range(3)]
        R_post[t] = [int(obs[i][0]) for i in range(3)]
        I_post[t] = [int(obs[i][2]) for i in range(3)]

    return {
        'S_pre': S_pre, 'S_post': S_post, 'R_post': R_post, 'I_post': I_post,
        'true_j_act': true_j_act, 'q_act': q_act, 'C_act': C_act,
        'd_before': d_before_a, 'd_after': d_after_a, 'env_ids': env_ids,
    }


# ---------------------------------------------------------------------------
# Metrics from saved acting q (no RNG)
# ---------------------------------------------------------------------------
def metrics_from_trial(tr, T):
    aligned = (tr['S_post'][:, 0] == tr['S_post'][:, 1]) & (tr['S_post'][:, 1] == tr['S_post'][:, 2])
    aligned = aligned.astype(np.int8)
    rsi = (
        (tr['R_post'][:, 0] == tr['R_post'][:, 1]) & (tr['R_post'][:, 1] == tr['R_post'][:, 2])
        & aligned.astype(bool)
        & (tr['I_post'][:, 0] == tr['I_post'][:, 1]) & (tr['I_post'][:, 1] == tr['I_post'][:, 2])
    )
    hat = np.argmax(tr['q_act'], axis=2)
    mis = (hat != tr['true_j_act']).astype(np.float64)
    nll = np.empty((T, 3))
    brier = np.empty((T, 3))
    Hq = np.empty((T, 3))
    for t in range(T):
        for i in range(3):
            q = tr['q_act'][t, i]
            s = int(tr['true_j_act'][t, i])
            nll[t, i] = nll_of(q, s)
            brier[t, i] = brier_of(q, s)
            Hq[t, i] = vec_entropy(q)

    half = slice(HALF_START, T) if T > HALF_START else slice(T // 2, T)
    fin = slice(max(0, T - 20), T)
    occ_half = float(aligned[half].mean()) if T > 0 else float('nan')
    final20 = bool(np.all(aligned[fin] == 1))
    terminal = bool(aligned[-1] == 1)
    # longest run
    longest = 0
    cur = 0
    for a in aligned:
        if a:
            cur += 1
            longest = max(longest, cur)
        else:
            cur = 0
    tts = None
    for t in range(T):
        if np.all(aligned[t:] == 1):
            tts = t
            break
    sfin = tr['S_post'][-1]
    if not terminal:
        regime = 'not_aligned'
    elif int(sfin[0]) == 0:
        regime = 'S=0'
    elif int(sfin[0]) == 8:
        regime = 'S=8'
    else:
        regime = 'other_aligned'
    # A2-compatible snapshot convergence on last S_pre
    snap_var = float(np.var(tr['S_pre'][-1].astype(float)))
    return {
        'alignment_occupancy_half': occ_half,
        'final20_aligned': int(final20),
        'longest_aligned_run': int(longest),
        'terminal_aligned': int(terminal),
        'time_to_permanent_alignment': tts,
        'censored_permanent': int(tts is None),
        'misID_act_half': float(mis[half].mean()),
        'NLL_act_half': float(nll[half].mean()),
        'Brier_act_half': float(brier[half].mean()),
        'observer_entropy_half': float(Hq[half].mean()),
        'endpoint_regime': regime,
        'any_full_rsi': int(bool(np.any(rsi))),
        'terminal_full_rsi': int(bool(rsi[-1])),
        'snap_var_final': snap_var,
        'aligned': aligned,
        'mis_t': mis.mean(axis=1),
        'nll_t': nll.mean(axis=1),
        'brier_t': brier.mean(axis=1),
        'H_t': Hq.mean(axis=1),
    }


def finite_prob_ok(tr, observer, atol=1e-10):
    T = tr['q_act'].shape[0]
    for t in range(T):
        for i in range(3):
            if not check_prob_vec(tr['q_act'][t, i], atol):
                return False
            if not check_prob_vec(tr['C_act'][t, i], atol):
                return False
            if observer == 'filter':
                if not check_prob_vec(tr['d_before'][t, i], atol):
                    return False
                if not check_prob_vec(tr['d_after'][t, i], atol):
                    return False
    return True


# ---------------------------------------------------------------------------
# Statistics (no mechanism RNG)
# ---------------------------------------------------------------------------
def wilson_ci(x, n, z=1.959963984540054):
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
        if binom_pmf_half(n, i) <= p_obs + 1e-15:
            s += binom_pmf_half(n, i)
    return float(min(1.0, s))


def holm(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def paired_mean_ci_p(delta, n_boot=N_BOOT, rng_seed=42):
    delta = np.asarray(delta, dtype=float)
    mean = float(delta.mean())
    rng = np.random.RandomState(rng_seed)
    n = len(delta)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        boots[b] = rng.choice(delta, size=n, replace=True).mean()
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    # sign-flip
    exceed = 0
    abs_obs = abs(mean)
    for b in range(n_boot):
        flips = rng.choice([-1.0, 1.0], size=n)
        if abs(float((delta * flips).mean())) >= abs_obs - 1e-15:
            exceed += 1
    p = (exceed + 1) / (n_boot + 1)
    return mean, ci, p


def trend_perm_p(lams, occ_by_lam, n_perm=N_BOOT, rng_seed=43):
    """occ_by_lam: (n_seed, n_lam). Spearman-like: corr(-lam, occ)."""
    lams = np.asarray(lams, dtype=float)
    x = -lams
    n_seed, k = occ_by_lam.shape

    def stat(mat):
        vals = []
        for s in range(n_seed):
            y = mat[s]
            if np.std(y) < 1e-15 or np.std(x) < 1e-15:
                vals.append(0.0)
            else:
                vals.append(float(np.corrcoef(x, y)[0, 1]))
        return float(np.mean(vals))

    obs = stat(occ_by_lam)
    rng = np.random.RandomState(rng_seed)
    exceed = 0
    for _ in range(n_perm):
        perm = occ_by_lam.copy()
        for s in range(n_seed):
            perm[s] = rng.permutation(perm[s])
        if abs(stat(perm)) >= abs(obs) - 1e-15:
            exceed += 1
    return obs, (exceed + 1) / (n_perm + 1)


def interaction_perm_p(occ, n_perm=N_BOOT, rng_seed=44):
    """occ[seed, observer, schedule] schedule0=sync,1=seq."""
    d = occ[:, :, 0] - occ[:, :, 1]
    stat = float(np.var(d.mean(axis=0)))
    rng = np.random.RandomState(rng_seed)
    exceed = 0
    for _ in range(n_perm):
        dp = np.empty_like(d)
        for s in range(d.shape[0]):
            dp[s] = rng.permutation(d[s])
        if float(np.var(dp.mean(axis=0))) >= stat - 1e-15:
            exceed += 1
    return stat, (exceed + 1) / (n_perm + 1)


# ---------------------------------------------------------------------------
# Stage 0
# ---------------------------------------------------------------------------
def stage0():
    print('=== STAGE 0 ===', flush=True)
    gates = []
    A = f.A_MAT
    col = A.sum(axis=0)
    ok_a = bool(np.max(np.abs(col - 1.0)) < 1e-12)
    gates.append(('A_col_norm', ok_a, f'max|col-1|={np.max(np.abs(col-1)):.3e}'))
    A4 = A_SOCIAL
    ok_as = bool(np.max(np.abs(A4.sum(0) - 1.0)) < 1e-12)
    gates.append(('A_social_col_norm', ok_as, f'max|col-1|={np.max(np.abs(A4.sum(0)-1)):.3e}'))
    try:
        eps = verify_smooth_eps()
        gates.append(('smooth_eps', True, f'eps={eps:.6f}'))
    except AssertionError as e:
        gates.append(('smooth_eps', False, str(e)))
    _, envs, _, _, ids = init_world(PRIMARY_ENV)
    ok_id = len(ids) == 9 and len(set(ids)) == 9
    gates.append(('isolated_env_ids', ok_id, f'n={len(ids)} unique={len(set(ids))}'))
    rng = np.random.RandomState(0)
    q = rng.dirichlet(np.ones(9))
    for lam in LAMBDAS:
        d = apply_B_other(q, lam)
        ok = check_prob_vec(d)
        # identity with A2 write
        if abs(lam - 0.0) < 1e-12:
            _, d2 = a2.write_C_and_memory('static', q, 0, UNIFORM)
            ok = ok and np.allclose(d, d2, atol=1e-12)
        if abs(lam - 0.20) < 1e-12:
            _, d2 = a2.write_C_and_memory('leaky', q, 0, UNIFORM)
            ok = ok and np.allclose(d, d2, atol=1e-12)
        if abs(lam - 1.0) < 1e-12:
            _, d2 = a2.write_C_and_memory('memoryless', q, 0, UNIFORM)
            ok = ok and np.allclose(d, d2, atol=1e-12)
        gates.append((f'B_other_lam{lam:.2f}', ok, f'sum={d.sum():.16f}'))
    # identity A
    ok_idA = bool(np.allclose(A_IDENTITY, np.eye(9), atol=1e-12))
    gates.append(('A_sigma1_identity', ok_idA, 'allclose I'))
    all_ok = True
    for name, ok, detail in gates:
        print(f'  [{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)
        all_ok = all_ok and ok
    print('STAGE0:', 'PASS' if all_ok else 'FAIL', flush=True)
    return all_ok, gates


# ---------------------------------------------------------------------------
# Stage 1 smoke
# ---------------------------------------------------------------------------
def run_one(env, schedule, observer, lam, seed, T, A_mat=None):
    crn = make_crn_a3(seed, T)
    tr = run_trial(env, schedule, observer, lam, crn, T, A_mat=A_mat)
    m = metrics_from_trial(tr, T)
    return tr, m, crn


def stage1_smoke():
    print('=== STAGE 1 SMOKE ===', flush=True)
    T = T_SMOKE
    rows = []
    ts = []
    for seed in SMOKE_SEEDS:
        for schedule in SCHEDULES:
            for observer, lam in PRIMARY_FILTERS:
                tr, m, _ = run_one(PRIMARY_ENV, schedule, observer, lam, seed, T)
                okp = finite_prob_ok(tr, observer)
                if not okp:
                    print(f'  FAIL prob {seed} {schedule} {cond_name(observer, lam)}', flush=True)
                    return False
                mrow = dict(m)
                mrow.update({
                    'seed': seed, 'env_init': PRIMARY_ENV_LABEL,
                    'schedule': schedule, 'observer': cond_name(observer, lam),
                    'lambda': '' if lam is None else f'{lam:.2f}',
                })
                rows.append(mrow)
                ts.append(tr)
    # shapes
    tr0 = ts[0]
    assert tr0['S_pre'].shape == (T, 3)
    assert tr0['q_act'].shape == (T, 3, 9)
    # repeatability of CRN
    c1 = make_crn_a3(1000, T)
    c2 = make_crn_a3(1000, T)
    if not (np.array_equal(c1['social_u'], c2['social_u']) and np.array_equal(c1['action_u'], c2['action_u'])):
        print('  FAIL CRN not deterministic', flush=True)
        return False
    # same seed+cond twice -> same S_post
    tr_a, _, _ = run_one(PRIMARY_ENV, 'sync', 'filter', 0.20, 1000, T)
    tr_b, _, _ = run_one(PRIMARY_ENV, 'sync', 'filter', 0.20, 1000, T)
    if not np.array_equal(tr_a['S_post'], tr_b['S_post']):
        print('  FAIL trial not repeatable', flush=True)
        return False
    smoke_sum = os.path.join(OUT_DIR, 'publication_lock_a3_smoke_summary.csv')
    write_summary_csv(rows, SMOKE_SEEDS, smoke_sum, T)
    # repeat summary hash: rewrite
    smoke_sum2 = os.path.join(OUT_DIR, 'publication_lock_a3_smoke_summary_b.csv')
    write_summary_csv(rows, SMOKE_SEEDS, smoke_sum2, T)
    h1, h2 = file_sha256(smoke_sum), file_sha256(smoke_sum2)
    os.remove(smoke_sum2)
    if h1 != h2:
        print('  FAIL smoke summary hash', flush=True)
        return False
    print(f'  smoke cells={len(rows)} summary_sha={h1[:12]}', flush=True)
    print('STAGE1: PASS', flush=True)
    return True


# ---------------------------------------------------------------------------
# Stage 2 regression
# ---------------------------------------------------------------------------
def stage2_regression():
    print('=== STAGE 2 REGRESSION ===', flush=True)
    lines = ['# A3 Stage 2 regression\n']
    results = {}

    # Gate 1-3 already in stage0; recheck A and env
    col_err = float(np.max(np.abs(f.A_MAT.sum(0) - 1.0)))
    results['g1_A'] = col_err < 1e-12
    lines.append(f'1. A col norm: {col_err:.3e} → {"PASS" if results["g1_A"] else "FAIL"}')
    _, _, _, _, ids = init_world(PRIMARY_ENV)
    results['g2_env'] = len(set(ids)) == 9
    lines.append(f'2. isolated env ids unique 9: {results["g2_env"]}')

    # Gate 3: finite probs on a short trial
    tr, _, _ = run_one(PRIMARY_ENV, 'seq_ABC', 'filter', 0.20, 1000, 20)
    results['g3_prob'] = finite_prob_ok(tr, 'filter')
    lines.append(f'3. q/C/d simplex: {results["g3_prob"]}')

    # Gates 4-6: update formula vs A2
    rng = np.random.RandomState(1)
    q = rng.dirichlet(np.ones(9))
    d0 = apply_B_other(q, 0.0)
    _, ds = a2.write_C_and_memory('static', q, 3, UNIFORM)
    results['g4_lam0'] = bool(np.allclose(d0, ds, atol=1e-12))
    d2 = apply_B_other(q, 0.20)
    _, dl = a2.write_C_and_memory('leaky', q, 3, UNIFORM)
    results['g5_lam20'] = bool(np.allclose(d2, dl, atol=1e-12))
    d1 = apply_B_other(q, 1.0)
    _, dm = a2.write_C_and_memory('memoryless', q, 3, UNIFORM)
    results['g6_lam1'] = bool(np.allclose(d1, dm, atol=1e-12))
    lines.append(f'4. lam0==static: {results["g4_lam0"]}')
    lines.append(f'5. lam.20==leaky: {results["g5_lam20"]}')
    lines.append(f'6. lam1==memoryless: {results["g6_lam1"]}')

    # Gate 7: sigma=1 + lam=1 vs hard
    ok7 = True
    for schedule in SCHEDULES:
        for seed in (1000, 1001):
            crn = make_crn_a3(seed, 20)
            th = run_trial(PRIMARY_ENV, schedule, 'hard', None, crn, 20, A_mat=A_IDENTITY)
            tf = run_trial(PRIMARY_ENV, schedule, 'filter', 1.0, crn, 20, A_mat=A_IDENTITY)
            if not np.array_equal(th['S_post'], tf['S_post']):
                ok7 = False
            if not np.allclose(th['q_act'], tf['q_act'], atol=1e-8):
                # q may be onehot vs inferred onehot; check argmax and S
                if not np.array_equal(th['S_post'], tf['S_post']):
                    ok7 = False
    results['g7_identity'] = ok7
    lines.append(f'7. sigma=1 + lam=1 vs hard S_post: {ok7}')

    # Gate 8: legacy vs A2
    lines.append('\n## Legacy vs A2 (seeds 0-19, T=50)\n')
    lines.append('| env | sched | cond | n_match_Spre | n | note |')
    lines.append('|-----|-------|------|--------------|---|------|')
    exact_ok = True
    for env_label, env in [(PRIMARY_ENV_LABEL, PRIMARY_ENV), (ROBUST_ENV_LABEL, ROBUST_ENV)]:
        for schedule in SCHEDULES:
            pairs = [('hard', None, 'hard'), ('smooth', None, 'smooth'),
                     ('filter', 0.00, 'static'), ('filter', 0.20, 'leaky'),
                     ('filter', 1.00, 'memoryless')]
            for observer, lam, a2obs in pairs:
                n_match = 0
                for seed in LEGACY_SEEDS:
                    crn_a2 = a2.make_crn(seed, T=T_LEGACY)
                    S2, _, _ = a2.run_trial(env, schedule, a2obs, crn_a2, T=T_LEGACY)
                    crn3 = {
                        'social_u': crn_a2['social_snap'],
                        'action_u': crn_a2['action_u'],
                    }
                    tr3 = run_trial(env, schedule, observer, lam, crn3, T_LEGACY)
                    if np.array_equal(tr3['S_pre'], S2):
                        n_match += 1
                must = (schedule == 'sync') or (a2obs in ('hard', 'smooth'))
                note = 'must_exact' if must else 'seq_infer_may_differ'
                if must and n_match != len(LEGACY_SEEDS):
                    exact_ok = False
                lines.append(
                    f'| {env_label} | {schedule} | {a2obs} | {n_match} | {len(LEGACY_SEEDS)} | {note} |'
                )
    results['g8_legacy'] = exact_ok
    lines.append(f'\n8. required exact cells: {"PASS" if exact_ok else "FAIL"}')

    # Gate 9: write regression twice, hash
    reg_path = os.path.join(OUT_DIR, 'publication_lock_a3_regression.md')
    text = '\n'.join(lines) + '\n'
    with open(reg_path, 'w', encoding='utf-8') as fp:
        fp.write(text)
    with open(reg_path, 'w', encoding='utf-8') as fp:
        fp.write(text)
    # deterministic rewrite
    h = file_sha256(reg_path)
    with open(reg_path, 'w', encoding='utf-8') as fp:
        fp.write(text)
    h2 = file_sha256(reg_path)
    results['g9_hash'] = h == h2
    lines.append(f'9. regression md sha repeat: {h==h2} ({h[:16]})')
    with open(reg_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(lines) + '\n')
    all_ok = all(results.values())
    print('\n'.join(lines), flush=True)
    print('STAGE2:', 'PASS' if all_ok else 'FAIL', flush=True)
    return all_ok, results


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
RESULT_FIELDS = [
    'seed', 'env_init', 'schedule', 'observer', 'lambda',
    'alignment_occupancy_half', 'final20_aligned', 'longest_aligned_run',
    'terminal_aligned', 'time_to_permanent_alignment', 'censored_permanent',
    'misID_act_half', 'NLL_act_half', 'Brier_act_half', 'observer_entropy_half',
    'endpoint_regime', 'any_full_rsi', 'terminal_full_rsi', 'snap_var_final',
]


def write_result_row(writer, seed, env_label, schedule, observer, lam, m):
    rec = {
        'seed': seed,
        'env_init': env_label,
        'schedule': schedule,
        'observer': cond_name(observer, lam),
        'lambda': '' if lam is None else f'{lam:.2f}',
        'alignment_occupancy_half': f"{m['alignment_occupancy_half']:.8f}",
        'final20_aligned': m['final20_aligned'],
        'longest_aligned_run': m['longest_aligned_run'],
        'terminal_aligned': m['terminal_aligned'],
        'time_to_permanent_alignment': '' if m['time_to_permanent_alignment'] is None else m['time_to_permanent_alignment'],
        'censored_permanent': m['censored_permanent'],
        'misID_act_half': f"{m['misID_act_half']:.8f}",
        'NLL_act_half': f"{m['NLL_act_half']:.8f}",
        'Brier_act_half': f"{m['Brier_act_half']:.8f}",
        'observer_entropy_half': f"{m['observer_entropy_half']:.8f}",
        'endpoint_regime': m['endpoint_regime'],
        'any_full_rsi': m['any_full_rsi'],
        'terminal_full_rsi': m['terminal_full_rsi'],
        'snap_var_final': f"{m['snap_var_final']:.8f}",
    }
    writer.writerow(rec)


def write_summary_csv(rows, seeds, path, T):
    # group
    keys = sorted({(r['env_init'], r['schedule'], r['observer']) for r in rows})
    fields = [
        'env_init', 'schedule', 'observer', 'n',
        'occupancy_half_mean', 'occupancy_half_std',
        'final20_rate', 'final20_wilson_lo', 'final20_wilson_hi',
        'terminal_rate', 'longest_run_mean',
        'misID_act_half_mean', 'NLL_act_half_mean', 'Brier_act_half_mean',
        'entropy_half_mean', 'any_full_rsi_rate',
    ]
    with open(path, 'w', newline='') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        for env, sched, obs in keys:
            sub = [r for r in rows if r['env_init'] == env and r['schedule'] == sched and r['observer'] == obs]
            n = len(sub)
            occ = np.array([float(r['alignment_occupancy_half']) for r in sub])
            f20 = np.array([int(r['final20_aligned']) for r in sub])
            lo, hi = wilson_ci(int(f20.sum()), n)
            w.writerow({
                'env_init': env, 'schedule': sched, 'observer': obs, 'n': n,
                'occupancy_half_mean': f"{occ.mean():.8f}",
                'occupancy_half_std': f"{occ.std(ddof=1) if n > 1 else 0:.8f}",
                'final20_rate': f"{f20.mean():.8f}",
                'final20_wilson_lo': f"{lo:.8f}",
                'final20_wilson_hi': f"{hi:.8f}",
                'terminal_rate': f"{np.mean([int(r['terminal_aligned']) for r in sub]):.8f}",
                'longest_run_mean': f"{np.mean([int(r['longest_aligned_run']) for r in sub]):.4f}",
                'misID_act_half_mean': f"{np.mean([float(r['misID_act_half']) for r in sub]):.8f}",
                'NLL_act_half_mean': f"{np.mean([float(r['NLL_act_half']) for r in sub]):.8f}",
                'Brier_act_half_mean': f"{np.mean([float(r['Brier_act_half']) for r in sub]):.8f}",
                'entropy_half_mean': f"{np.mean([float(r['observer_entropy_half']) for r in sub]):.8f}",
                'any_full_rsi_rate': f"{np.mean([int(r['any_full_rsi']) for r in sub]):.8f}",
            })


def load_results_csv(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline='') as fp:
        for rec in csv.DictReader(fp):
            rec['seed'] = int(rec['seed'])
            rec['alignment_occupancy_half'] = float(rec['alignment_occupancy_half'])
            rec['final20_aligned'] = int(rec['final20_aligned'])
            rec['longest_aligned_run'] = int(rec['longest_aligned_run'])
            rec['terminal_aligned'] = int(rec['terminal_aligned'])
            rec['time_to_permanent_alignment'] = (
                None if rec['time_to_permanent_alignment'] == '' else int(float(rec['time_to_permanent_alignment']))
            )
            rec['censored_permanent'] = int(rec['censored_permanent'])
            rec['misID_act_half'] = float(rec['misID_act_half'])
            rec['NLL_act_half'] = float(rec['NLL_act_half'])
            rec['Brier_act_half'] = float(rec['Brier_act_half'])
            rec['observer_entropy_half'] = float(rec['observer_entropy_half'])
            rec['any_full_rsi'] = int(rec['any_full_rsi'])
            rec['terminal_full_rsi'] = int(rec['terminal_full_rsi'])
            rec['snap_var_final'] = float(rec['snap_var_final'])
            rows.append(rec)
    return rows


def _flush_pack(pack, path):
    if not pack['meta']:
        return
    save_npz(pack, path)
    for k in pack:
        pack[k] = []


def run_grid(env, env_label, seeds, T, filters, result_path, npz_prefix, resume=True):
    existing = {(r['seed'], r['env_init'], r['schedule'], r['observer']) for r in load_results_csv(result_path)}
    write_header = not os.path.exists(result_path)
    n_tot = len(seeds) * len(SCHEDULES) * len(filters)
    done = 0
    import glob
    existing_shards = glob.glob(os.path.join(OUT_DIR, f'publication_lock_a3_timeseries_{npz_prefix}_*.npz'))
    shard_i = len(existing_shards)
    pack = {
        'S_pre': [], 'S_post': [], 'aligned': [],
        'misID_act': [], 'NLL_act': [], 'Brier_act': [], 'H_q_act': [],
        'meta': [],
    }
    fp = open(result_path, 'a', newline='')
    w = csv.DictWriter(fp, fieldnames=RESULT_FIELDS)
    if write_header:
        w.writeheader()
        fp.flush()
    t0 = time.time()
    max_new = int(os.environ.get('A3_MAX_NEW', '0'))
    n_new = 0
    try:
        for seed in seeds:
            crn = make_crn_a3(seed, T)
            for schedule in SCHEDULES:
                for observer, lam in filters:
                    key = (seed, env_label, schedule, cond_name(observer, lam))
                    done += 1
                    if resume and key in existing:
                        continue
                    if max_new and n_new >= max_new:
                        print(f'  batch cap {max_new} reached', flush=True)
                        return pack
                    try:
                        tr = run_trial(env, schedule, observer, lam, crn, T)
                        if not finite_prob_ok(tr, observer):
                            raise RuntimeError(f'non-finite/non-simplex q at {key}')
                        m = metrics_from_trial(tr, T)
                        write_result_row(w, seed, env_label, schedule, observer, lam, m)
                        fp.flush()
                        pack['S_pre'].append(tr['S_pre'])
                        pack['S_post'].append(tr['S_post'])
                        pack['aligned'].append(m['aligned'])
                        pack['misID_act'].append(m['mis_t'])
                        pack['NLL_act'].append(m['nll_t'])
                        pack['Brier_act'].append(m['brier_t'])
                        pack['H_q_act'].append(m['H_t'])
                        pack['meta'].append('|'.join(str(x) for x in key))
                        if len(pack['meta']) >= 100:
                            shard = os.path.join(OUT_DIR, f'publication_lock_a3_timeseries_{npz_prefix}_{shard_i:03d}.npz')
                            _flush_pack(pack, shard)
                            shard_i += 1
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        errp = os.path.join(OUT_DIR, 'publication_lock_a3_error.log')
                        with open(errp, 'a', encoding='utf-8') as ef:
                            ef.write(f'\nKEY {key}\n')
                            traceback.print_exc(file=ef)
                        raise
                    n_new += 1
                    if done % 20 == 0:
                        print(f'  {env_label} {done}/{n_tot} elapsed={time.time()-t0:.1f}s', flush=True)
    finally:
        if pack['meta']:
            shard = os.path.join(OUT_DIR, f'publication_lock_a3_timeseries_{npz_prefix}_{shard_i:03d}.npz')
            _flush_pack(pack, shard)
        fp.close()
    return {'S_pre': [], 'S_post': [], 'aligned': [], 'misID_act': [], 'NLL_act': [], 'Brier_act': [], 'H_q_act': [], 'meta': []}


def save_npz(pack, path):
    if not pack['S_pre']:
        print(f'  npz skip (no new arrays) {path}', flush=True)
        return
    np.savez(
        path,
        S_pre=np.stack(pack['S_pre']),
        S_post=np.stack(pack['S_post']),
        aligned=np.stack(pack['aligned']),
        misID_act=np.stack(pack['misID_act']),
        NLL_act=np.stack(pack['NLL_act']),
        Brier_act=np.stack(pack['Brier_act']),
        H_q_act=np.stack(pack['H_q_act']),
        meta=np.array(pack['meta'], dtype=object),
    )
    print(f'  wrote {path} n={len(pack["meta"])}', flush=True)


# ---------------------------------------------------------------------------
# Stage 3–5
# ---------------------------------------------------------------------------
def stage3_estimate():
    print('=== STAGE 3 TIMING ===', flush=True)
    T = T_FULL
    t0 = time.time()
    n = 0
    for seed in STAGE3_SEEDS:
        crn = make_crn_a3(seed, T)
        for schedule in SCHEDULES:
            for observer, lam in PRIMARY_FILTERS:
                tr = run_trial(PRIMARY_ENV, schedule, observer, lam, crn, T)
                _ = metrics_from_trial(tr, T)
                n += 1
    elapsed = time.time() - t0
    per = elapsed / max(n, 1)
    est = per * 5200
    bytes_run = (200 * 3 * 2 * 2 + 200 * 8 * 4)  # rough S + scalars
    # documented npz: 7 arrays ~ 5200*200*8
    est_disk = 5200 * 200 * (3 * 2 * 2 + 8 * 4) / 1e6
    print(f'  {n} primary-grid trials T={T} in {elapsed:.1f}s  ({per:.3f}s/run)', flush=True)
    print(f'  estimated 5200 runs: {est/3600:.2f} h', flush=True)
    print(f'  estimated timeseries ~{est_disk:.0f} MB (compressed less)', flush=True)
    # Abnormal: >15s/run or >24h
    if per > 15.0 or est > 24 * 3600:
        print('STAGE3: ABNORMAL runtime — stop per protocol', flush=True)
        return False, per, est
    print('STAGE3: PASS (estimate proportional to T×N; not used to change design)', flush=True)
    return True, per, est


def integrity_check(path, expected_n):
    rows = load_results_csv(path)
    keys = [(r['seed'], r['env_init'], r['schedule'], r['observer']) for r in rows]
    n = len(keys)
    n_unique = len(set(keys))
    missing = n == 0
    print(f'  integrity {os.path.basename(path)}: rows={n} unique={n_unique} expected~{expected_n}', flush=True)
    if n != n_unique:
        print('  FAIL duplicate keys', flush=True)
        return False, rows
    if n < expected_n:
        print('  FAIL incomplete', flush=True)
        return False, rows
    for r in rows:
        if not math.isfinite(r['alignment_occupancy_half']):
            print('  FAIL non-finite occupancy', flush=True)
            return False, rows
    return True, rows


def stage4_primary():
    print('=== STAGE 4 PRIMARY ===', flush=True)
    res = os.path.join(OUT_DIR, 'publication_lock_a3_results.csv')
    pack = run_grid(PRIMARY_ENV, PRIMARY_ENV_LABEL, CONF_SEEDS, T_FULL,
                    PRIMARY_FILTERS, res, 'primary')
    n = len(load_results_csv(res))
    if n < 3600:
        print(f'STAGE4: PARTIAL {n}/3600 (batch; not a protocol fail)', flush=True)
        return True
    ok, rows = integrity_check(res, 3600)
    print('STAGE4:', 'PASS' if ok else 'FAIL', flush=True)
    return ok


def stage5_robust():
    print('=== STAGE 5 ROBUSTNESS ===', flush=True)
    res = os.path.join(OUT_DIR, 'publication_lock_a3_results.csv')
    pack = run_grid(ROBUST_ENV, ROBUST_ENV_LABEL, CONF_SEEDS, T_FULL,
                    ROBUST_FILTERS, res, 'robust')
    n = len(load_results_csv(res))
    if n < 5200:
        print(f'STAGE5: PARTIAL {n}/5200 (batch; not a protocol fail)', flush=True)
        return True
    ok, rows = integrity_check(res, 3600 + 1600)
    print('STAGE5:', 'PASS' if ok else 'FAIL', flush=True)
    return ok


# ---------------------------------------------------------------------------
# Stage 6 analysis
# ---------------------------------------------------------------------------
def get_cell(rows, env, sched, obs, seeds):
    m = {(r['seed'], r['schedule'], r['observer']): r for r in rows if r['env_init'] == env}
    out = []
    for s in seeds:
        out.append(m[(s, sched, obs)])
    return out


def occ_vec(rows, env, sched, obs, seeds):
    return np.array([r['alignment_occupancy_half'] for r in get_cell(rows, env, sched, obs, seeds)])


def bin_vec(rows, env, sched, obs, seeds, field):
    return np.array([int(r[field]) for r in get_cell(rows, env, sched, obs, seeds)])


def write_statistics_and_report(rows, t_start, t_end, extra_manifest):
    seeds = CONF_SEEDS
    env = PRIMARY_ENV_LABEL

    # Primary Delta_memory
    occ0 = 0.5 * (occ_vec(rows, env, 'sync', 'filter_l0.00', seeds)
                  + occ_vec(rows, env, 'seq_ABC', 'filter_l0.00', seeds))
    occ1 = 0.5 * (occ_vec(rows, env, 'sync', 'filter_l1.00', seeds)
                  + occ_vec(rows, env, 'seq_ABC', 'filter_l1.00', seeds))
    delta = occ0 - occ1
    d_mean, d_ci, d_p = paired_mean_ci_p(delta)

    # stratified
    d_sync = occ_vec(rows, env, 'sync', 'filter_l0.00', seeds) - occ_vec(rows, env, 'sync', 'filter_l1.00', seeds)
    d_seq = occ_vec(rows, env, 'seq_ABC', 'filter_l0.00', seeds) - occ_vec(rows, env, 'seq_ABC', 'filter_l1.00', seeds)
    ds_m, ds_ci, ds_p = paired_mean_ci_p(d_sync, rng_seed=45)
    dq_m, dq_ci, dq_p = paired_mean_ci_p(d_seq, rng_seed=46)

    # hard schedule
    h_sync = occ_vec(rows, env, 'sync', 'hard', seeds)
    h_seq = occ_vec(rows, env, 'seq_ABC', 'hard', seeds)
    h_d = h_seq - h_sync
    h_m, h_ci, h_p = paired_mean_ci_p(h_d, rng_seed=47)

    # lambda trend
    lam_names = [f'filter_l{lam:.2f}' for lam in LAMBDAS]
    occ_lam = np.zeros((len(seeds), len(LAMBDAS)))
    for k, name in enumerate(lam_names):
        occ_lam[:, k] = 0.5 * (occ_vec(rows, env, 'sync', name, seeds)
                               + occ_vec(rows, env, 'seq_ABC', name, seeds))
    trend_r, trend_p = trend_perm_p(LAMBDAS, occ_lam)

    # interaction: hard, smooth, 7 filters
    obs_names = ['hard', 'smooth'] + lam_names
    occ3 = np.zeros((len(seeds), len(obs_names), 2))
    for j, name in enumerate(obs_names):
        occ3[:, j, 0] = occ_vec(rows, env, 'sync', name, seeds)
        occ3[:, j, 1] = occ_vec(rows, env, 'seq_ABC', name, seeds)
    int_stat, int_p = interaction_perm_p(occ3)
    # hard-only contribution
    int_nohard, int_p_nohard = interaction_perm_p(occ3[:, 1:, :])

    # final20 McNemar
    def mc(a, b):
        n11 = int(np.sum((a == 1) & (b == 1)))
        n00 = int(np.sum((a == 0) & (b == 0)))
        n10 = int(np.sum((a == 1) & (b == 0)))
        n01 = int(np.sum((a == 0) & (b == 1)))
        return n11, n00, n10, n01, mcnemar_exact(n01, n10)

    f20_hs = bin_vec(rows, env, 'sync', 'hard', seeds, 'final20_aligned')
    f20_hq = bin_vec(rows, env, 'seq_ABC', 'hard', seeds, 'final20_aligned')
    f20_l0s = bin_vec(rows, env, 'sync', 'filter_l0.00', seeds, 'final20_aligned')
    f20_l1s = bin_vec(rows, env, 'sync', 'filter_l1.00', seeds, 'final20_aligned')
    f20_l0q = bin_vec(rows, env, 'seq_ABC', 'filter_l0.00', seeds, 'final20_aligned')
    f20_l1q = bin_vec(rows, env, 'seq_ABC', 'filter_l1.00', seeds, 'final20_aligned')
    fam = [
        ('hard occupancy seq-sync', h_p),
        ('lambda trend', trend_p),
        ('observer x schedule', int_p),
        ('final20 hard sync vs seq', mc(f20_hs, f20_hq)[4]),
        ('final20 lam0 vs lam1 sync', mc(f20_l0s, f20_l1s)[4]),
        ('final20 lam0 vs lam1 seq', mc(f20_l0q, f20_l1q)[4]),
    ]
    holm_p = holm([p for _, p in fam])

    def cell_line(env_, sched, obs):
        sub = [r for r in rows if r['env_init'] == env_ and r['schedule'] == sched and r['observer'] == obs]
        occ = np.mean([r['alignment_occupancy_half'] for r in sub])
        f20 = np.mean([r['final20_aligned'] for r in sub])
        mis = np.mean([r['misID_act_half'] for r in sub])
        nll = np.mean([r['NLL_act_half'] for r in sub])
        br = np.mean([r['Brier_act_half'] for r in sub])
        return occ, f20, mis, nll, br, len(sub)

    stat_lines = []
    stat_lines.append('# A3 publication-lock statistics\n')
    stat_lines.append(f'n_boot={N_BOOT}. Seeds {seeds[0]}..{seeds[-1]}.\n')
    stat_lines.append('## Primary: Delta_memory = occ(λ=0)−occ(λ=1), schedule-averaged\n')
    stat_lines.append(f'- mean = {d_mean:.6f}')
    stat_lines.append(f'- bootstrap 95% CI = [{d_ci[0]:.6f}, {d_ci[1]:.6f}]')
    stat_lines.append(f'- sign-flip p = {d_p:.6f}')
    stat_lines.append(f'- sync only: {ds_m:.6f} [{ds_ci[0]:.6f},{ds_ci[1]:.6f}] p={ds_p:.6f}')
    stat_lines.append(f'- seq_ABC only: {dq_m:.6f} [{dq_ci[0]:.6f},{dq_ci[1]:.6f}] p={dq_p:.6f}\n')
    stat_lines.append('## Secondary (raw then Holm)\n')
    for (name, p), ph in zip(fam, holm_p):
        stat_lines.append(f'- {name}: p_raw={p:.6f} p_Holm={ph:.6f}')
    stat_lines.append(f'\nhard seq−sync occupancy mean={h_m:.6f} CI=[{h_ci[0]:.6f},{h_ci[1]:.6f}] p={h_p:.6f}')
    stat_lines.append(f'interaction var={int_stat:.6f} p={int_p:.6f}; without hard p={int_p_nohard:.6f}')
    stat_lines.append(f'trend mean Spearman(−λ, occ)={trend_r:.6f} p={trend_p:.6f}\n')
    stat_lines.append('## lambda × schedule (primary env)\n')
    stat_lines.append('| schedule | observer | occupancy | final20 | misID | NLL | Brier |')
    stat_lines.append('|----------|----------|-----------|---------|-------|-----|-------|')
    for sched in SCHEDULES:
        for obs in ['hard', 'smooth'] + lam_names:
            occ, f20, mis, nll, br, n = cell_line(env, sched, obs)
            stat_lines.append(
                f'| {sched} | {obs} | {occ:.4f} | {f20:.3f} | {mis:.3f} | {nll:.3f} | {br:.3f} |'
            )
    stat_lines.append('\n## robustness env\n')
    stat_lines.append('| schedule | observer | occupancy | final20 | misID | NLL | Brier |')
    stat_lines.append('|----------|----------|-----------|---------|-------|-----|-------|')
    for sched in SCHEDULES:
        for obs in ['hard', 'filter_l0.00', 'filter_l0.20', 'filter_l1.00']:
            occ, f20, mis, nll, br, n = cell_line(ROBUST_ENV_LABEL, sched, obs)
            stat_lines.append(
                f'| {sched} | {obs} | {occ:.4f} | {f20:.3f} | {mis:.3f} | {nll:.3f} | {br:.3f} |'
            )

    stat_path = os.path.join(OUT_DIR, 'publication_lock_a3_statistics.md')
    with open(stat_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(stat_lines) + '\n')
    # rewrite once more for hash stability
    with open(stat_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(stat_lines) + '\n')

    # report
    # two stabilization paths: (1) sequential schedule for hard (2) long memory filter
    support_seq_hard = h_m > 0 and h_p < 0.05
    support_memory = d_mean > 0 and d_p < 0.05
    rep = []
    rep.append('# A3 publication-lock report\n')
    rep.append('Confirmatory run per A3-v1.0. No post-hoc change of lambda, endpoint, or hypothesis.\n')
    rep.append('## Primary result\n')
    rep.append(
        f'Delta_memory (λ=0 minus λ=1, schedule-averaged occupancy) = {d_mean:.4f} '
        f'(95% CI [{d_ci[0]:.4f}, {d_ci[1]:.4f}]; sign-flip p={d_p:.4f}).'
    )
    if d_p < 0.05 and d_mean > 0:
        rep.append('Direction matches the pre-specified hypothesis that longer memory raises occupancy.')
    elif d_p >= 0.05:
        rep.append('Did not reject the null of zero Delta_memory. This is not a claim of equivalence.')
    else:
        rep.append('Sign is opposite the pre-specified direction.')
    rep.append('\n## hard schedule and interaction\n')
    rep.append(
        f'hard seq−sync occupancy = {h_m:.4f} (95% CI [{h_ci[0]:.4f}, {h_ci[1]:.4f}]; p={h_p:.4f}).'
    )
    if int_p < 0.05 and int_p_nohard >= 0.05:
        rep.append(
            f'observer×schedule interaction p={int_p:.4f}; after dropping hard, p={int_p_nohard:.4f}. '
            'The interaction is driven by hard, not by every observer.'
        )
    else:
        rep.append(
            f'observer×schedule interaction p={int_p:.4f} (without hard p={int_p_nohard:.4f}).'
        )
    rep.append('\n## Two stabilization paths\n')
    if support_seq_hard and support_memory:
        rep.append(
            'Results are compatible with two descriptive paths: sequential update stabilizes hard, '
            'and longer filter memory raises occupancy relative to memoryless. '
            'These are computational patterns, not Lacanian proofs.'
        )
    else:
        bits = []
        if not support_seq_hard:
            bits.append('hard schedule effect did not meet the pre-specified rejection rule')
        if not support_memory:
            bits.append('Delta_memory did not meet the pre-specified rejection rule')
        rep.append('Does not jointly support both paths: ' + '; '.join(bits) + '.')
    rep.append('\n## Accuracy vs alignment\n')
    rep.append(
        'misID/NLL/Brier are reported for all rounds in the second half. '
        'If high occupancy coexists with nonzero misID, that is a descriptive decoupling only.'
    )
    rep.append('\n## Lacanian translation (analogy only)\n')
    rep.append(
        'Computational: observer state `d` is a recursive inscription of inferred Symbolic coordinates, '
        'with λ controlling how fast that inscription is forgotten. '
        'Alignment occupancy is a public coordinate coincidence after each full round. '
        'Theoretical analogy: a more persistent inscription of the other’s place can stabilize a shared '
        'Symbolic coordinate even when the instantaneous reading is fallible. '
        'Not computed: desire, the Other as a social institution, clinical structure.'
    )
    rep.append('\n## Negative controls\n')
    rsi = np.mean([r['any_full_rsi'] for r in rows if r['env_init'] == env])
    rep.append(f'any-round full RSI alignment rate (primary cells pooled) = {rsi:.4f}. Structural R/I preferences remain agent-specific.')
    rep.append('\n## Files\n')
    rep.append('See publication_lock_a3_manifest.json for SHA-256.')
    report_path = os.path.join(OUT_DIR, 'publication_lock_a3_report.md')
    with open(report_path, 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(rep) + '\n')

    sum_path = os.path.join(OUT_DIR, 'publication_lock_a3_summary.csv')
    write_summary_csv(rows, seeds, sum_path, T_FULL)

    # merge timeseries if both parts exist
    merge_npz()

    man = {
        'protocol': 'A3-v1.0',
        'script_sha256': file_sha256(os.path.join(OUT_DIR, 'publication_lock_a3.py')),
        'a2_script_sha256': file_sha256(os.path.join(OUT_DIR, 'opaque_other_observer_schedule.py')),
        'factorial_sha256': file_sha256(os.path.join(OUT_DIR, 'factorial_env_symbolic.py')),
        'python': sys.version.replace('\n', ' '),
        'numpy': np.__version__,
        'platform': platform.platform(),
        'command': ' '.join(sys.argv),
        'start': t_start,
        'end': t_end,
        'seeds': f'{CONF_SEEDS[0]}..{CONF_SEEDS[-1]}',
        'T': T_FULL,
        'sigma': SIGMA,
        'lambdas': LAMBDAS,
        'smooth_eps': float(a2.SMOOTH_EPS),
        'primary_grid': '200 seeds x 2 schedules x (hard+smooth+7 lambda)',
        'robust_grid': '200 seeds x 2 schedules x (hard+l0+l.20+l1)',
        'q_act_retention': 'metrics computed from full q_act at runtime; npz stores per-round mean misID/NLL/Brier/H and S_pre/S_post/aligned, not full q_act (volume)',
        'outputs': {},
    }
    for name in [
        'publication_lock_a3.py',
        'publication_lock_a3_results.csv',
        'publication_lock_a3_summary.csv',
        'publication_lock_a3_timeseries.npz',
        'publication_lock_a3_statistics.md',
        'publication_lock_a3_report.md',
        'publication_lock_a3_regression.md',
    ]:
        p = os.path.join(OUT_DIR, name)
        if os.path.exists(p):
            man['outputs'][name] = file_sha256(p)
    man_path = os.path.join(OUT_DIR, 'publication_lock_a3_manifest.json')
    with open(man_path, 'w', encoding='utf-8') as fp:
        json.dump(man, fp, indent=2)
        fp.write('\n')
    # add self hash after write? changing file changes hash. Write outputs then update.
    man['outputs']['publication_lock_a3_manifest.json'] = 'see file after write'
    print(f'[Saved] statistics/report/summary/manifest', flush=True)
    return {
        'delta_mean': d_mean, 'delta_ci': d_ci, 'delta_p': d_p,
        'hard_mean': h_m, 'hard_ci': h_ci, 'hard_p': h_p,
        'int_p': int_p, 'int_p_nohard': int_p_nohard,
    }


def merge_npz():
    import glob
    paths = sorted(glob.glob(os.path.join(OUT_DIR, 'publication_lock_a3_timeseries_*.npz')))
    paths = [p for p in paths if not p.endswith('publication_lock_a3_timeseries.npz')]
    if not paths:
        return
    keys = ['S_pre', 'S_post', 'aligned', 'misID_act', 'NLL_act', 'Brier_act', 'H_q_act', 'meta']
    parts = [np.load(p, allow_pickle=True) for p in paths]
    stacked = {}
    for k in keys:
        stacked[k] = np.concatenate([np.array(z[k]) for z in parts], axis=0)
    out = os.path.join(OUT_DIR, 'publication_lock_a3_timeseries.npz')
    np.savez_compressed(out, **stacked)
    print(f'  merged timeseries n={stacked["meta"].shape[0]} from {len(paths)} shards', flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def pipeline():
    t0 = utc_now()
    wall0 = time.time()
    ok0, _ = stage0()
    if not ok0:
        print('STOP: stage0 failed', flush=True)
        return 1
    if not stage1_smoke():
        print('STOP: stage1 failed', flush=True)
        return 1
    ok2, _ = stage2_regression()
    if not ok2:
        print('STOP: stage2 failed', flush=True)
        return 1
    ok3, per, est = stage3_estimate()
    if not ok3:
        print('STOP: stage3 abnormal runtime', flush=True)
        return 1
    if not stage4_primary():
        print('STOP: stage4 failed', flush=True)
        return 1
    if not stage5_robust():
        print('STOP: stage5 failed', flush=True)
        return 1
    rows = load_results_csv(os.path.join(OUT_DIR, 'publication_lock_a3_results.csv'))
    write_statistics_and_report(rows, t0, utc_now(), {'per_run': per, 'est_h': est / 3600})
    # hash repeat of summary+statistics
    s1 = file_sha256(os.path.join(OUT_DIR, 'publication_lock_a3_summary.csv'))
    t1 = file_sha256(os.path.join(OUT_DIR, 'publication_lock_a3_statistics.md'))
    write_summary_csv(rows, CONF_SEEDS, os.path.join(OUT_DIR, 'publication_lock_a3_summary.csv'), T_FULL)
    # rewrite statistics deterministically by running analysis again
    write_statistics_and_report(rows, t0, utc_now(), {'per_run': per, 'est_h': est / 3600})
    s2 = file_sha256(os.path.join(OUT_DIR, 'publication_lock_a3_summary.csv'))
    t2 = file_sha256(os.path.join(OUT_DIR, 'publication_lock_a3_statistics.md'))
    print(f'summary sha repeat: {s1==s2} {s1}', flush=True)
    print(f'statistics sha repeat: {t1==t2} {t1}', flush=True)
    if s1 != s2 or t1 != t2:
        print('STOP: analysis outputs not bit-stable', flush=True)
        return 1
    print(f'PIPELINE DONE wall={time.time()-wall0:.1f}s', flush=True)
    return 0


def backfill_timeseries():
    """Re-simulate CSV keys missing from npz shards (same CRN). Does not rewrite CSV."""
    import glob
    print('=== BACKFILL TIMESERIES ===', flush=True)
    rows = load_results_csv(os.path.join(OUT_DIR, 'publication_lock_a3_results.csv'))
    have = set()
    for p in glob.glob(os.path.join(OUT_DIR, 'publication_lock_a3_timeseries_*.npz')):
        if p.endswith(os.path.join('', 'publication_lock_a3_timeseries.npz')):
            continue
        z = np.load(p, allow_pickle=True)
        for m in z['meta']:
            have.add(str(m))
    missing = []
    for r in rows:
        meta = f"{r['seed']}|{r['env_init']}|{r['schedule']}|{r['observer']}"
        if meta not in have:
            missing.append(r)
    print(f'  csv={len(rows)} npz={len(have)} missing={len(missing)}', flush=True)
    pack = {k: [] for k in ['S_pre', 'S_post', 'aligned', 'misID_act', 'NLL_act', 'Brier_act', 'H_q_act', 'meta']}
    import glob as _glob
    existing_bf = _glob.glob(os.path.join(OUT_DIR, 'publication_lock_a3_timeseries_backfill_*.npz'))
    shard_i = 920 + len(existing_bf)
    t0 = time.time()
    for k, r in enumerate(missing, 1):
        env = PRIMARY_ENV if r['env_init'] == PRIMARY_ENV_LABEL else ROBUST_ENV
        obs = r['observer']
        if obs == 'hard':
            observer, lam = 'hard', None
        elif obs == 'smooth':
            observer, lam = 'smooth', None
        else:
            observer, lam = 'filter', float(r['lambda'])
        crn = make_crn_a3(r['seed'], T_FULL)
        tr = run_trial(env, r['schedule'], observer, lam, crn, T_FULL)
        m = metrics_from_trial(tr, T_FULL)
        pack['S_pre'].append(tr['S_pre'])
        pack['S_post'].append(tr['S_post'])
        pack['aligned'].append(m['aligned'])
        pack['misID_act'].append(m['mis_t'])
        pack['NLL_act'].append(m['nll_t'])
        pack['Brier_act'].append(m['brier_t'])
        pack['H_q_act'].append(m['H_t'])
        pack['meta'].append(f"{r['seed']}|{r['env_init']}|{r['schedule']}|{r['observer']}")
        if len(pack['meta']) >= 100:
            _flush_pack(pack, os.path.join(OUT_DIR, f'publication_lock_a3_timeseries_backfill_{shard_i:03d}.npz'))
            shard_i += 1
        if k % 20 == 0:
            print(f'  backfill {k}/{len(missing)} {time.time()-t0:.1f}s', flush=True)
    if pack['meta']:
        _flush_pack(pack, os.path.join(OUT_DIR, f'publication_lock_a3_timeseries_backfill_{shard_i:03d}.npz'))
    print('BACKFILL DONE', flush=True)
    return True


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'pipeline'
    if cmd == 'stage0':
        ok, _ = stage0()
        raise SystemExit(0 if ok else 1)
    if cmd == 'smoke' or cmd == 'stage1':
        raise SystemExit(0 if stage1_smoke() else 1)
    if cmd == 'stage2':
        ok, _ = stage2_regression()
        raise SystemExit(0 if ok else 1)
    if cmd == 'stage3':
        ok, _, _ = stage3_estimate()
        raise SystemExit(0 if ok else 1)
    if cmd == 'stage4':
        raise SystemExit(0 if stage4_primary() else 1)
    if cmd == 'stage5':
        raise SystemExit(0 if stage5_robust() else 1)
    if cmd == 'stage6':
        rows = load_results_csv(os.path.join(OUT_DIR, 'publication_lock_a3_results.csv'))
        write_statistics_and_report(rows, utc_now(), utc_now(), {})
        return
    if cmd == 'backfill':
        raise SystemExit(0 if backfill_timeseries() else 1)
    if cmd == 'pipeline':
        raise SystemExit(pipeline())
    print('usage: publication_lock_a3.py [stage0|smoke|stage2|stage3|stage4|stage5|stage6|backfill|pipeline]')
    raise SystemExit(2)


if __name__ == '__main__':
    main()
