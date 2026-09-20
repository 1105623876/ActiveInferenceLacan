"""
Corrected Deep Experiments v2
=============================
Fixes applied per reviewer feedback:
1. State space 0-10 -> 0-8 (9 states), action boundary < 8 -> < 9 (now < n_states-1)
2. A matrix: regenerated + strict column normalization (col 3 was 1.01)
3. Environment isolation: each agent gets own env_r/env_s/env_i (was shared!)
4. KL divergence: mask-based + max(0, ...) to fix negative values from log(a+eps)
5. J1 renamed: "preference mismatch" (not "residual free energy")
6. J2 decomposed: reports H_A (entropy) term and KL term separately
7. Exp 1 title: "ablation" not "psychosis model"
8. J3: 20-seed experiment, reports convergence distribution

Theoretical claims downgraded to operational observations.
"""
import sys, os, types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_paths import output_dir

OUT_DIR = str(output_dir("exploratory_v2"))

# ============================================================
# Clean reimplementation — no dependency on buggy cofig.py
# ============================================================
N_STATES = 9   # was 11, states 9/10 were unreachable
N_OBS = 9
N_ACTIONS = 3
ACTIONS = ["UP", "DOWN", "STAY"]

def make_A_matrix(n_states=N_STATES, sharpness=0.70):
    """Create a properly normalized likelihood matrix.
    Diagonal = sharpness, off-diagonal decays geometrically.
    Strict column normalization."""
    A = np.zeros((n_states, n_states))
    for j in range(n_states):
        weights = []
        for i in range(n_states):
            d = abs(i - j)
            if d == 0:
                weights.append(sharpness)
            else:
                weights.append(sharpness * (1 - sharpness) ** d)
        total = sum(weights)
        for i in range(n_states):
            A[i, j] = weights[i] / total
    return A

A_MAT = make_A_matrix()

def create_B_matrix(n_states=N_STATES):
    """Transition matrix. UP/DOWN now properly bounded [0, n_states-1]."""
    B = np.zeros((n_states, n_states, N_ACTIONS))
    for action_id, action_label in enumerate(ACTIONS):
        for curr_state in range(n_states):
            if action_label == "UP":
                next_x = min(curr_state + 1, n_states - 1)
            elif action_label == "DOWN":
                next_x = max(curr_state - 1, 0)
            else:  # STAY
                next_x = curr_state
            B[next_x, curr_state, action_id] = 1.0
    return B

B_MAT = create_B_matrix()

def softmax_np(x):
    x = x - x.max(axis=0)
    e = np.exp(x)
    return e / e.sum(axis=0)

def log_stable(x):
    return np.log(np.maximum(x, 1e-16))

def onehot(idx, n):
    v = np.zeros(n); v[idx] = 1.0; return v

def norm_dist(d):
    d = np.array(d); return d / d.sum()

def sample(dist):
    dist = np.array(dist); dist = dist / dist.sum()
    return int(np.random.choice(len(dist), p=dist))

# ---- KL divergence (FIXED: mask-based, no negative values) ----
def kl_divergence(a, b):
    """KL(a || b) with numerical stability. Returns >= 0."""
    a = np.array(a); b = np.array(b)
    mask = a > 1e-15
    if not mask.any():
        return 0.0
    result = np.sum(a[mask] * (np.log(a[mask]) - np.log(np.maximum(b[mask], 1e-15))))
    return max(float(result), 0.0)

def infer_states(obs_idx, A, prior):
    log_likelihood = log_stable(A[obs_idx, :])
    log_prior = log_stable(prior)
    return softmax_np(log_likelihood + log_prior)

def get_expected_states(B, qs, action):
    return B[:, :, action].dot(qs)

def get_expected_observations(A, qs_u):
    return A.dot(qs_u)

def entropy(A):
    return -np.sum(A * log_stable(A), axis=0)

def calculate_G_policies(A, B, C, qs_current, policies):
    G = np.zeros(len(policies))
    H_A = entropy(A)
    for pid, policy in enumerate(policies):
        t_horizon = policy.shape[0]
        G_pi = 0.0
        for t in range(t_horizon):
            action = policy[t, 0]
            qs_prev = qs_current if t == 0 else qs_pi_t
            qs_pi_t = get_expected_states(B, qs_prev, action)
            qo_pi_t = get_expected_observations(A, qs_pi_t)
            kld = kl_divergence(qo_pi_t, C)
            G_pi_t = H_A.dot(qs_pi_t) + kld
            G_pi += G_pi_t
        G[pid] += G_pi
    return G

def construct_policies(num_states, num_controls, policy_len=1):
    import itertools
    x = num_controls * policy_len
    policies = list(itertools.product(*[list(range(i)) for i in x]))
    for i in range(len(policies)):
        policies[i] = np.array(policies[i]).reshape(policy_len, len(num_states))
    return policies

def active_inference_with_planning(A, B, C, D, obs_idx, env, policy_len, T):
    prior = D
    obs = obs_idx if obs_idx is not None else env.reset()
    policies = construct_policies([N_STATES], [N_ACTIONS], policy_len=policy_len)
    for t in range(T):
        obs_idx = int(obs)
        qs_current = infer_states(obs_idx, A, prior)
        G = calculate_G_policies(A, B, C, qs_current, policies)
        Q_pi = softmax_np(-G)
        # marginalize
        P_u = np.zeros(N_ACTIONS)
        for pid, policy in enumerate(policies):
            P_u[int(policy[0, 0])] += Q_pi[pid]
        P_u = norm_dist(P_u)
        chosen_action = sample(P_u)
        prior = B[:, :, chosen_action].dot(qs_current)
        obs = env.step(ACTIONS[chosen_action])
    obs_idx = int(obs)
    qs_current = infer_states(obs_idx, A, prior)
    dkl = kl_divergence(qs_current, A[obs_idx, :])
    evidence = log_stable(prior)  # vector
    F = dkl - evidence  # vector over states (matches original cofig.py semantics)
    return obs_idx, qs_current, F

def residual(C, qs):
    """Preference mismatch (renamed from 'residual free energy')."""
    return kl_divergence(C, qs)

def D_update(D, F, R, w):
    """Update prior D based on variational free energy F (vector) and residual R.
    Original semantics: F is a vector over states, find min-F state p,
    add w*R to F[p], then softmax(-F) gives new prior."""
    if w is None: w = 1
    F_vec = np.array(F, dtype=float).copy()
    if F_vec.ndim == 0:
        # scalar fallback (shouldn't happen with correct active_inference)
        F_vec = np.full_like(D, float(F_vec))
    p = int(np.argmin(F_vec))
    F_vec[p] = F_vec[p] + w * R
    return softmax_np(-F_vec)

def change_C(obs_idx):
    return onehot(obs_idx, N_OBS)

def compute_min_G(qs_current, A_mat, B_mat, C_vec, policy_len=2):
    policies = construct_policies([N_STATES], [N_ACTIONS], policy_len=policy_len)
    G = calculate_G_policies(A_mat, B_mat, C_vec, qs_current, policies)
    return G.min()

def compute_G_decomposed(qs_current, A_mat, B_mat, C_vec, policy_len=2):
    """Decompose min G into H_A (entropy) term and KL term separately."""
    policies = construct_policies([N_STATES], [N_ACTIONS], policy_len=policy_len)
    H_A = entropy(A_mat)
    best_G = np.inf
    best_H = 0.0
    best_KL = 0.0
    for policy in policies:
        G_pi = 0.0; H_part = 0.0; KL_part = 0.0
        for t in range(policy.shape[0]):
            action = policy[t, 0]
            qs_prev = qs_current if t == 0 else qs_pi_t
            qs_pi_t = get_expected_states(B_mat, qs_prev, action)
            qo_pi_t = get_expected_observations(A_mat, qs_pi_t)
            kld = kl_divergence(qo_pi_t, C_vec)
            h_val = H_A.dot(qs_pi_t)
            G_pi += h_val + kld
            H_part += h_val; KL_part += kld
        if G_pi < best_G:
            best_G = G_pi; best_H = H_part; best_KL = KL_part
    return best_G, best_H, best_KL


class Agent:
    """Single environment for one order. Each agent gets its OWN instance."""
    def __init__(self, starting_state):
        self.init_state = starting_state
        self.current_state = starting_state

    def step(self, action_label):
        x = self.current_state
        if action_label == "UP":
            self.current_state = min(x + 1, N_STATES - 1)
        elif action_label == "DOWN":
            self.current_state = max(x - 1, 0)
        return self.current_state

    def reset(self):
        self.current_state = self.init_state
        return self.current_state


# ============================================================
# EXP 1 (revised): Symbolic-coupling ablation inspired by RSI disconnection
# ============================================================
def run_individual_ablation(w_r, w_s, w_i, epochs=15, seed=42):
    np.random.seed(seed)
    env_r, env_s, env_i = Agent(5), Agent(1), Agent(6)
    A_R, B_R = A_MAT.copy(), B_MAT.copy(); C_R, D_R = onehot(5, N_STATES), onehot(5, N_STATES)
    A_S, B_S = A_MAT.copy(), B_MAT.copy(); C_S, D_S = onehot(5, N_STATES), onehot(0, N_STATES)
    A_I, B_I = A_MAT.copy(), B_MAT.copy(); C_I, D_I = onehot(6, N_STATES), onehot(6, N_STATES)
    traj, mismatch, min_G_hist, G_decomp_hist = [], [], [], []
    obs_r, obs_s, obs_i = 5, 0, 6
    for j in range(epochs):
        obs_r, qs_r, F_r = active_inference_with_planning(A_R, B_R, C_R, D_R, obs_r, env_r, 2, 2)
        obs_s, qs_s, F_s = active_inference_with_planning(A_S, B_S, C_S, D_S, obs_s, env_s, 4, 1)
        obs_i, qs_i, F_i = active_inference_with_planning(A_I, B_I, C_I, D_I, obs_i, env_i, 2, 1)
        m_R, m_S, m_I = residual(C_R, qs_r), residual(C_S, qs_s), residual(C_I, qs_i)
        mismatch.append([m_R, m_S, m_I])
        G_R, H_R, KL_R = compute_G_decomposed(qs_r, A_R, B_R, C_R)
        G_S, H_S, KL_S = compute_G_decomposed(qs_s, A_S, B_S, C_S)
        G_I, H_I, KL_I = compute_G_decomposed(qs_i, A_I, B_I, C_I)
        min_G_hist.append([G_R, G_S, G_I])
        G_decomp_hist.append([[H_R, KL_R], [H_S, KL_S], [H_I, KL_I]])
        R = m_R + m_S + m_I
        D_R = D_update(D_R, F_r, R, w_r); D_S = D_update(D_S, F_s, R, w_s); D_I = D_update(D_I, F_i, R, w_i)
        traj.append([obs_r, obs_s, obs_i])
    return np.array(traj), np.array(mismatch), np.array(min_G_hist), np.array(G_decomp_hist)


# ============================================================
# EXP 2 (revised): Multi-seed dyadic — isolated envs, 20 seeds
# ============================================================
def run_dyadic_isolated(epochs, seed=42):
    """Each agent has OWN env_r/env_s/env_i. No shared environment."""
    np.random.seed(seed)
    # Agent A envs
    env_r_a, env_s_a, env_i_a = Agent(5), Agent(2), Agent(6)
    # Agent B envs (separate!)
    env_r_b, env_s_b, env_i_b = Agent(3), Agent(4), Agent(5)
    A_R_a, B_R_a = A_MAT.copy(), B_MAT.copy(); C_R_a, D_R_a = onehot(8 % N_STATES, N_STATES), onehot(8 % N_STATES, N_STATES)
    A_S_a, B_S_a = A_MAT.copy(), B_MAT.copy(); C_S_a, D_S_a = onehot(1, N_STATES), onehot(1, N_STATES)
    A_I_a, B_I_a = A_MAT.copy(), B_MAT.copy(); C_I_a, D_I_a = onehot(6, N_STATES), onehot(6, N_STATES)
    A_R_b, B_R_b = A_MAT.copy(), B_MAT.copy(); C_R_b, D_R_b = onehot(3, N_STATES), onehot(3, N_STATES)
    A_S_b, B_S_b = A_MAT.copy(), B_MAT.copy(); C_S_b, D_S_b = onehot(6, N_STATES), onehot(6, N_STATES)
    A_I_b, B_I_b = A_MAT.copy(), B_MAT.copy(); C_I_b, D_I_b = onehot(8 % N_STATES, N_STATES), onehot(8 % N_STATES, N_STATES)
    obs_s_b, obs_s_a = 4, 2
    traj_a, traj_b, J3_kl, J3_obs = [], [], [], []
    obs_r_a, obs_i_a, obs_r_b, obs_i_b = 8 % N_STATES, 6, 3, 8 % N_STATES
    for j in range(epochs):
        # Agent A — uses its OWN envs
        C_S_a = change_C(obs_s_b)
        obs_r_a, qs_r_a, F_r_a = active_inference_with_planning(A_R_a, B_R_a, C_R_a, D_R_a, obs_r_a, env_r_a, 2, 2)
        obs_s_a, qs_s_a, F_s_a = active_inference_with_planning(A_S_a, B_S_a, C_S_a, D_S_a, obs_s_a, env_s_a, 2, 1)
        obs_i_a, qs_i_a, F_i_a = active_inference_with_planning(A_I_a, B_I_a, C_I_a, D_I_a, obs_i_a, env_i_a, 2, 1)
        R_a = residual(C_R_a, qs_r_a) + residual(C_S_a, qs_s_a) + residual(C_I_a, qs_i_a)
        D_R_a = D_update(D_R_a, F_r_a, R_a, 2); D_S_a = D_update(D_S_a, F_s_a, R_a, 0.5); D_I_a = D_update(D_I_a, F_i_a, R_a, 1)
        traj_a.append([obs_r_a, obs_s_a, obs_i_a])
        # Agent B — uses its OWN envs (separate from A!)
        C_S_b = change_C(obs_s_a)
        obs_r_b, qs_r_b, F_r_b = active_inference_with_planning(A_R_b, B_R_b, C_R_b, D_R_b, obs_r_b, env_r_b, 2, 2)
        obs_s_b, qs_s_b, F_s_b = active_inference_with_planning(A_S_b, B_S_b, C_S_b, D_S_b, obs_s_b, env_s_b, 4, 1)
        obs_i_b, qs_i_b, F_i_b = active_inference_with_planning(A_I_b, B_I_b, C_I_b, D_I_b, obs_i_b, env_i_b, 2, 1)
        R_b = residual(C_R_b, qs_r_b) + residual(C_S_b, qs_s_b) + residual(C_I_b, qs_i_b)
        D_R_b = D_update(D_R_b, F_r_b, R_b, 0.5); D_S_b = D_update(D_S_b, F_s_b, R_b, 2); D_I_b = D_update(D_I_b, F_i_b, R_b, 2)
        traj_b.append([obs_r_b, obs_s_b, obs_i_b])
        J3_kl.append(kl_divergence(qs_s_a, qs_s_b))
        J3_obs.append(abs(obs_s_a - obs_s_b))
    return np.array(traj_a), np.array(traj_b), np.array(J3_kl), np.array(J3_obs)


# ============================================================
# EXP 3 (revised): A sharpness with decomposed G
# ============================================================
def run_individual_with_sharpness(sharpness, epochs=15, seed=42):
    np.random.seed(seed)
    A_sharp = make_A_matrix(sharpness=sharpness)
    env_r, env_s, env_i = Agent(5), Agent(1), Agent(6)
    B = B_MAT.copy()
    C_R, D_R = onehot(5, N_STATES), onehot(5, N_STATES)
    C_S, D_S = onehot(5, N_STATES), onehot(0, N_STATES)
    C_I, D_I = onehot(6, N_STATES), onehot(6, N_STATES)
    traj, G_hist, H_hist, KL_hist = [], [], [], []
    obs_r, obs_s, obs_i = 5, 0, 6
    for j in range(epochs):
        obs_r, qs_r, F_r = active_inference_with_planning(A_sharp, B, C_R, D_R, obs_r, env_r, 2, 2)
        obs_s, qs_s, F_s = active_inference_with_planning(A_sharp, B, C_S, D_S, obs_s, env_s, 4, 1)
        obs_i, qs_i, F_i = active_inference_with_planning(A_sharp, B, C_I, D_I, obs_i, env_i, 2, 1)
        G_R, H_R, KL_R = compute_G_decomposed(qs_r, A_sharp, B, C_R)
        G_S, H_S, KL_S = compute_G_decomposed(qs_s, A_sharp, B, C_S)
        G_I, H_I, KL_I = compute_G_decomposed(qs_i, A_sharp, B, C_I)
        G_hist.append([G_R, G_S, G_I]); H_hist.append([H_R, H_S, H_I]); KL_hist.append([KL_R, KL_S, KL_I])
        R = residual(C_R, qs_r) + residual(C_S, qs_s) + residual(C_I, qs_i)
        D_R = D_update(D_R, F_r, R, 2); D_S = D_update(D_S, F_s, R, 0.5); D_I = D_update(D_I, F_i, R, 1)
        traj.append([obs_r, obs_s, obs_i])
    return np.array(traj), np.array(G_hist), np.array(H_hist), np.array(KL_hist)


# ============================================================
# EXP 4 (revised): Triadic with isolated envs
# ============================================================
def run_triadic_isolated(epochs, seed=42):
    np.random.seed(seed)
    # Each agent gets OWN envs
    configs = [
        (8 % N_STATES, 1, 6, 8 % N_STATES, 1, 6),
        (3, 6, 8 % N_STATES, 3, 6, 8 % N_STATES),
        (7, 5, 4, 7, 5, 4),
    ]
    agents = []
    envs = []
    for idx, (c_r, c_s, c_i, d_r, d_s, d_i) in enumerate(configs):
        ag = {
            'A_R': A_MAT.copy(), 'B_R': B_MAT.copy(), 'C_R': onehot(c_r, N_STATES), 'D_R': onehot(d_r, N_STATES),
            'A_S': A_MAT.copy(), 'B_S': B_MAT.copy(), 'C_S': onehot(c_s, N_STATES), 'D_S': onehot(d_s, N_STATES),
            'A_I': A_MAT.copy(), 'B_I': B_MAT.copy(), 'C_I': onehot(c_i, N_STATES), 'D_I': onehot(d_i, N_STATES),
        }
        agents.append(ag)
        # Each agent has its OWN environment instances
        envs.append((Agent(d_r), Agent(d_s), Agent(d_i)))
    weights = [(2, 0.5, 1), (0.5, 2, 2), (0.2, 3, 5)]
    obs = [[8 % N_STATES, 1, 6], [3, 6, 8 % N_STATES], [7, 5, 4]]
    trajs = [[], [], []]
    for j in range(epochs):
        for i, ag in enumerate(agents):
            next_i = (i + 1) % 3
            ag['C_S'] = onehot(obs[next_i][1], N_STATES)
            env_r, env_s, env_i = envs[i]  # own envs
            o_r, qs_r, F_r = active_inference_with_planning(ag['A_R'], ag['B_R'], ag['C_R'], ag['D_R'], obs[i][0], env_r, 2, 2)
            o_s, qs_s, F_s = active_inference_with_planning(ag['A_S'], ag['B_S'], ag['C_S'], ag['D_S'], obs[i][1], env_s, 4, 1)
            o_i, qs_i, F_i = active_inference_with_planning(ag['A_I'], ag['B_I'], ag['C_I'], ag['D_I'], obs[i][2], env_i, 2, 1)
            R = residual(ag['C_R'], qs_r) + residual(ag['C_S'], qs_s) + residual(ag['C_I'], qs_i)
            w_r, w_s, w_i = weights[i]
            ag['D_R'] = D_update(ag['D_R'], F_r, R, w_r)
            ag['D_S'] = D_update(ag['D_S'], F_s, R, w_s)
            ag['D_I'] = D_update(ag['D_I'], F_i, R, w_i)
            obs[i] = [o_r, o_s, o_i]
            trajs[i].append([o_r, o_s, o_i])
    return [np.array(t) for t in trajs]


# ============================================================
# RUN ALL EXPERIMENTS
# ============================================================
print("=" * 70)
print("CORRECTED EXPERIMENTS v2")
print("=" * 70)

# Verify A matrix
print("\n[Verify] A matrix column sums:")
cs = A_MAT.sum(axis=0)
print(f"  all 1.0? {np.allclose(cs, 1.0)}, max dev: {abs(cs-1.0).max():.2e}")
print(f"[Verify] B matrix UP reachability: from 0 -> {np.where(B_MAT[:,0,0]>0)[0].tolist()}, from 7 -> {np.where(B_MAT[:,7,0]>0)[0].tolist()}, from 8 -> {np.where(B_MAT[:,8,0]>0)[0].tolist()}")

# ---- EXP 1 ----
print("\n" + "=" * 70)
print("EXP 1 (revised): Symbolic-coupling ablation (NOT 'psychosis model')")
print("=" * 70)
ablation_configs = [
    ("Baseline", 2.0, 0.5, 1.0),
    ("w_S=0 (Symbolic-coupling off)", 2.0, 0.0, 1.0),
    ("Equal weights", 1.0, 1.0, 1.0),
    ("Full decoupling", 0.0, 0.0, 0.0),
    ("Imaginary-emphasis", 0.5, 0.5, 3.0),
]
exp1_results = {}
for name, wr, ws, wi in ablation_configs:
    traj, mis, minG, decomp = run_individual_ablation(wr, ws, wi)
    exp1_results[name] = (traj, mis, minG, decomp)
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1)) if len(traj) > 1 else 0
    print(f"\n  [{name}] w=({wr},{ws},{wi})")
    print(f"    Preference mismatch (J1) total mean: {mis.sum(axis=1).mean():.3f}, std: {mis.sum(axis=1).std():.3f}")
    print(f"    min-G (J2) mean (R/S/I): ({minG[:,0].mean():.2f}, {minG[:,1].mean():.2f}, {minG[:,2].mean():.2f})")
    print(f"    G decomposition (R: H_A / KL) mean: ({decomp[:,0,0].mean():.2f} / {decomp[:,0,1].mean():.2f})")
    print(f"    Final state: {tuple(traj[-1])}, path length: {path_len:.2f}")

# ---- EXP 2: 20-seed dyadic ----
print("\n" + "=" * 70)
print("EXP 2 (revised): Multi-seed dyadic with ISOLATED envs (20 seeds, 100 steps)")
print("=" * 70)
seeds = list(range(20))
convergence_results = []
all_J3_curves = []
for seed in seeds:
    traj_a, traj_b, J3_kl, J3_obs = run_dyadic_isolated(100, seed=seed)
    all_J3_curves.append(J3_kl)
    first_half = J3_kl[:50].mean()
    second_half = J3_kl[50:].mean()
    converged = second_half < 0.1
    convergence_results.append({
        'seed': seed, 'first_half': first_half, 'second_half': second_half,
        'min': J3_kl.min(), 'converged': converged,
        'near_zero_count': int((J3_kl < 0.001).sum())
    })
print(f"\n  20-seed results (isolated envs, 100 steps):")
print(f"  {'Seed':<6} {'First half':<12} {'Second half':<14} {'Min':<12} {'Near-zero':<12} {'Converged?'}")
for r in convergence_results:
    print(f"  {r['seed']:<6} {r['first_half']:<12.4f} {r['second_half']:<14.4f} {r['min']:<12.6f} {r['near_zero_count']:<12} {r['converged']}")
n_converged = sum(r['converged'] for r in convergence_results)
print(f"\n  Converged (second half < 0.1): {n_converged}/20 seeds")
print(f"  Mean second-half J3 across seeds: {np.mean([r['second_half'] for r in convergence_results]):.4f}")
print(f"  Median second-half J3: {np.median([r['second_half'] for r in convergence_results]):.4f}")

# ---- EXP 3: A sharpness with decomposed G ----
print("\n" + "=" * 70)
print("EXP 3 (revised): A sharpness — decomposed G (H_A term vs KL term)")
print("=" * 70)
sharpness_vals = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0]
exp3_results = {}
for s in sharpness_vals:
    traj, G, H, KL = run_individual_with_sharpness(s)
    exp3_results[s] = (traj, G, H, KL)
    print(f"\n  [sharpness={s}]")
    print(f"    min-G (J2) mean (R/S/I): ({G[:,0].mean():.3f}, {G[:,1].mean():.3f}, {G[:,2].mean():.3f})")
    print(f"    H_A term mean (R/S/I):   ({H[:,0].mean():.3f}, {H[:,1].mean():.3f}, {H[:,2].mean():.3f})")
    print(f"    KL term mean (R/S/I):    ({KL[:,0].mean():.3f}, {KL[:,1].mean():.3f}, {KL[:,2].mean():.3f})")
    print(f"    J2 min: {G.min():.4f}")
print(f"\n  At A=identity: J2 = {exp3_results[1.0][1].min():.6f}, H_A = {exp3_results[1.0][2].min():.6f}, KL = {exp3_results[1.0][3].min():.6f}")

# ---- EXP 4: Triadic isolated ----
print("\n" + "=" * 70)
print("EXP 4 (revised): Triadic with ISOLATED envs (50 steps)")
print("=" * 70)
trajs = run_triadic_isolated(50)
all_trajs = np.stack(trajs)
collective_var = np.var(all_trajs, axis=0).mean(axis=1)
sym_var = np.var(all_trajs[:, :, 1], axis=0)
pairwise_dist = np.array([np.mean([np.linalg.norm(all_trajs[i,t] - all_trajs[j,t]) for i in range(3) for j in range(i+1,3)]) for t in range(50)])
print(f"  Collective var: {collective_var[0]:.3f} -> {collective_var[-1]:.3f}")
print(f"  Symbolic var: {sym_var[0]:.3f} -> {sym_var[-1]:.3f}")
print(f"  Pairwise dist: {pairwise_dist[0]:.3f} -> {pairwise_dist[-1]:.3f}")
print(f"  Final states: A={tuple(trajs[0][-1])}, B={tuple(trajs[1][-1])}, C={tuple(trajs[2][-1])}")
print(f"  Symbolic converged to 0? {sym_var[-1] < 0.01}")


# ============================================================
# PLOTS
# ============================================================
print("\n  Saving plots...")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (name, wr, ws, wi) in zip(axes.flat, ablation_configs):
    traj, mis, minG, decomp = exp1_results[name]
    steps = np.arange(len(traj))
    ax.plot(steps, traj[:, 0], 'r-o', label='Real', markersize=4)
    ax.plot(steps, traj[:, 1], 'b-s', label='Symbolic', markersize=4)
    ax.plot(steps, traj[:, 2], 'g-^', label='Imaginary', markersize=4)
    ax.set_title(f'{name}\nw=({wr},{ws},{wi})', fontsize=9)
    ax.set_xlabel('Timestep'); ax.set_ylabel('State')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Exp 1 (revised): Symbolic-coupling ablation — isolated envs, 9 states', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_v2_exp1_ablation.png'), dpi=120, bbox_inches='tight')

# Exp 2: 20-seed J3 curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for i, J3 in enumerate(all_J3_curves):
    ax.plot(np.arange(100), J3, alpha=0.5, linewidth=1)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Timestep'); ax.set_ylabel('J3 = KL(qs_S_a || qs_S_b)')
ax.set_title(f'Exp 2 (revised): 20-seed J3 curves (isolated envs)\n{n_converged}/20 seeds converge (2nd half < 0.1)')
ax.grid(alpha=0.3)
ax = axes[1]
second_halves = [r['second_half'] for r in convergence_results]
ax.hist(second_halves, bins=10, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(0.1, color='red', linestyle='--', label='convergence threshold (0.1)')
ax.set_xlabel('Second-half mean J3'); ax.set_ylabel('Seed count')
ax.set_title('Exp 2: Convergence distribution across 20 seeds')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_v2_exp2_multiseed.png'), dpi=120, bbox_inches='tight')

# Exp 3: decomposed G
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sharp_arr = np.array(sharpness_vals)
ax = axes[0]
ax.plot(sharp_arr, [exp3_results[s][1].mean() for s in sharp_arr], 'r-o', label='J2 = min G (total)')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('A sharpness'); ax.set_ylabel('J2 mean')
ax.set_title('J2 (total min-G)'); ax.legend(); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(sharp_arr, [exp3_results[s][2].mean() for s in sharp_arr], 'b-s', label='H_A (entropy) term')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('A sharpness'); ax.set_ylabel('H_A term mean')
ax.set_title('H_A term (trivially decreases with sharpness)'); ax.legend(); ax.grid(alpha=0.3)
ax = axes[2]
ax.plot(sharp_arr, [exp3_results[s][3].mean() for s in sharp_arr], 'g-^', label='KL term')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('A sharpness'); ax.set_ylabel('KL term mean')
ax.set_title('KL term (the non-trivial part)'); ax.legend(); ax.grid(alpha=0.3)
plt.suptitle('Exp 3 (revised): Decomposed G — separating H_A (trivial) from KL (non-trivial)', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_v2_exp3_decomposed.png'), dpi=120, bbox_inches='tight')

# Exp 4: triadic isolated
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for i, (c, lbl) in enumerate(zip(['red', 'black', 'green'], ['A', 'B', 'C'])):
    ax.plot(np.arange(50), trajs[i][:, 1], '-', color=c, linewidth=2, label=f'Agent {lbl} Symbolic')
ax.set_xlabel('Timestep'); ax.set_ylabel('Symbolic state')
ax.set_title('Exp 4 (revised): Triadic Symbolic dynamics (isolated envs)')
ax.legend(); ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(np.arange(50), collective_var, '-o', color='darkred', label='Collective var (RSI mean)')
ax.plot(np.arange(50), sym_var, '-s', color='darkblue', label='Symbolic var')
ax.plot(np.arange(50), pairwise_dist, '-^', color='darkgreen', label='Pairwise dist')
ax.set_xlabel('Timestep'); ax.set_ylabel('Dispersion')
ax.set_title('Exp 4: The Other emergence metrics (isolated envs)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'plot_v2_exp4_triadic.png'), dpi=120, bbox_inches='tight')

print("\n  Saved: plot_v2_exp1_ablation.png, plot_v2_exp2_multiseed.png, plot_v2_exp3_decomposed.png, plot_v2_exp4_triadic.png")
print("\n=== DONE ===")
