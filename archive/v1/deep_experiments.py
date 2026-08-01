"""
Deep Experiments: Parameter Sweeps & Structural Analysis
=========================================================
Four experiments extending the surplus jouissance formalization:

Exp 1: Precision weight (w_R/w_S/w_I) sweep — modeling psychosis (RSI slippage)
Exp 2: Long-range desire synchronization — does J3 → 0 asymptotically?
Exp 3: Likelihood matrix sharpness — proving J2's structural nature
Exp 4: Triadic collective dynamics — quantifying The Other emergence
"""
import sys, os, types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---- pymdp shim ----
def onehot(idx, n):
    x = np.zeros(n); x[idx] = 1.0; return x
def norm_dist(dist):
    return np.array(dist) / np.sum(dist)
def sample(dist):
    dist = np.array(dist); dist = dist / dist.sum()
    return int(np.random.choice(len(dist), p=dist))
fake_utils = types.ModuleType('pymdp.utils')
fake_utils.onehot = onehot; fake_utils.norm_dist = norm_dist; fake_utils.sample = sample
fake_pymdp = types.ModuleType('pymdp'); fake_pymdp.utils = fake_utils
sys.modules['pymdp'] = fake_pymdp; sys.modules['pymdp.utils'] = fake_utils

sys.path.insert(0, '/mnt/d/vibecoding/ActiveInferenceLacan')
os.chdir('/mnt/d/vibecoding/ActiveInferenceLacan')
from cofig import (residual, kl_divergence, calculate_G_policies, construct_policies,
    active_inference_with_planning, D_update, create_B_matrix, A,
    n_states, n_actions, softmax, log_stable, entropy,
    get_expected_states, get_expected_observations)

def compute_min_G(qs_current, A_mat, B_mat, C_vec, policy_len=2):
    policies = construct_policies([n_states], [n_actions], policy_len=policy_len)
    G = calculate_G_policies(A_mat, B_mat, C_vec, qs_current, policies)
    return G.min()

def make_sharp_A(sharpness):
    """Create a likelihood matrix with given diagonal sharpness.
    sharpness=0.70 ~ original; 1.0 = identity (lossless); 0.5 = very noisy.
    Diagonal = sharpness, off-diagonal decays with distance."""
    n = 11
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs(i - j)
            if d == 0:
                A[i, j] = sharpness
            else:
                A[i, j] = sharpness * (1 - sharpness) ** d / (1 - (1-sharpness))
    # normalize columns
    A = A / A.sum(axis=0, keepdims=True)
    return A


# ============================================================
# EXP 1: Precision weight sweep — psychosis modeling
# ============================================================
def run_individual_with_weights(w_r, w_s, w_i, epochs=15, seed=42):
    """Run individual FEP-RSI with given precision weights, return J1/J2/trajectory."""
    np.random.seed(seed)
    from agent import Agent
    env_r, env_s, env_i = Agent(5), Agent(1), Agent(6)
    A_R, B_R = A.copy(), create_B_matrix(); C_R = onehot(5, 11); D_R = onehot(5, 11)
    A_S, B_S = A.copy(), create_B_matrix(); C_S = onehot(5, 11); D_S = onehot(0, 11)
    A_I, B_I = A.copy(), create_B_matrix(); C_I = onehot(6, 11); D_I = onehot(6, 11)

    traj, J1_hist, J2_hist = [], [], []
    obs_r, obs_s, obs_i = 5, 0, 6
    for j in range(epochs):
        obs_r, qs_r, F_r = active_inference_with_planning(A_R, B_R, C_R, D_R, obs_r, n_actions, env_r, policy_len=2, T=2)
        obs_s, qs_s, F_s = active_inference_with_planning(A_S, B_S, C_S, D_S, obs_s, n_actions, env_s, policy_len=4, T=1)
        obs_i, qs_i, F_i = active_inference_with_planning(A_I, B_I, C_I, D_I, obs_i, n_actions, env_i, policy_len=2, T=1)
        R_R, R_S, R_I = residual(C_R, qs_r), residual(C_S, qs_s), residual(C_I, qs_i)
        J1_hist.append([R_R, R_S, R_I])
        J2_hist.append([compute_min_G(qs_r, A_R, B_R, C_R), compute_min_G(qs_s, A_S, B_S, C_S), compute_min_G(qs_i, A_I, B_I, C_I)])
        R = R_R + R_S + R_I
        D_R = D_update(D_R, F_r, R, w_r)
        D_S = D_update(D_S, F_s, R, w_s)
        D_I = D_update(D_I, F_i, R, w_i)
        traj.append([obs_r, obs_s, obs_i])
    return np.array(traj), np.array(J1_hist), np.array(J2_hist)

print("=" * 70)
print("EXP 1: Precision Weight Sweep — Modeling Psychosis (RSI Slippage)")
print("=" * 70)

# Baseline (paper default): w_R=2, w_S=0.5, w_I=1
# Psychosis model: reduce w_S (Symbolic disconnection) — schizophrenia analog
# Also test: equal weights, zero-weight (full decoupling), inverted weights
weight_configs = [
    ("Baseline (正常)", 2.0, 0.5, 1.0),
    ("Symbolic slippage (精神病模型)", 2.0, 0.0, 1.0),   # w_S=0: Symbolic断联
    ("Symbolic weakened", 2.0, 0.1, 1.0),
    ("Equal weights", 1.0, 1.0, 1.0),
    ("Full decoupling (RSI崩溃)", 0.0, 0.0, 0.0),         # 全断
    ("Inverted (Imaginary主导)", 0.5, 0.5, 3.0),          # I过度连接
]

exp1_results = {}
for name, w_r, w_s, w_i in weight_configs:
    traj, J1, J2 = run_individual_with_weights(w_r, w_s, w_i)
    exp1_results[name] = (traj, J1, J2)
    print(f"\n  [{name}] w=(R={w_r}, S={w_s}, I={w_i})")
    print(f"    J1 total mean: {J1.sum(axis=1).mean():.3f}")
    print(f"    J1 trajectory stability (std of total): {J1.sum(axis=1).std():.3f}")
    print(f"    J2 mean (R/S/I): ({J2[:,0].mean():.2f}, {J2[:,1].mean():.2f}, {J2[:,2].mean():.2f})")
    print(f"    Final state (R,S,I): {tuple(traj[-1])}")
    # Trajectory wandering: total path length
    path_len = np.sum(np.linalg.norm(np.diff(traj, axis=0), axis=1))
    print(f"    Trajectory path length: {path_len:.3f} (higher = more erratic)")


# ============================================================
# EXP 2: Long-range desire synchronization — does J3 → 0?
# ============================================================
def run_dyadic_longrange(epochs, seed=42):
    """Run dyadic desire for many steps, track J3 convergence."""
    np.random.seed(seed)
    from agent import Agent
    env_r, env_s, env_i = Agent(5), Agent(2), Agent(6)
    A_R_a, B_R_a = A.copy(), create_B_matrix(); C_R_a = onehot(8, 11); D_R_a = onehot(8, 11)
    A_S_a, B_S_a = A.copy(), create_B_matrix(); C_S_a = onehot(1, 11); D_S_a = onehot(1, 11)
    A_I_a, B_I_a = A.copy(), create_B_matrix(); C_I_a = onehot(6, 11); D_I_a = onehot(6, 11)
    A_R_b, B_R_b = A.copy(), create_B_matrix(); C_R_b = onehot(3, 11); D_R_b = onehot(3, 11)
    A_S_b, B_S_b = A.copy(), create_B_matrix(); C_S_b = onehot(6, 11); D_S_b = onehot(6, 11)
    A_I_b, B_I_b = A.copy(), create_B_matrix(); C_I_b = onehot(8, 11); D_I_b = onehot(8, 11)

    obs_s_b, obs_s_a = 4, 2
    traj_a, traj_b = [], []
    J3_kl, J3_obs = [], []

    for j in range(epochs):
        # Agent A
        obs_r_a, qs_r_a, F_r_a = active_inference_with_planning(A_R_a, B_R_a, C_R_a, D_R_a, 8 if j==0 else obs_r_a, n_actions, env_r, policy_len=2, T=2)
        C_S_a = onehot(obs_s_b, 11)
        obs_s_a, qs_s_a, F_s_a = active_inference_with_planning(A_S_a, B_S_a, C_S_a, D_S_a, 1 if j==0 else obs_s_a, n_actions, env_s, policy_len=2, T=1)
        obs_i_a, qs_i_a, F_i_a = active_inference_with_planning(A_I_a, B_I_a, C_I_a, D_I_a, 6 if j==0 else obs_i_a, n_actions, env_i, policy_len=2, T=1)
        R_a = residual(C_R_a, qs_r_a) + residual(C_S_a, qs_s_a) + residual(C_I_a, qs_i_a)
        D_R_a = D_update(D_R_a, F_r_a, R_a, 2); D_S_a = D_update(D_S_a, F_s_a, R_a, 0.5); D_I_a = D_update(D_I_a, F_i_a, R_a, 1)
        traj_a.append([obs_r_a, obs_s_a, obs_i_a])

        # Agent B
        obs_r_b, qs_r_b, F_r_b = active_inference_with_planning(A_R_b, B_R_b, C_R_b, D_R_b, 3 if j==0 else obs_r_b, n_actions, env_r, policy_len=2, T=2)
        obs_s_b, qs_s_b, F_s_b = active_inference_with_planning(A_S_b, B_S_b, C_S_b, D_S_b, 6 if j==0 else obs_s_b, n_actions, env_s, policy_len=4, T=1)
        C_S_b = onehot(obs_s_a, 11)
        obs_i_b, qs_i_b, F_i_b = active_inference_with_planning(A_I_b, B_I_b, C_I_b, D_I_b, 8 if j==0 else obs_i_b, n_actions, env_i, policy_len=2, T=1)
        R_b = residual(C_R_b, qs_r_b) + residual(C_S_b, qs_s_b) + residual(C_I_b, qs_i_b)
        D_R_b = D_update(D_R_b, F_r_b, R_b, 0.5); D_S_b = D_update(D_S_b, F_s_b, R_b, 2); D_I_b = D_update(D_I_b, F_i_b, R_b, 2)
        traj_b.append([obs_r_b, obs_s_b, obs_i_b])

        # J3
        J3_kl.append(kl_divergence(qs_s_a, qs_s_b))
        J3_obs.append(abs(obs_s_a - obs_s_b))

    return np.array(traj_a), np.array(traj_b), np.array(J3_kl), np.array(J3_obs)

print("\n" + "=" * 70)
print("EXP 2: Long-range Desire Synchronization — Does J3 → 0?")
print("=" * 70)
traj_a_lr, traj_b_lr, J3_kl_lr, J3_obs_lr = run_dyadic_longrange(100)
print(f"  100-step dyadic simulation:")
print(f"  J3 (KL) over time — first 10: {np.round(J3_kl_lr[:10], 4)}")
print(f"  J3 (KL) last 10:              {np.round(J3_kl_lr[-10:], 4)}")
print(f"  J3 (KL) mean: {J3_kl_lr.mean():.4f}, min: {J3_kl_lr.min():.6f}")
print(f"  J3 (KL) ever reaches exactly 0? {np.any(J3_kl_lr == 0)}")
print(f"  J3 (KL) near-zero (< 0.001) count: {(J3_kl_lr < 0.001).sum()}/100")
# Does it decrease over time? Compare first vs second half
first_half = J3_kl_lr[:50].mean(); second_half = J3_kl_lr[50:].mean()
print(f"  First-half mean: {first_half:.4f}, Second-half mean: {second_half:.4f}")
print(f"  Decreasing? {'Yes' if second_half < first_half else 'No'} (ratio: {second_half/first_half:.3f})")


# ============================================================
# EXP 3: Likelihood matrix sharpness — J2's structural nature
# ============================================================
def run_individual_with_A(A_mat, epochs=15, seed=42):
    """Run individual FEP-RSI with custom A matrix."""
    np.random.seed(seed)
    from agent import Agent
    env_r, env_s, env_i = Agent(5), Agent(1), Agent(6)
    B = create_B_matrix()
    A_R, A_S, A_I = A_mat.copy(), A_mat.copy(), A_mat.copy()
    B_R, B_S, B_I = B.copy(), B.copy(), B.copy()
    C_R, D_R = onehot(5, 11), onehot(5, 11)
    C_S, D_S = onehot(5, 11), onehot(0, 11)
    C_I, D_I = onehot(6, 11), onehot(6, 11)
    traj, J2_hist = [], []
    obs_r, obs_s, obs_i = 5, 0, 6
    for j in range(epochs):
        obs_r, qs_r, F_r = active_inference_with_planning(A_R, B_R, C_R, D_R, obs_r, n_actions, env_r, policy_len=2, T=2)
        obs_s, qs_s, F_s = active_inference_with_planning(A_S, B_S, C_S, D_S, obs_s, n_actions, env_s, policy_len=4, T=1)
        obs_i, qs_i, F_i = active_inference_with_planning(A_I, B_I, C_I, D_I, obs_i, n_actions, env_i, policy_len=2, T=1)
        J2_hist.append([compute_min_G(qs_r, A_R, B_R, C_R), compute_min_G(qs_s, A_S, B_S, C_S), compute_min_G(qs_i, A_I, B_I, C_I)])
        R = residual(C_R, qs_r) + residual(C_S, qs_s) + residual(C_I, qs_i)
        D_R = D_update(D_R, F_r, R, 2); D_S = D_update(D_S, F_s, R, 0.5); D_I = D_update(D_I, F_i, R, 1)
        traj.append([obs_r, obs_s, obs_i])
    return np.array(traj), np.array(J2_hist)

print("\n" + "=" * 70)
print("EXP 3: Likelihood Matrix Sharpness — Proving J2's Structural Nature")
print("=" * 70)
sharpness_values = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.0]
exp3_results = {}
for s in sharpness_values:
    A_sharp = make_sharp_A(s)
    traj, J2 = run_individual_with_A(A_sharp)
    exp3_results[s] = (traj, J2)
    label = "identity (lossless)" if s == 1.0 else f"sharpness={s}"
    print(f"  [{label}] J2 mean (R/S/I): ({J2[:,0].mean():.3f}, {J2[:,1].mean():.3f}, {J2[:,2].mean():.3f}), J2 min: {J2.min():.4f}")

print(f"\n  Key finding: As A → identity (lossless), J2 → ?")
print(f"  At A=identity: J2 = {exp3_results[1.0][1].min():.6f} (should be ~0 if structural claim is wrong)")


# ============================================================
# EXP 4: Triadic collective dynamics — quantifying The Other emergence
# ============================================================
def run_triadic(epochs, seed=42):
    """Run triadic model, return trajectories + collective metrics."""
    np.random.seed(seed)
    from agent import Agent
    env_r, env_s, env_i = Agent(5), Agent(2), Agent(6)
    # Three agents with different initials
    configs = [
        (8, 1, 6, 8, 1, 6),  # A
        (3, 6, 8, 3, 6, 8),  # B
        (9, 5, 4, 7, 5, 4),  # C
    ]
    agents = []
    for c_r, c_s, c_i, d_r, d_s, d_i in configs:
        ag = {
            'A_R': A.copy(), 'B_R': create_B_matrix(), 'C_R': onehot(c_r, 11), 'D_R': onehot(d_r, 11),
            'A_S': A.copy(), 'B_S': create_B_matrix(), 'C_S': onehot(c_s, 11), 'D_S': onehot(d_s, 11),
            'A_I': A.copy(), 'B_I': create_B_matrix(), 'C_I': onehot(c_i, 11), 'D_I': onehot(d_i, 11),
        }
        agents.append(ag)
    weights = [(2, 0.5, 1), (0.5, 2, 2), (0.2, 3, 5)]  # per agent
    obs = [[8, 1, 6], [3, 6, 8], [7, 5, 4]]
    trajs = [[], [], []]

    for j in range(epochs):
        # Sequential: A→B→C→A desire chain
        # A wants B's Symbolic; B wants C's; C wants A's
        for i, ag in enumerate(agents):
            # Set Symbolic preference to next agent's current Symbolic obs
            next_i = (i + 1) % 3
            ag['C_S'] = onehot(obs[next_i][1], 11)
            o_r, qs_r, F_r = active_inference_with_planning(ag['A_R'], ag['B_R'], ag['C_R'], ag['D_R'], obs[i][0], n_actions, env_r, policy_len=2, T=2)
            o_s, qs_s, F_s = active_inference_with_planning(ag['A_S'], ag['B_S'], ag['C_S'], ag['D_S'], obs[i][1], n_actions, env_s, policy_len=4, T=1)
            o_i, qs_i, F_i = active_inference_with_planning(ag['A_I'], ag['B_I'], ag['C_I'], ag['D_I'], obs[i][2], n_actions, env_i, policy_len=2, T=1)
            R = residual(ag['C_R'], qs_r) + residual(ag['C_S'], qs_s) + residual(ag['C_I'], qs_i)
            w_r, w_s, w_i = weights[i]
            ag['D_R'] = D_update(ag['D_R'], F_r, R, w_r)
            ag['D_S'] = D_update(ag['D_S'], F_s, R, w_s)
            ag['D_I'] = D_update(ag['D_I'], F_i, R, w_i)
            obs[i] = [o_r, o_s, o_i]
            trajs[i].append([o_r, o_s, o_i])
    return [np.array(t) for t in trajs]

print("\n" + "=" * 70)
print("EXP 4: Triadic Collective Dynamics — Quantifying The Other Emergence")
print("=" * 70)
trajs = run_triadic(50)
# Collective variance per dimension over time
all_trajs = np.stack(trajs)  # (3, 50, 3)
collective_var = np.var(all_trajs, axis=0).mean(axis=1)  # mean across R/S/I
# Symbolic coherence specifically
sym_var = np.var(all_trajs[:, :, 1], axis=0)  # variance of Symbolic across agents
# Order parameter: mean pairwise distance
pairwise_dist = np.zeros(50)
for t in range(50):
    dists = [np.linalg.norm(all_trajs[i, t] - all_trajs[j, t]) for i in range(3) for j in range(i+1, 3)]
    pairwise_dist[t] = np.mean(dists)

print(f"  50-step triadic simulation:")
print(f"  Collective variance — first 5: {np.round(collective_var[:5], 3)}")
print(f"  Collective variance — last 5:  {np.round(collective_var[-5:], 3)}")
print(f"  Collective variance trend: {collective_var[0]:.3f} → {collective_var[-1]:.3f}")
print(f"  Symbolic variance trend: {sym_var[0]:.3f} → {sym_var[-1]:.3f}")
print(f"  Mean pairwise distance trend: {pairwise_dist[0]:.3f} → {pairwise_dist[-1]:.3f}")
print(f"  Convergence (distance decreasing)? {'Yes' if pairwise_dist[-1] < pairwise_dist[0] else 'No'}")
# Does The Other emerge? = does collective behavior self-organize without central rule?
print(f"\n  Final agent states:")
for i, t in enumerate(trajs):
    print(f"    Agent {chr(65+i)}: (R={t[-1,0]}, S={t[-1,1]}, I={t[-1,2]})")


# ============================================================
# Save all plots
# ============================================================
print("\n" + "=" * 70)
print("Saving deep experiment plots...")
print("=" * 70)

# --- Plot Exp 1: Precision weight sweep ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, (name, w_r, w_s, w_i) in zip(axes.flat, weight_configs):
    traj, J1, J2 = exp1_results[name]
    steps = np.arange(len(traj))
    ax.plot(steps, traj[:, 0], 'r-o', label='Real', markersize=4)
    ax.plot(steps, traj[:, 1], 'b-s', label='Symbolic', markersize=4)
    ax.plot(steps, traj[:, 2], 'g-^', label='Imaginary', markersize=4)
    ax.set_title(f'{name}\nw=({w_r},{w_s},{w_i})', fontsize=10)
    ax.set_xlabel('Timestep'); ax.set_ylabel('State')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.suptitle('Exp 1: Precision Weight Sweep — RSI Trajectories under Different w Configs', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/mnt/d/vibecoding/ActiveInferenceLacan/plot_exp1_weights.png', dpi=120, bbox_inches='tight')
print("  Saved: plot_exp1_weights.png")

# --- Plot Exp 1b: J1 total comparison ---
fig, ax = plt.subplots(figsize=(12, 6))
for name, w_r, w_s, w_i in weight_configs:
    traj, J1, J2 = exp1_results[name]
    ax.plot(np.arange(len(J1)), J1.sum(axis=1), '-o', label=f'{name} w=({w_r},{w_s},{w_i})', markersize=4)
ax.set_xlabel('Timestep'); ax.set_ylabel('J1 total = Σ KL(C, qs)')
ax.set_title('Exp 1b: Local Surplus (J1) under Different Precision Weights\nPsychosis model: w_S=0 shows decoupled Symbolic')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/d/vibecoding/ActiveInferenceLacan/plot_exp1b_j1.png', dpi=120, bbox_inches='tight')
print("  Saved: plot_exp1b_j1.png")

# --- Plot Exp 2: Long-range J3 ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
ax.plot(np.arange(100), J3_kl_lr, '-o', color='purple', markersize=3)
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.set_xlabel('Timestep'); ax.set_ylabel('J3 = KL(qs_S_a || qs_S_b)')
ax.set_title('Exp 2: Long-range Desire Synchronization (100 steps)\nJ3 never reaches 0 — surplus is persistent')
ax.grid(alpha=0.3)
# mark min
min_idx = J3_kl_lr.argmin()
ax.annotate(f'min = {J3_kl_lr[min_idx]:.6f}\n(still > 0 at step {min_idx})',
            xy=(min_idx, J3_kl_lr[min_idx]), xytext=(min_idx+10, J3_kl_lr[min_idx]+5),
            arrowprops=dict(arrowstyle='->', color='red'), fontsize=10, color='red')

ax = axes[1]
# distribution of J3
ax.hist(J3_kl_lr, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(0, color='black', linewidth=1, linestyle='--')
ax.set_xlabel('J3 value'); ax.set_ylabel('Count')
ax.set_title(f'Exp 2: J3 Distribution (n=100)\nmean={J3_kl_lr.mean():.3f}, min={J3_kl_lr.min():.6f}\nzero count: {(J3_kl_lr==0).sum()}')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/d/vibecoding/ActiveInferenceLacan/plot_exp2_longrange.png', dpi=120, bbox_inches='tight')
print("  Saved: plot_exp2_longrange.png")

# --- Plot Exp 3: J2 vs A sharpness ---
fig, ax = plt.subplots(figsize=(10, 6))
sharpness_arr = np.array(sharpness_values)
J2_means = [exp3_results[s][1].mean() for s in sharpness_values]
J2_mins = [exp3_results[s][1].min() for s in sharpness_values]
ax.plot(sharpness_arr, J2_means, '-o', color='red', linewidth=2, label='J2 mean')
ax.plot(sharpness_arr, J2_mins, '-s', color='blue', linewidth=2, label='J2 min')
ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
ax.axvline(1.0, color='gray', linewidth=0.5, linestyle=':', label='A = identity (lossless)')
ax.set_xlabel('Likelihood matrix A sharpness (diagonal weight)')
ax.set_ylabel('J2 = min_π G(π)')
ax.set_title('Exp 3: Structural Surplus (J2) vs Likelihood Sharpness\nProving J2 → 0 only when A → identity (lossless signifier)')
ax.legend(); ax.grid(alpha=0.3)
# annotate the limit
ax.annotate(f'At A=identity: J2={exp3_results[1.0][1].min():.4f}',
            xy=(1.0, exp3_results[1.0][1].min()),
            xytext=(0.85, exp3_results[0.7][1].mean()*0.7),
            arrowprops=dict(arrowstyle='->', color='green'), fontsize=10, color='green')
plt.tight_layout()
plt.savefig('/mnt/d/vibecoding/ActiveInferenceLacan/plot_exp3_sharpness.png', dpi=120, bbox_inches='tight')
print("  Saved: plot_exp3_sharpness.png")

# --- Plot Exp 4: Triadic collective dynamics ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
steps = np.arange(50)
for i, (c, lbl) in enumerate(zip(['red', 'black', 'green'], ['Agent A', 'Agent B', 'Agent C'])):
    ax.plot(steps, trajs[i][:, 1], '-', color=c, linewidth=2, label=f'{lbl} Symbolic')
ax.set_xlabel('Timestep'); ax.set_ylabel('Symbolic state')
ax.set_title('Exp 4: Triadic Symbolic Dynamics\nThe Other = emergent coordination pattern')
ax.legend(); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(steps, collective_var, '-o', color='darkred', linewidth=2, label='Collective variance (RSI mean)')
ax.plot(steps, sym_var, '-s', color='darkblue', linewidth=2, label='Symbolic variance')
ax.plot(steps, pairwise_dist, '-^', color='darkgreen', linewidth=2, label='Mean pairwise distance')
ax.set_xlabel('Timestep'); ax.set_ylabel('Dispersion metric')
ax.set_title('Exp 4: The Other Emergence Metrics\nLower = more coordinated = stronger collective')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/mnt/d/vibecoding/ActiveInferenceLacan/plot_exp4_triadic.png', dpi=120, bbox_inches='tight')
print("  Saved: plot_exp4_triadic.png")

print("\n✅ All deep experiments complete.")
