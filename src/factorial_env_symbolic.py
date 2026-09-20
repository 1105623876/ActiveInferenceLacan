"""
Factorial 2×2 Experiment: Environment Mode × Inter-agent Symbolic Coupling
=========================================================================

研究问题：
  在三主体循环 Symbolic coupling 模型中，Symbolic 层收敛究竟来自：
  1. 共享物理环境；
  2. agent 间 change_C / Symbolic preference coupling；
  3. 二者的组合。

实验设计（2×2 factorial）：
  因素一 - 环境模式 (env_mode)：
    - shared    : 三个 agent 共用同一组 env_r/env_s/env_i 实例（初始 [5,2,6]）
    - isolated  : 每个 agent 拥有独立的 env_r/env_s/env_i 副本（初始 [5,2,6]）
  因素二 - inter-agent Symbolic preference coupling (c_update)：
    - C_update_on  : 保持 change_C 机制（agent i 更新时把 next agent 的当前
                     Symbolic observation 设为自己的 C_S）
    - C_update_off : 不调用 change_C，保留每个 agent 的初始固定 C_S

  注意：C_update_off 不是"完全移除 Symbolic Order"。每个 agent 内部仍保留
  R/S/I 三界结构、w_S 权重和 Symbolic active-inference unit；只是移除了
  agent 之间的 Symbolic preference 耦合（inter-agent coupling ablation）。

公平性：
  - 使用 deep_experiments_v2.py 当前 triadic 配置（agent C_R/C_S/C_I/D_R/D_S/D_I、
    weights、policy_len、T、A->B->C sequential update order）
  - shared 和 isolated 条件的环境初始状态统一为 [5,2,6]
  - 不沿用 v2 run_triadic_isolated 中按 agent d_r/d_s/d_i 创建不同环境初始状态的做法
  - 每个 seed、每个条件开始前重新初始化所有环境、agent 参数和 observation
  - 20 seeds (0..19)，50 steps
  - w_S（agent 内部 RSI coupling 权重）在四个条件中保持不变

不修改 deep_experiments_v2.py；复制最小模型函数以避免 import 触发顶层执行。
仅使用 numpy + matplotlib（当前已有依赖）。
"""
import numpy as np
import csv
import os
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from project_paths import output_dir

# ============================================================
# 复制 deep_experiments_v2.py 的核心模型函数
# （v2 顶层无 __main__ 守卫，import 会触发全部实验，故复制最小集）
# ============================================================
N_STATES = 9
N_OBS = 9
N_ACTIONS = 3
ACTIONS = ["UP", "DOWN", "STAY"]


def make_A_matrix(n_states=N_STATES, sharpness=0.70):
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
    B = np.zeros((n_states, n_states, N_ACTIONS))
    for action_id, action_label in enumerate(ACTIONS):
        for curr_state in range(n_states):
            if action_label == "UP":
                next_x = min(curr_state + 1, n_states - 1)
            elif action_label == "DOWN":
                next_x = max(curr_state - 1, 0)
            else:
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


def kl_divergence(a, b):
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
    evidence = log_stable(prior)
    F = dkl - evidence
    return obs_idx, qs_current, F


def residual(C, qs):
    return kl_divergence(C, qs)


def D_update(D, F, R, w):
    if w is None:
        w = 1
    F_vec = np.array(F, dtype=float).copy()
    if F_vec.ndim == 0:
        F_vec = np.full_like(D, float(F_vec))
    p = int(np.argmin(F_vec))
    F_vec[p] = F_vec[p] + w * R
    return softmax_np(-F_vec)


def change_C(obs_idx):
    return onehot(obs_idx, N_OBS)


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
# Factorial 实验配置（来自 deep_experiments_v2.py run_triadic_isolated）
# ============================================================

# (c_r, c_s, c_i, d_r, d_s, d_i) per agent — 与 v2 完全一致
AGENT_CONFIGS = [
    (8 % N_STATES, 1, 6, 8 % N_STATES, 1, 6),   # Agent A
    (3, 6, 8 % N_STATES, 3, 6, 8 % N_STATES),   # Agent B
    (7, 5, 4, 7, 5, 4),                           # Agent C
]
WEIGHTS = [(2, 0.5, 1), (0.5, 2, 2), (0.2, 3, 5)]
INIT_OBS = [[8 % N_STATES, 1, 6], [3, 6, 8 % N_STATES], [7, 5, 4]]

# 环境初始状态统一为 [5, 2, 6]（公平性要求：不按 agent d_r/d_s/d_i 区分）
ENV_INIT = (5, 2, 6)

# 收敛阈值：v2 Exp4 标准 sym_var[-1] < 0.01
CONV_THRESHOLD = 0.01
T_STEPS = 50
SEEDS = list(range(20))

OUT_DIR = str(output_dir("factorial"))


# ============================================================
# 单条件单 seed 运行
# ============================================================
def run_condition(env_mode, c_update, seed, T=T_STEPS):
    """
    运行单个条件的单次 seed，返回三个 agent 的轨迹。

    env_mode : 'shared' 或 'isolated'
    c_update : True (C_update_on) 或 False (C_update_off)
    """
    np.random.seed(seed)

    # ---- 初始化 agents（每 seed 重置）----
    agents = []
    for (c_r, c_s, c_i, d_r, d_s, d_i) in AGENT_CONFIGS:
        ag = {
            'A_R': A_MAT.copy(), 'B_R': B_MAT.copy(),
            'C_R': onehot(c_r, N_STATES), 'D_R': onehot(d_r, N_STATES),
            'A_S': A_MAT.copy(), 'B_S': B_MAT.copy(),
            'C_S': onehot(c_s, N_STATES), 'D_S': onehot(d_s, N_STATES),
            'A_I': A_MAT.copy(), 'B_I': B_MAT.copy(),
            'C_I': onehot(c_i, N_STATES), 'D_I': onehot(d_i, N_STATES),
            'init_C_S': onehot(c_s, N_STATES),  # 保留初始 C_S 用于 C_update_off
        }
        agents.append(ag)

    # ---- 初始化环境 ----
    if env_mode == 'shared':
        # 三个 agent 共享同一组 env 实例（原始 agent.py Three 类的设计）
        shared_envs = (Agent(ENV_INIT[0]), Agent(ENV_INIT[1]), Agent(ENV_INIT[2]))
        envs = [shared_envs, shared_envs, shared_envs]
    else:  # isolated
        envs = [
            (Agent(ENV_INIT[0]), Agent(ENV_INIT[1]), Agent(ENV_INIT[2])),
            (Agent(ENV_INIT[0]), Agent(ENV_INIT[1]), Agent(ENV_INIT[2])),
            (Agent(ENV_INIT[0]), Agent(ENV_INIT[1]), Agent(ENV_INIT[2])),
        ]

    obs = [list(o) for o in INIT_OBS]
    trajs = [[], [], []]

    for j in range(T):
        for i, ag in enumerate(agents):
            next_i = (i + 1) % 3
            if c_update:
                # C_update_on: 把下一个 agent 的当前 Symbolic observation 设为自己的 C_S
                ag['C_S'] = change_C(obs[next_i][1])
            else:
                # C_update_off: 保留初始固定 C_S（inter-agent coupling ablation）
                ag['C_S'] = ag['init_C_S'].copy()

            env_r, env_s, env_i = envs[i]
            o_r, qs_r, F_r = active_inference_with_planning(
                ag['A_R'], ag['B_R'], ag['C_R'], ag['D_R'], obs[i][0], env_r, 2, 2)
            o_s, qs_s, F_s = active_inference_with_planning(
                ag['A_S'], ag['B_S'], ag['C_S'], ag['D_S'], obs[i][1], env_s, 4, 1)
            o_i, qs_i, F_i = active_inference_with_planning(
                ag['A_I'], ag['B_I'], ag['C_I'], ag['D_I'], obs[i][2], env_i, 2, 1)

            R = residual(ag['C_R'], qs_r) + residual(ag['C_S'], qs_s) + residual(ag['C_I'], qs_i)
            w_r, w_s, w_i = WEIGHTS[i]
            ag['D_R'] = D_update(ag['D_R'], F_r, R, w_r)
            ag['D_S'] = D_update(ag['D_S'], F_s, R, w_s)
            ag['D_I'] = D_update(ag['D_I'], F_i, R, w_i)

            obs[i] = [o_r, o_s, o_i]
            trajs[i].append([o_r, o_s, o_i])

    return [np.array(t) for t in trajs]


# ============================================================
# 指标计算
# ============================================================
def compute_metrics(trajs, T=T_STEPS):
    """从轨迹计算所有指标。"""
    all_trajs = np.stack(trajs)  # (3, T, 3)

    # Symbolic observation variance (across 3 agents, per timestep)
    sym_var = np.var(all_trajs[:, :, 1], axis=0)  # (T,)

    # Collective RSI variance (mean over R/S/I of var across agents)
    collective_var = np.var(all_trajs, axis=0).mean(axis=1)  # (T,)

    # Pairwise distance (mean over 3 pairs, per timestep)
    pairwise_dist = np.array([
        np.mean([np.linalg.norm(all_trajs[i, t] - all_trajs[j, t])
                 for i in range(3) for j in range(i + 1, 3)])
        for t in range(T)
    ])

    final_states = {
        'A': tuple(int(x) for x in trajs[0][-1]),
        'B': tuple(int(x) for x in trajs[1][-1]),
        'C': tuple(int(x) for x in trajs[2][-1]),
    }

    return {
        'sym_var': sym_var,
        'collective_var': collective_var,
        'pairwise_dist': pairwise_dist,
        'sym_var_init': float(sym_var[0]),
        'sym_var_final': float(sym_var[-1]),
        'sym_var_second_half_mean': float(sym_var[T // 2:].mean()),
        'collective_var_init': float(collective_var[0]),
        'collective_var_final': float(collective_var[-1]),
        'collective_var_second_half_mean': float(collective_var[T // 2:].mean()),
        'pairwise_dist_init': float(pairwise_dist[0]),
        'pairwise_dist_final': float(pairwise_dist[-1]),
        'pairwise_dist_second_half_mean': float(pairwise_dist[T // 2:].mean()),
        'final_states': final_states,
        'converged': bool(sym_var[-1] < CONV_THRESHOLD),
    }


# ============================================================
# CSV 输出
# ============================================================
def write_results_csv(results, out_path):
    """每 seed 每条件的详细记录。"""
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'seed', 'env_mode', 'c_update',
            'sym_var_init', 'sym_var_final', 'sym_var_second_half_mean',
            'collective_var_init', 'collective_var_final', 'collective_var_second_half_mean',
            'pairwise_dist_init', 'pairwise_dist_final', 'pairwise_dist_second_half_mean',
            'final_R_A', 'final_S_A', 'final_I_A',
            'final_R_B', 'final_S_B', 'final_I_B',
            'final_R_C', 'final_S_C', 'final_I_C',
            'converged',
        ])
        for seed in SEEDS:
            for env_mode in ['shared', 'isolated']:
                for c_update in [True, False]:
                    r = results[(seed, env_mode, c_update)]
                    fs = r['final_states']
                    w.writerow([
                        seed, env_mode, 'C_update_on' if c_update else 'C_update_off',
                        f"{r['sym_var_init']:.6f}", f"{r['sym_var_final']:.6f}",
                        f"{r['sym_var_second_half_mean']:.6f}",
                        f"{r['collective_var_init']:.6f}", f"{r['collective_var_final']:.6f}",
                        f"{r['collective_var_second_half_mean']:.6f}",
                        f"{r['pairwise_dist_init']:.6f}", f"{r['pairwise_dist_final']:.6f}",
                        f"{r['pairwise_dist_second_half_mean']:.6f}",
                        fs['A'][0], fs['A'][1], fs['A'][2],
                        fs['B'][0], fs['B'][1], fs['B'][2],
                        fs['C'][0], fs['C'][1], fs['C'][2],
                        int(r['converged']),
                    ])


def write_timeseries_csv(results, out_path, T=T_STEPS):
    """每 seed 每条件每 step 的时间序列。"""
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'seed', 'env_mode', 'c_update', 'step',
            'sym_var', 'collective_var', 'pairwise_dist',
            'obs_R_A', 'obs_S_A', 'obs_I_A',
            'obs_R_B', 'obs_S_B', 'obs_I_B',
            'obs_R_C', 'obs_S_C', 'obs_I_C',
        ])
        for seed in SEEDS:
            for env_mode in ['shared', 'isolated']:
                for c_update in [True, False]:
                    r = results[(seed, env_mode, c_update)]
                    trajs = r['_trajs']
                    all_t = np.stack(trajs)  # (3, T, 3)
                    for t in range(T):
                        w.writerow([
                            seed, env_mode, 'C_update_on' if c_update else 'C_update_off', t,
                            f"{r['sym_var'][t]:.6f}",
                            f"{r['collective_var'][t]:.6f}",
                            f"{r['pairwise_dist'][t]:.6f}",
                            int(all_t[0, t, 0]), int(all_t[0, t, 1]), int(all_t[0, t, 2]),
                            int(all_t[1, t, 0]), int(all_t[1, t, 1]), int(all_t[1, t, 2]),
                            int(all_t[2, t, 0]), int(all_t[2, t, 1]), int(all_t[2, t, 2]),
                        ])


def write_summary_csv(results, out_path):
    """四条件汇总（跨 20 seeds 聚合）。"""
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'env_mode', 'c_update', 'n_seeds', 'n_converged', 'convergence_rate',
            'sym_var_final_mean', 'sym_var_final_std',
            'sym_var_second_half_mean', 'sym_var_second_half_std',
            'collective_var_final_mean', 'collective_var_final_std',
            'pairwise_dist_final_mean', 'pairwise_dist_final_std',
        ])
        for env_mode in ['shared', 'isolated']:
            for c_update in [True, False]:
                subset = [results[(s, env_mode, c_update)] for s in SEEDS]
                n_conv = sum(1 for r in subset if r['converged'])
                sv_final = np.array([r['sym_var_final'] for r in subset])
                sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
                cv_final = np.array([r['collective_var_final'] for r in subset])
                pd_final = np.array([r['pairwise_dist_final'] for r in subset])
                w.writerow([
                    env_mode, 'C_update_on' if c_update else 'C_update_off',
                    len(SEEDS), n_conv, f"{n_conv / len(SEEDS):.2f}",
                    f"{sv_final.mean():.6f}", f"{sv_final.std():.6f}",
                    f"{sv_sh.mean():.6f}", f"{sv_sh.std():.6f}",
                    f"{cv_final.mean():.6f}", f"{cv_final.std():.6f}",
                    f"{pd_final.mean():.6f}", f"{pd_final.std():.6f}",
                ])


# ============================================================
# 绘图
# ============================================================
def plot_factorial(results, out_path, T=T_STEPS):
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))

    conditions = [
        ('shared', True,   'shared + C_update_on',   axes[0, 0]),
        ('shared', False,  'shared + C_update_off',  axes[0, 1]),
        ('isolated', True,  'isolated + C_update_on',  axes[1, 0]),
        ('isolated', False, 'isolated + C_update_off', axes[1, 1]),
    ]

    color_map = {'shared': '#e74c3c', 'isolated': '#3498db'}

    for env_mode, c_update, title, ax in conditions:
        color = color_map[env_mode]
        linestyle = '-' if c_update else '--'
        all_sym_vars = []
        for seed in SEEDS:
            r = results[(seed, env_mode, c_update)]
            all_sym_vars.append(r['sym_var'])
            ax.plot(r['sym_var'], color='gray', alpha=0.2, linewidth=0.8)
        mean_sv = np.mean(all_sym_vars, axis=0)
        ax.plot(mean_sv, color=color, linewidth=2.5, linestyle=linestyle, label='mean (20 seeds)')
        ax.axhline(CONV_THRESHOLD, color='red', linestyle=':', alpha=0.5,
                   label=f'threshold={CONV_THRESHOLD}')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Step')
        ax.set_ylabel('Symbolic observation variance')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, max(2.0, np.max(all_sym_vars) * 1.1))

    # 收敛率柱状图
    ax_conv = axes[2, 0]
    conv_rates = []
    labels = []
    bar_colors = []
    for env_mode, c_update, title, _ in conditions:
        n_conv = sum(1 for s in SEEDS if results[(s, env_mode, c_update)]['converged'])
        conv_rates.append(n_conv / len(SEEDS))
        labels.append(title)
        bar_colors.append(color_map[env_mode])
    bars = ax_conv.bar(labels, conv_rates, color=bar_colors, alpha=0.7, edgecolor='black')
    ax_conv.set_ylabel('Convergence rate (20 seeds)')
    ax_conv.set_title('Convergence rate by condition\n(threshold: sym_var_final < 0.01)')
    ax_conv.set_ylim(0, 1.15)
    for bar, v in zip(bars, conv_rates):
        ax_conv.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                     f'{v:.0%}', ha='center', fontsize=10)
    ax_conv.tick_params(axis='x', rotation=20)
    ax_conv.grid(alpha=0.3, axis='y')

    # sym_var_final 箱线图
    ax_box = axes[2, 1]
    box_data = []
    for env_mode, c_update, title, _ in conditions:
        finals = [results[(s, env_mode, c_update)]['sym_var_final'] for s in SEEDS]
        box_data.append(finals)
    bp = ax_box.boxplot(box_data, labels=labels, showmeans=True, patch_artist=True)
    for patch, c in zip(bp['boxes'], bar_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.4)
    ax_box.set_ylabel('Symbolic var (final)')
    ax_box.set_title('Distribution of final Symbolic variance (20 seeds)')
    ax_box.tick_params(axis='x', rotation=20)
    ax_box.grid(alpha=0.3)

    plt.suptitle(
        'Factorial 2×2: Environment Mode × Inter-agent Symbolic Coupling\n'
        '(20 seeds, 50 steps, sym_var across 3 agents)',
        fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()


# ============================================================
# 报告生成
# ============================================================
def write_report(results, out_path, T=T_STEPS):
    # 按条件聚合
    cond_data = {}
    for env_mode in ['shared', 'isolated']:
        for c_update in [True, False]:
            subset = [results[(s, env_mode, c_update)] for s in SEEDS]
            sv_final = np.array([r['sym_var_final'] for r in subset])
            sv_sh = np.array([r['sym_var_second_half_mean'] for r in subset])
            cv_final = np.array([r['collective_var_final'] for r in subset])
            pd_final = np.array([r['pairwise_dist_final'] for r in subset])
            n_conv = sum(1 for r in subset if r['converged'])
            cond_data[(env_mode, c_update)] = {
                'sv_final_mean': sv_final.mean(),
                'sv_final_std': sv_final.std(),
                'sv_sh_mean': sv_sh.mean(),
                'sv_sh_std': sv_sh.std(),
                'cv_final_mean': cv_final.mean(),
                'cv_final_std': cv_final.std(),
                'pd_final_mean': pd_final.mean(),
                'pd_final_std': pd_final.std(),
                'n_conv': n_conv,
                'conv_rate': n_conv / len(SEEDS),
                'sv_final_arr': sv_final,
            }

    # 主效应
    # C_update 主效应：mean(C_on) - mean(C_off)，跨 env_mode
    c_on_mean = np.mean([cond_data[(e, True)]['sv_final_arr'] for e in ['shared', 'isolated']])
    c_off_mean = np.mean([cond_data[(e, False)]['sv_final_arr'] for e in ['shared', 'isolated']])
    c_effect = c_on_mean - c_off_mean

    # env_mode 主效应：mean(shared) - mean(isolated)，跨 c_update
    shared_mean = np.mean([cond_data[('shared', c)]['sv_final_arr'] for c in [True, False]])
    iso_mean = np.mean([cond_data[('isolated', c)]['sv_final_arr'] for c in [True, False]])
    env_effect = shared_mean - iso_mean

    # 交互：(shared_C_on - shared_C_off) - (iso_C_on - iso_C_off)
    shared_c_effect = cond_data[('shared', True)]['sv_final_mean'] - cond_data[('shared', False)]['sv_final_mean']
    iso_c_effect = cond_data[('isolated', True)]['sv_final_mean'] - cond_data[('isolated', False)]['sv_final_mean']
    interaction = shared_c_effect - iso_c_effect

    # 逐 seed 收敛表
    seed_conv_lines = []
    header = f"| seed | shared/C_on | shared/C_off | isolated/C_on | isolated/C_off |"
    sep = f"|------|-------------|--------------|---------------|----------------|"
    seed_conv_lines.append(header)
    seed_conv_lines.append(sep)
    for s in SEEDS:
        row = []
        for env_mode in ['shared', 'isolated']:
            for c_update in [True, False]:
                r = results[(s, env_mode, c_update)]
                row.append(f"{'✓' if r['converged'] else '✗'} ({r['sym_var_final']:.3f})")
        seed_conv_lines.append(f"| {s} | {row[0]} | {row[1]} | {row[2]} | {row[3]} |")

    report = f"""# Factorial 2×2 Experiment Report

## 研究问题

在三主体循环 Symbolic coupling 模型中，Symbolic 层收敛究竟来自：
1. 共享物理环境；
2. agent 间 change_C / Symbolic preference coupling；
3. 二者的组合。

## 实验设计

2×2 factorial，两个因素各两个水平：

| 因素 | 水平 1 | 水平 2 |
|------|--------|--------|
| 环境模式 (env_mode) | shared（三 agent 共享同一组 env） | isolated（每 agent 独立 env 副本） |
| inter-agent Symbolic coupling (c_update) | C_update_on（保持 change_C） | C_update_off（保留初始固定 C_S） |

### 关键设计说明

- **shared 环境是原始模型（agent.py 中 Three 类）有意的建模设定**：三个 agent 共用同一组 env_r/env_s/env_i 实例，A 的动作会改变 B/C 看到的环境状态。v2 修正版将其改为 isolated 以排除共享环境的混淆。本实验同时测试两种模式以分解环境因素的贡献。
- **isolated 环境是替代模型**（v2 修正版采用的设计）：每个 agent 拥有独立的环境实例副本。
- **C_update_off 是 inter-agent Symbolic preference coupling ablation**，不是"完全移除 Symbolic Order"。每个 agent 内部仍保留 R/S/I 三界结构、w_S 权重和 Symbolic active-inference unit；只是不再把 next agent 的 Symbolic observation 设为自己的 C_S，而是保持初始固定 C_S。Symbolic Order 作为 agent 内部结构依然存在。
- w_S（agent 内部 RSI coupling 权重）在四个条件中保持 v2 triadic 配置不变：weights = [(2,0.5,1), (0.5,2,2), (0.2,3,5)]。
- 三个 agent 的 C_R/C_S/C_I/D_R/D_S/D_I 配置、policy_len（R=2, S=4, I=2）、T（R=2, S=1, I=1）、A→B→C sequential update order 均与 v2 run_triadic_isolated 一致。
- shared 和 isolated 条件的环境初始状态统一为 [5, 2, 6]（不沿用 v2 中按 agent d_r/d_s/d_i 创建不同环境初始状态的做法，以避免环境模式与初始状态混淆）。
- 20 seeds (0..19)，50 steps，每 seed 每条件开始前重新初始化所有环境、agent 参数和 observation。
- 收敛阈值：sym_var_final < 0.01（与 v2 Exp4 一致）。

## 四条件汇总表

| 条件 | 收敛率 | sym_var_final (mean±std) | sym_var_2nd_half (mean±std) | collective_var_final (mean±std) | pairwise_dist_final (mean±std) |
|------|--------|--------------------------|----------------------------|--------------------------------|-------------------------------|
| shared + C_update_on   | {cond_data[('shared',True)]['n_conv']}/20 ({cond_data[('shared',True)]['conv_rate']:.0%}) | {cond_data[('shared',True)]['sv_final_mean']:.4f} ± {cond_data[('shared',True)]['sv_final_std']:.4f} | {cond_data[('shared',True)]['sv_sh_mean']:.4f} ± {cond_data[('shared',True)]['sv_sh_std']:.4f} | {cond_data[('shared',True)]['cv_final_mean']:.4f} ± {cond_data[('shared',True)]['cv_final_std']:.4f} | {cond_data[('shared',True)]['pd_final_mean']:.4f} ± {cond_data[('shared',True)]['pd_final_std']:.4f} |
| shared + C_update_off  | {cond_data[('shared',False)]['n_conv']}/20 ({cond_data[('shared',False)]['conv_rate']:.0%}) | {cond_data[('shared',False)]['sv_final_mean']:.4f} ± {cond_data[('shared',False)]['sv_final_std']:.4f} | {cond_data[('shared',False)]['sv_sh_mean']:.4f} ± {cond_data[('shared',False)]['sv_sh_std']:.4f} | {cond_data[('shared',False)]['cv_final_mean']:.4f} ± {cond_data[('shared',False)]['cv_final_std']:.4f} | {cond_data[('shared',False)]['pd_final_mean']:.4f} ± {cond_data[('shared',False)]['pd_final_std']:.4f} |
| isolated + C_update_on  | {cond_data[('isolated',True)]['n_conv']}/20 ({cond_data[('isolated',True)]['conv_rate']:.0%}) | {cond_data[('isolated',True)]['sv_final_mean']:.4f} ± {cond_data[('isolated',True)]['sv_final_std']:.4f} | {cond_data[('isolated',True)]['sv_sh_mean']:.4f} ± {cond_data[('isolated',True)]['sv_sh_std']:.4f} | {cond_data[('isolated',True)]['cv_final_mean']:.4f} ± {cond_data[('isolated',True)]['cv_final_std']:.4f} | {cond_data[('isolated',True)]['pd_final_mean']:.4f} ± {cond_data[('isolated',True)]['pd_final_std']:.4f} |
| isolated + C_update_off | {cond_data[('isolated',False)]['n_conv']}/20 ({cond_data[('isolated',False)]['conv_rate']:.0%}) | {cond_data[('isolated',False)]['sv_final_mean']:.4f} ± {cond_data[('isolated',False)]['sv_final_std']:.4f} | {cond_data[('isolated',False)]['sv_sh_mean']:.4f} ± {cond_data[('isolated',False)]['sv_sh_std']:.4f} | {cond_data[('isolated',False)]['cv_final_mean']:.4f} ± {cond_data[('isolated',False)]['cv_final_std']:.4f} | {cond_data[('isolated',False)]['pd_final_mean']:.4f} ± {cond_data[('isolated',False)]['pd_final_std']:.4f} |

## 主效应分析

### C_update 主效应（inter-agent Symbolic coupling）

mean(sym_var_final | C_update_on) - mean(sym_var_final | C_update_off)，跨 env_mode：

- C_update_on 均值（跨 env_mode, 40 seeds）: {c_on_mean:.4f}
- C_update_off 均值（跨 env_mode, 40 seeds）: {c_off_mean:.4f}
- **主效应 = {c_effect:.4f}**

{'C_update_on 显著降低了 sym_var_final（负值），说明 inter-agent Symbolic coupling 是 Symbolic 层收敛的主要驱动因素。' if c_effect < -0.01 else 'C_update 的主效应较小，说明 inter-agent Symbolic coupling 对 sym_var_final 的影响有限。' if abs(c_effect) <= 0.01 else 'C_update_on 反而提高了 sym_var_final，与预期不符，需进一步检查。'}

### 环境模式主效应

mean(sym_var_final | shared) - mean(sym_var_final | isolated)，跨 c_update：

- shared 均值（跨 c_update, 40 seeds）: {shared_mean:.4f}
- isolated 均值（跨 c_update, 40 seeds）: {iso_mean:.4f}
- **主效应 = {env_effect:.4f}**

{'shared 环境显著降低了 sym_var_final，说明共享物理环境对 Symbolic 收敛有贡献。' if env_effect < -0.01 else '环境模式的主效应较小，说明共享物理环境对 sym_var_final 的影响有限。' if abs(env_effect) <= 0.01 else 'shared 环境反而提高了 sym_var_final，说明共享环境可能引入了额外的扰动。'}

### 环境模式 × C_update 交互差异

(shared_C_on - shared_C_off) - (isolated_C_on - isolated_C_off)：

- shared 条件下 C_update 效应: {shared_c_effect:.4f}
- isolated 条件下 C_update 效应: {iso_c_effect:.4f}
- **交互差异 = {interaction:.4f}**

{'交互差异较大，说明 C_update 的效果依赖于环境模式。' if abs(interaction) > 0.05 else '交互差异较小，说明 C_update 的效果在 shared 和 isolated 条件下相对一致。'}

## 20 seeds 逐 seed 收敛表

{chr(10).join(seed_conv_lines)}

## 与 v2 Exp 4 的差异

v2 Exp 4（run_triadic_isolated, 50 steps）使用的是 **isolated + C_update_on** 条件，但环境初始状态按 agent 各自的 D 创建：
- Agent A env 初始: [8, 1, 6]（来自 d_r=8, d_s=1, d_i=6）
- Agent B env 初始: [3, 6, 8]
- Agent C env 初始: [7, 5, 4]

本实验的 isolated + C_update_on 条件与之区别在于：环境初始状态统一为 [5, 2, 6]（公平性要求：避免环境模式与初始状态混淆）。agent 配置（C/D/weights/obs/policy_len）完全一致。

v2 Exp 4 报告 sym_var 从 1.556 收敛到 0.000。本实验的 isolated + C_update_on 条件 20 seeds 收敛率为 {cond_data[('isolated',True)]['conv_rate']:.0%}，sym_var_final 均值为 {cond_data[('isolated',True)]['sv_final_mean']:.4f}。差异（如有）主要来自环境初始状态的不同（[5,2,6] vs [8,1,6]/[3,6,8]/[7,5,4]）。

## 重要声明

1. **shared 环境是原模型有意的建模设定**（对应原始 agent.py 中 Three 类的共享 env 设计），isolated 环境是替代模型（v2 修正版采用）。本实验不评判哪种设计"正确"，而是分解两种因素各自对 Symbolic 收敛的贡献。
2. **C_update_off 不是"完全移除 Symbolic Order"**。每个 agent 内部仍保留完整的 R/S/I 三界结构、w_S 权重和 Symbolic active-inference unit；只是移除了 agent 之间的 Symbolic preference 耦合（inter-agent coupling ablation）。Symbolic Order 作为 agent 内部结构依然存在。
3. 本实验结果**不构成拉康理论的证明**。这是一个在有限状态空间、固定策略长度、one-hot preference 的 toy model 上的计算观察。Symbolic 收敛的具体数值依赖于 A 矩阵 sharpness、policy_len、weights 等参数配置。
4. 本实验报告基于 20 seeds 的聚合统计，而非单个 seed 或单条代表性轨迹。

## 文件清单

| 文件 | 说明 |
|------|------|
| src/factorial_env_symbolic.py | 本实验脚本（独立，不修改 src/deep_experiments_v2.py） |
| outputs/factorial/factorial_results.csv | 每 seed 每条件的详细记录 |
| outputs/factorial/factorial_summary.csv | 四条件汇总（跨 20 seeds 聚合） |
| outputs/factorial/factorial_timeseries.csv | 每 seed 每条件每 step 的时间序列 |
| outputs/factorial/plot_factorial_2x2.png | 四条件 sym_var 曲线 + 收敛率 + 分布图 |
| outputs/factorial/factorial_report.md | 本报告 |
"""
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(report)


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 70)
    print("FACTORIAL 2×2: Environment Mode × Inter-agent Symbolic Coupling")
    print("=" * 70)

    # 验证 A 矩阵
    cs = A_MAT.sum(axis=0)
    assert np.allclose(cs, 1.0), f"A matrix not normalized: max dev {abs(cs - 1.0).max()}"
    print(f"[Verify] A matrix column sums all 1.0? {np.allclose(cs, 1.0)}")

    results = {}
    conditions = [
        ('shared', True,   'shared + C_update_on'),
        ('shared', False,  'shared + C_update_off'),
        ('isolated', True,  'isolated + C_update_on'),
        ('isolated', False, 'isolated + C_update_off'),
    ]

    for seed in SEEDS:
        for env_mode, c_update, cond_name in conditions:
            trajs = run_condition(env_mode, c_update, seed)
            metrics = compute_metrics(trajs)
            metrics['_trajs'] = trajs
            results[(seed, env_mode, c_update)] = metrics
        if (seed + 1) % 5 == 0:
            print(f"  Completed {seed + 1}/20 seeds")

    # 写 CSV
    write_results_csv(results, os.path.join(OUT_DIR, 'factorial_results.csv'))
    write_summary_csv(results, os.path.join(OUT_DIR, 'factorial_summary.csv'))
    write_timeseries_csv(results, os.path.join(OUT_DIR, 'factorial_timeseries.csv'))
    print("[Saved] factorial_results.csv, factorial_summary.csv, factorial_timeseries.csv")

    # 画图
    plot_factorial(results, os.path.join(OUT_DIR, 'plot_factorial_2x2.png'))
    print("[Saved] plot_factorial_2x2.png")

    # 写报告
    write_report(results, os.path.join(OUT_DIR, 'factorial_report.md'))
    print("[Saved] factorial_report.md")

    # 打印汇总
    print("\n" + "=" * 70)
    print("SUMMARY (sym_var_final mean ± std, convergence rate)")
    print("=" * 70)
    print(f"{'Condition':<30} {'sym_var_final':<20} {'conv_rate':<12}")
    print("-" * 62)
    for env_mode, c_update, cond_name in conditions:
        subset = [results[(s, env_mode, c_update)] for s in SEEDS]
        sv = np.array([r['sym_var_final'] for r in subset])
        n_conv = sum(1 for r in subset if r['converged'])
        cu_label = 'C_update_on' if c_update else 'C_update_off'
        print(f"{env_mode + ' + ' + cu_label:<30} {sv.mean():.4f} ± {sv.std():.4f}      {n_conv}/20")

    # 主效应
    c_on_mean = np.mean([results[(s, e, True)]['sym_var_final'] for s in SEEDS for e in ['shared', 'isolated']])
    c_off_mean = np.mean([results[(s, e, False)]['sym_var_final'] for s in SEEDS for e in ['shared', 'isolated']])
    shared_mean = np.mean([results[(s, 'shared', c)]['sym_var_final'] for s in SEEDS for c in [True, False]])
    iso_mean = np.mean([results[(s, 'isolated', c)]['sym_var_final'] for s in SEEDS for c in [True, False]])
    print(f"\nC_update main effect:      {c_on_mean:.4f} - {c_off_mean:.4f} = {c_on_mean - c_off_mean:.4f}")
    print(f"env_mode main effect:      {shared_mean:.4f} - {iso_mean:.4f} = {shared_mean - iso_mean:.4f}")
    shared_c_eff = np.mean([results[(s, 'shared', True)]['sym_var_final'] for s in SEEDS]) - np.mean([results[(s, 'shared', False)]['sym_var_final'] for s in SEEDS])
    iso_c_eff = np.mean([results[(s, 'isolated', True)]['sym_var_final'] for s in SEEDS]) - np.mean([results[(s, 'isolated', False)]['sym_var_final'] for s in SEEDS])
    print(f"Interaction (shared_C - iso_C): {shared_c_eff:.4f} - {iso_c_eff:.4f} = {shared_c_eff - iso_c_eff:.4f}")

    print("\nDone.")


if __name__ == '__main__':
    main()
