"""
Standalone runner for ActiveInferenceLacan - no pymdp dependency.
Replaces pymdp.utils functions with numpy equivalents.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---- pymdp.utils replacements ----
def onehot(idx, n):
    x = np.zeros(n)
    x[idx] = 1.0
    return x

def norm_dist(dist):
    return np.array(dist) / np.sum(dist)

def sample(dist):
    dist = np.array(dist)
    dist = dist / dist.sum()
    return int(np.random.choice(len(dist), p=dist))

# Patch into the namespace before importing agent
import types
fake_utils = types.ModuleType('pymdp.utils')
fake_utils.onehot = onehot
fake_utils.norm_dist = norm_dist
fake_utils.sample = sample
fake_utils.__file__ = '<fake>'

fake_pymdp = types.ModuleType('pymdp')
fake_pymdp.utils = fake_utils
fake_pymdp.__file__ = '<fake>'
fake_pymdp.__path__ = []

sys.modules['pymdp'] = fake_pymdp
sys.modules['pymdp.utils'] = fake_utils

# Now import the agent module from the current repository layout.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')
OUTPUT_ROOT = os.path.join(REPO_ROOT, 'outputs', 'legacy')
os.makedirs(OUTPUT_ROOT, exist_ok=True)
sys.path.insert(0, SRC_ROOT)
os.chdir(REPO_ROOT)

from agent import Subject, Sync, Three

# ---- Run all three simulations ----
print("=" * 60)
print("Simulation 1: Individual FEP-RSI (Borromean knot dynamics)")
print("=" * 60)
np.random.seed(42)
subj = Subject()
subj.run_active_inference(15)
print(f"  Trajectory shape: {np.array(subj.trajectory).shape}")
print(f"  Final state (R, S, I): {subj.trajectory[-1]}")
print(f"  Trajectory:\n{np.array(subj.trajectory)}")

print("\n" + "=" * 60)
print("Simulation 2: Dyadic Desire (generalized synchronization)")
print("=" * 60)
np.random.seed(42)
sync = Sync()
sync.run_active_inference(15)
traj_a = np.array(sync.trajectory_a)
traj_b = np.array(sync.trajectory_b)
print(f"  Agent A trajectory shape: {traj_a.shape}")
print(f"  Agent B trajectory shape: {traj_b.shape}")
# Measure synchronization: distance between agents' Symbolic states
symbolic_dist = np.abs(traj_a[:, 1] - traj_b[:, 1])
print(f"  Symbolic state distance over time: {symbolic_dist}")
print(f"  Mean Symbolic distance: {symbolic_dist.mean():.3f} (lower = more synchronized)")
print(f"  Agent A final (R,S,I): {traj_a[-1]}")
print(f"  Agent B final (R,S,I): {traj_b[-1]}")

print("\n" + "=" * 60)
print("Simulation 3: Triadic The Other (collective emergence)")
print("=" * 60)
np.random.seed(42)
three = Three()
three.run_active_inference(15)
traj_a = np.array(three.trajectory_a)
traj_b = np.array(three.trajectory_b)
traj_c = np.array(three.trajectory_c)
print(f"  Agent A final (R,S,I): {traj_a[-1]}")
print(f"  Agent B final (R,S,I): {traj_b[-1]}")
print(f"  Agent C final (R,S,I): {traj_c[-1]}")
# Collective dynamics: variance across agents per dimension
collective_var = np.var([traj_a, traj_b, traj_c], axis=0).mean(axis=0)
print(f"  Collective variance (R, S, I): {collective_var}")
print(f"  (lower variance = more coordinated = stronger Other emergence)")

# ---- Save plots ----
print("\n" + "=" * 60)
print("Saving plots...")
print("=" * 60)

# Plot 1: Individual
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
traj = np.array(subj.trajectory)
ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], c='gray', linewidth=2, linestyle='-')
ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c='gray', s=25, marker='o')
ax.set_xlabel('The Real', fontsize=12)
ax.set_ylabel('The Symbolic', fontsize=12)
ax.set_zlabel('The Imaginary', fontsize=12)
ax.set_title('Individual FEP-RSI: Borromean Knot Dynamics', fontsize=14)
ax.set_xlim3d(0, 10); ax.set_ylim3d(0, 10); ax.set_zlim3d(0, 10)
ax.grid(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_ROOT, 'plot_individual.png'), dpi=100, bbox_inches='tight')
print("  Saved: plot_individual.png")

# Plot 2: Dyadic
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2] if False else np.array(sync.trajectory_a)[:, 2], c='red', linewidth=1, linestyle='-', label='Agent A')
ta = np.array(sync.trajectory_a); tb = np.array(sync.trajectory_b)
ax.plot(ta[:, 0], ta[:, 1], ta[:, 2], c='red', linewidth=1, linestyle='-', label='Agent A')
ax.plot(tb[:, 0], tb[:, 1], tb[:, 2], c='black', linewidth=1, linestyle='-', label='Agent B')
ax.scatter(ta[:, 0], ta[:, 1], ta[:, 2], c='red', s=25, marker='o')
ax.scatter(tb[:, 0], tb[:, 1], tb[:, 2], c='black', s=25, marker='o')
ax.set_xlabel('The Real', fontsize=12)
ax.set_ylabel('The Symbolic', fontsize=12)
ax.set_zlabel('The Imaginary', fontsize=12)
ax.set_title('Dyadic Desire: Generalized Synchronization', fontsize=14)
ax.set_xlim3d(0, 8); ax.set_ylim3d(0, 8); ax.set_zlim3d(0, 8)
ax.grid(False)
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_ROOT, 'plot_dyadic.png'), dpi=100, bbox_inches='tight')
print("  Saved: plot_dyadic.png")

# Plot 3: Triadic
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2], c='red', linewidth=1, linestyle='-', label='Agent A')
ax.plot(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2], c='black', linewidth=1, linestyle='-', label='Agent B')
ax.plot(traj_c[:, 0], traj_c[:, 1], traj_c[:, 2], c='green', linewidth=1, linestyle='-', label='Agent C')
ax.scatter(traj_a[:, 0], traj_a[:, 1], traj_a[:, 2], c='red', s=25, marker='o')
ax.scatter(traj_b[:, 0], traj_b[:, 1], traj_b[:, 2], c='black', s=25, marker='o')
ax.scatter(traj_c[:, 0], traj_c[:, 1], traj_c[:, 2], c='green', s=25, marker='o')
ax.set_xlabel('The Real', fontsize=12)
ax.set_ylabel('The Symbolic', fontsize=12)
ax.set_zlabel('The Imaginary', fontsize=12)
ax.set_title('Triadic The Other: Collective Emergence', fontsize=14)
ax.set_xlim3d(0, 10); ax.set_ylim3d(0, 10); ax.set_zlim3d(0, 10)
ax.grid(False)
ax.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_ROOT, 'plot_triadic.png'), dpi=100, bbox_inches='tight')
print("  Saved: plot_triadic.png")

print("\n✅ All simulations completed successfully!")
