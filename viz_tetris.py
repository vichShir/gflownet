# import pickle
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).resolve().parent.parent))
# from gflownet.utils.common import load_gflownet_from_rundir

# RUNDIR = "/home/victor/projects/sardinha/projeto/gflownet/logs/local/2026-10-10_18-16-47_156999"
# K_TOP = 10

# gflownet, run_config = load_gflownet_from_rundir(
#     rundir=RUNDIR, device="cuda", no_wandb=True,
#     print_config=False, load_last_checkpoint=True,
# )
# env = gflownet.env

# with open(f"{RUNDIR}/eval/samples/gfn_samples.pkl", "rb") as f:
#     data = pickle.load(f)
# samples = data["x"]
# print("n samples:", len(samples), "| first board shape:", torch.as_tensor(samples[0]).shape)

# # Rewards from the run's own proxy; fall back to occupied cells if that call differs
# try:
#     with torch.no_grad():
#         rewards = gflownet.proxy.rewards(env.states2proxy(samples)).tolist()
#     print("rewards computed with the proxy")
# except Exception as e:
#     print("proxy.rewards failed (", e, ") -> using occupied-cell count")
#     rewards = [float((torch.as_tensor(s) > 0).sum()) for s in samples]

# print("max reward:", max(rewards), "| mean reward:", float(np.mean(rewards)))

# rewards = torch.as_tensor(rewards, dtype=torch.float32)
# fig = env.plot_samples_topk(samples, rewards, k_top=K_TOP, n_rows=2, dpi=150)
# if fig is None:
#     fig = plt.gcf()
# fig.savefig("./tetris_topk.png", dpi=150, bbox_inches="tight")
# print("saved ./tetris_topk.png")









import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

from gflownet.utils.common import load_gflownet_from_rundir

from tqdm import tqdm

RUNDIR = "/home/victor/projects/sardinha/projeto/gflownet/logs/local/2026-10-12_21-38-43_674148"
N_TRIES = 5
DEVICE = "cuda"

gflownet, run_config = load_gflownet_from_rundir(
    rundir=RUNDIR, device=DEVICE, no_wandb=True,
    print_config=False, load_last_checkpoint=True,
)
env = gflownet.env

gflownet.forward_policy.model.eval()
gflownet.backward_policy.model.eval()
# import ipdb; ipdb.set_trace()

# Follow the learned policy only (no exploration noise), if the attribute exists
for name in ["random_action_prob", "random_action_prob_forward"]:
    if hasattr(gflownet, name):
        print(f"setting gflownet.{name} = 0.0 (was {getattr(gflownet, name)})")
        setattr(gflownet, name, 0.0)


def to_board(s):
    return torch.as_tensor(s).detach().cpu().numpy()


def occupied(board):
    return float((board > 0).sum())


# ---------- 1. Sample full trajectories (train=True keeps all steps) ----------
best = None
with torch.no_grad():
    for _ in tqdm(range(N_TRIES)):
        batch, _ = gflownet.sample_batch(n_forward=1, train=True)
        final = to_board(batch.get_terminating_states()[0])
        r = occupied(final)
        if best is None or r > best[0]:
            best = (r, batch)
best_reward, batch = best
print("best occupied cells among tries:", best_reward)

# import ipdb; ipdb.set_trace()

# ---------- 2. Extract states and actions ----------
states = [to_board(s) for s in batch.states]
actions = [tuple(int(v) for v in a) for a in batch.actions]
final_board = to_board(batch.get_terminating_states()[0])
print("n states:", len(states), "| n actions:", len(actions))
print("actions:", actions)

# ---------- 3. Build frames: (board, label) ----------
EOS = tuple(int(v) for v in env.eos)


def describe(a):
    if a == EOS:
        return "EOS (stop)"
    try:
        return f"piece {env.idx2piece[a[0]]}, rot {a[1]}, pos {a[2]}"
    except Exception:
        return str(a)


frames = []
if len(states) == 0:
    raise SystemExit("batch.states is empty - paste dir(batch)")

source_first = states[0].sum() == 0
if source_first:
    # states[t] is the board BEFORE actions[t]
    frames.append((states[0], "start"))
    for t in range(1, len(states)):
        frames.append((states[t], f"step {t}: {describe(actions[t - 1])}"))
    frames.append((final_board, f"step {len(states)}: {describe(actions[-1])}"))
else:
    # states[t] is the board AFTER actions[t]
    frames.append((np.zeros_like(states[0]), "start"))
    for t, s in enumerate(states):
        frames.append((s, f"step {t + 1}: {describe(actions[t])}"))

print("n frames:", len(frames))
if len(frames) <= 2:
    print("Still only 2 frames. Run print([a for a in dir(batch) if not a.startswith('_')]) and share it.")

# ---------- 4. Plot ----------
from matplotlib.colors import to_rgb

COLOR_BY = "piece"     # "piece" -> one color per piece; "type" -> one color per piece type
MAX_PER_TYPE = getattr(env, "max_pieces_per_type", 100)

all_boards = [b for b, _ in frames]
all_ids = sorted({int(v) for b in all_boards for v in np.unique(b) if v > 0})


def color_key(pid):
    return pid // MAX_PER_TYPE if COLOR_BY == "type" else pid


keys = sorted({color_key(pid) for pid in all_ids})
palette = plt.get_cmap("tab10" if COLOR_BY == "type" else "tab20")
n_colors = palette.N
color_of = {k: to_rgb(palette(i % n_colors)) for i, k in enumerate(keys)}
print("color mapping (key -> rgb):", {k: tuple(round(c, 2) for c in v) for k, v in color_of.items()})


def board_to_rgb(board):
    img = np.ones((*board.shape, 3))          # white background
    for pid in np.unique(board[board > 0]):
        img[board == pid] = color_of[color_key(int(pid))]
    return img


def draw_board(ax, board, title):
    ax.clear()
    ax.imshow(board_to_rgb(board))
    ax.set_xticks(np.arange(-0.5, board.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, board.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)


n = len(frames)
cols = min(5, n)
rows = int(np.ceil(n / cols))
fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 4.2 * rows))
axes = np.atleast_1d(axes).ravel()
for i, ax in enumerate(axes):
    if i < n:
        draw_board(ax, *frames[i])
    else:
        ax.axis("off")
plt.tight_layout()
fig.savefig("./tetris_trajectory.png", dpi=150, bbox_inches="tight")
print("saved ./tetris_trajectory.png")

# ---------- 5. GIF ----------
fig2, ax2 = plt.subplots(figsize=(3, 5))
FuncAnimation(fig2, lambda t: draw_board(ax2, *frames[t]), frames=n, interval=800).save(
    "./tetris_trajectory.gif", writer=PillowWriter(fps=1.2))
print("saved ./tetris_trajectory.gif")