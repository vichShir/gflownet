import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pandas as pd
import argparse
import seaborn as sns
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation, PillowWriter

from gflownet.utils.common import load_gflownet_from_rundir
from tqdm import tqdm


def to_board(s):
    return torch.as_tensor(s).detach().cpu().numpy()


def occupied(board):
    return float((board > 0).sum())


def describe(a, EOS):
    if a == EOS:
        return "EOS (stop)"
    try:
        return f"piece {env.idx2piece[a[0]]}, rot {a[1]}, pos {a[2]}"
    except Exception:
        return str(a)


def color_key(pid, COLOR_BY, MAX_PER_TYPE):
    return pid // MAX_PER_TYPE if COLOR_BY == "type" else pid


def board_to_rgb(board, color_of, COLOR_BY, MAX_PER_TYPE):
    img = np.ones((*board.shape, 3))          # white background
    for pid in np.unique(board[board > 0]):
        img[board == pid] = color_of[color_key(int(pid), COLOR_BY, MAX_PER_TYPE)]
    return img


def draw_board(ax, board, title, color_of, COLOR_BY, MAX_PER_TYPE):
    ax.clear()
    ax.imshow(board_to_rgb(board, color_of, COLOR_BY, MAX_PER_TYPE))
    ax.set_xticks(np.arange(-0.5, board.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, board.shape[0], 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=0.6)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8)


def main(args):
    gflownet, run_config = load_gflownet_from_rundir(
        rundir=args.ckpt_path, device=args.device, no_wandb=True,
        print_config=False, load_last_checkpoint=not args.random,
    )
    env = gflownet.env

    gflownet.forward_policy.model.eval()
    gflownet.backward_policy.model.eval()

    # Follow the learned policy only (no exploration noise), if the attribute exists
    for name in ["random_action_prob", "random_action_prob_forward"]:
        if hasattr(gflownet, name):
            print(f"setting gflownet.{name} = 0.0 (was {getattr(gflownet, name)})")
            setattr(gflownet, name, 0.0)

    # ---------- 1. Sample full trajectories (train=True keeps all steps) ----------
    best = None
    occupied_stats = []
    with torch.no_grad():
        for _ in tqdm(range(args.n_tries)):
            batch, _ = gflownet.sample_batch(n_forward=1, train=True)
            final = to_board(batch.get_terminating_states()[0])
            r = occupied(final)
            occupied_stats.append(r)
            if best is None or r > best[0]:
                best = (r, batch)
    best_reward, batch = best
    random_tag = 'random' if args.random else 'trained'
    occupied_series = pd.Series(occupied_stats)
    occupied_series.to_csv(f'occupied_stats_{random_tag}.csv')
    print("best occupied cells among tries:", best_reward)
    print(occupied_series.describe())

    plt.figure(figsize=(7, 8))
    sns.boxplot(occupied_series, color='tab:blue' if random_tag == 'random' else 'tab:green')
    plt.ylim(-5, 63)
    plt.ylabel('Occupied cells', fontsize=18)
    plt.title(f'Distribution {args.n_tries} plays - {random_tag}', fontsize=22)
    plt.savefig(f"distribution_boxplot_{random_tag}.png", dpi=300, bbox_inches="tight")

    # ---------- 2. Extract states and actions ----------
    states = [to_board(s) for s in batch.states]
    actions = [tuple(int(v) for v in a) for a in batch.actions]
    final_board = to_board(batch.get_terminating_states()[0])
    print("n states:", len(states), "| n actions:", len(actions))
    print("actions:", actions)

    # ---------- 3. Build frames: (board, label) ----------
    EOS = tuple(int(v) for v in env.eos)

    frames = []
    if len(states) == 0:
        raise SystemExit("batch.states is empty - paste dir(batch)")

    source_first = states[0].sum() == 0
    if source_first:
        # states[t] is the board BEFORE actions[t]
        frames.append((states[0], "start"))
        for t in range(1, len(states)):
            frames.append((states[t], f"step {t}: {describe(actions[t - 1], EOS)}"))
        frames.append((final_board, f"step {len(states)}: {describe(actions[-1], EOS)}"))
    else:
        # states[t] is the board AFTER actions[t]
        frames.append((np.zeros_like(states[0]), "start"))
        for t, s in enumerate(states):
            frames.append((s, f"step {t + 1}: {describe(actions[t], EOS)}"))

    print("n frames:", len(frames))
    if len(frames) <= 2:
        print("Still only 2 frames. Run print([a for a in dir(batch) if not a.startswith('_')]) and share it.")

    # ---------- 4. Plot ----------
    COLOR_BY = "piece"     # "piece" -> one color per piece; "type" -> one color per piece type
    MAX_PER_TYPE = getattr(env, "max_pieces_per_type", 100)

    all_boards = [b for b, _ in frames]
    all_ids = sorted({int(v) for b in all_boards for v in np.unique(b) if v > 0})

    keys = sorted({color_key(pid, COLOR_BY, MAX_PER_TYPE) for pid in all_ids})
    palette = plt.get_cmap("tab10" if COLOR_BY == "type" else "tab20")
    n_colors = palette.N
    color_of = {k: to_rgb(palette(i % n_colors)) for i, k in enumerate(keys)}
    print("color mapping (key -> rgb):", {k: tuple(round(c, 2) for c in v) for k, v in color_of.items()})

    n = len(frames)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        if i < n:
            draw_board(ax, *frames[i], color_of, COLOR_BY, MAX_PER_TYPE)
        else:
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(f"./best_tetris_trajectory_{random_tag}.png", dpi=150, bbox_inches="tight")
    print(f"saved ./best_tetris_trajectory_{random_tag}.png")

    # ---------- 5. GIF ----------
    fig2, ax2 = plt.subplots(figsize=(3, 5))
    FuncAnimation(fig2, lambda t: draw_board(ax2, *frames[t], color_of, COLOR_BY, MAX_PER_TYPE), frames=n, interval=800).save(
        f"./best_tetris_trajectory_{random_tag}.gif", writer=PillowWriter(fps=1.2))
    print(f"saved ./best_tetris_trajectory_{random_tag}.gif")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path", 
        type=str,
        help="The path to the checkpoint"
    )
    parser.add_argument(
        "--random", 
        action='store_true',
        help="Random weights"
    )
    parser.add_argument(
        "--n_tries", 
        type=int, 
        default=10,
        help="Number of sampling tries"
    )
    parser.add_argument(
        "--device", 
        type=str,
        default='cuda', 
        help="cuda or cpu"
    )
    args = parser.parse_args()
    main(args)