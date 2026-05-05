from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


def compute_embedding(latent_vectors: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(latent_vectors)


def row_normalize(mat: np.ndarray) -> np.ndarray:
    mat = mat.astype(float)
    row_sum = mat.sum(axis=1, keepdims=True)
    out = np.zeros_like(mat, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(mat, row_sum, out=out, where=row_sum > 0)
    out[np.isnan(out)] = 0.0
    return out


def community_transition_probs(labels: np.ndarray, n_communities: int) -> np.ndarray:
    counts = np.zeros((n_communities, n_communities), dtype=float)
    for a, b in zip(labels[:-1], labels[1:]):
        counts[int(a), int(b)] += 1.0
    return row_normalize(counts)


def estimate_speed(selected_csv: Path, target_len: int) -> np.ndarray:
    df = pd.read_csv(selected_csv)
    # Motion index from ankle displacement; robust for dance footwork dynamics.
    ldx = np.diff(df["left_ankle_x"].to_numpy(), prepend=df["left_ankle_x"].iloc[0])
    ldy = np.diff(df["left_ankle_y"].to_numpy(), prepend=df["left_ankle_y"].iloc[0])
    rdx = np.diff(df["right_ankle_x"].to_numpy(), prepend=df["right_ankle_x"].iloc[0])
    rdy = np.diff(df["right_ankle_y"].to_numpy(), prepend=df["right_ankle_y"].iloc[0])
    speed = 0.5 * (np.sqrt(ldx**2 + ldy**2) + np.sqrt(rdx**2 + rdy**2))

    if len(speed) >= target_len:
        return speed[:target_len]

    pad = np.full(target_len - len(speed), np.nanmean(speed))
    return np.concatenate([speed, pad])


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def estimate_turning(selected_csv: Path, target_len: int) -> np.ndarray:
    df = pd.read_csv(selected_csv)

    # Heading proxy from torso center to nose captures spin/turn phases in 2D view.
    cx = 0.25 * (
        df["left_shoulder_x"].to_numpy()
        + df["right_shoulder_x"].to_numpy()
        + df["left_hip_x"].to_numpy()
        + df["right_hip_x"].to_numpy()
    )
    cy = 0.25 * (
        df["left_shoulder_y"].to_numpy()
        + df["right_shoulder_y"].to_numpy()
        + df["left_hip_y"].to_numpy()
        + df["right_hip_y"].to_numpy()
    )

    dx = df["nose_x"].to_numpy() - cx
    dy = df["nose_y"].to_numpy() - cy

    heading = np.arctan2(dy, dx)
    dtheta = wrap_to_pi(np.diff(heading, prepend=heading[0]))
    turn_rate = np.abs(dtheta)
    turn_rate = gaussian_filter(turn_rate, sigma=2.0)

    if len(turn_rate) >= target_len:
        return turn_rate[:target_len]

    pad = np.full(target_len - len(turn_rate), np.nanmean(turn_rate))
    return np.concatenate([turn_rate, pad])


def behavioral_names_from_speed(
    comm_labels: np.ndarray,
    speed: np.ndarray,
    turn_rate: np.ndarray,
) -> Dict[int, str]:
    n_communities = int(comm_labels.max()) + 1
    means: List[Tuple[int, float]] = []
    for cid in range(n_communities):
        mask = comm_labels == cid
        means.append((cid, float(np.nanmean(speed[mask])) if mask.any() else 0.0))

    means_sorted = sorted(means, key=lambda x: x[1])
    presets = [
        "Idle",
        "Slow",
        "Locomotion",
        "Turn/Spin",
        "Anterior",
        "Side Leg",
    ]

    name_map: Dict[int, str] = {}
    for rank, (cid, _) in enumerate(means_sorted):
        if rank < len(presets):
            name_map[cid] = presets[rank]
        else:
            name_map[cid] = f"State {cid}"

    # Ensure the highest-rotation community is explicitly marked.
    n_communities = int(comm_labels.max()) + 1
    turn_means = []
    for cid in range(n_communities):
        mask = comm_labels == cid
        turn_means.append((cid, float(np.nanmean(turn_rate[mask])) if mask.any() else 0.0))
    turn_comm = max(turn_means, key=lambda x: x[1])[0]
    name_map[turn_comm] = "Turn/Spin"

    return name_map


def draw_primary_figure(
    embedding: np.ndarray,
    motif_labels: np.ndarray,
    comm_labels: np.ndarray,
    motif_transition_probs: np.ndarray,
    comm_transition_probs: np.ndarray,
    name_map: Dict[int, str],
    turn_rate: Optional[np.ndarray],
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-white")
    fig = plt.figure(figsize=(14, 10), dpi=170)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.22, wspace=0.2)

    colors = np.array(["#7f1d1d", "#1d4ed8", "#6d28d9", "#0f766e", "#c2410c", "#1f2937"])

    x, y = embedding[:, 0], embedding[:, 1]

    ax_a = fig.add_subplot(gs[0, 0])
    kde = gaussian_kde(np.vstack([x, y]))
    xx, yy = np.meshgrid(
        np.linspace(x.min() - 1.0, x.max() + 1.0, 240),
        np.linspace(y.min() - 1.0, y.max() + 1.0, 240),
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    im = ax_a.imshow(
        zz,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="plasma",
        aspect="auto",
    )
    cb_a = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.02)
    cb_a.set_label("PDF", rotation=270, labelpad=14, fontsize=12)
    ax_a.set_xticks([])
    ax_a.set_yticks([])
    ax_a.set_title("A", loc="left", fontweight="bold", fontsize=20)

    ax_b = fig.add_subplot(gs[0, 1])
    order = np.argsort(comm_labels)
    sorted_motif = motif_labels[order]
    n_motifs = motif_transition_probs.shape[0]
    motif_tm_sorted = motif_transition_probs[np.ix_(sorted(range(n_motifs)), sorted(range(n_motifs)))]
    hm = ax_b.imshow(motif_tm_sorted, cmap="cividis", vmin=0, vmax=max(0.15, motif_tm_sorted.max()))
    cb_b = fig.colorbar(hm, ax=ax_b, fraction=0.046, pad=0.02)
    cb_b.set_label("T(1)", rotation=270, labelpad=14, fontsize=12)
    ax_b.set_xlabel("Final State", fontsize=12, fontstyle="italic")
    ax_b.set_ylabel("Initial State", fontsize=12, fontstyle="italic")
    ax_b.set_title("B", loc="left", fontweight="bold", fontsize=20)

    sorted_comm = comm_labels[order]
    bounds = []
    start = 0
    for idx in range(1, len(sorted_comm) + 1):
        if idx == len(sorted_comm) or sorted_comm[idx] != sorted_comm[start]:
            bounds.append((start, idx - start, int(sorted_comm[start])))
            start = idx

    total_len = max(len(sorted_comm), 1)
    for st, length, cid in bounds:
        rect = Rectangle((st - 0.5, st - 0.5), length, length, fill=False, edgecolor="black", linewidth=2)
        ax_b.add_patch(rect)
        xloc = (st + length / 2) / total_len
        ax_b.text(
            xloc,
            1.02,
            name_map.get(cid, f"C{cid}"),
            transform=ax_b.transAxes,
            ha="center",
            va="bottom",
            rotation=25,
            fontsize=10,
        )

    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.imshow(
        zz,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        cmap="BuPu",
        alpha=0.6,
        aspect="auto",
    )

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(x) - 1, size=min(260, len(x) - 1), replace=False)
    for i in sample_idx:
        ax_c.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color="black", alpha=0.23, linewidth=0.8)
    ax_c.scatter(x[sample_idx], y[sample_idx], s=8, c="#b91c1c", alpha=0.75)

    if turn_rate is not None and len(turn_rate) == len(x):
        q = np.nanquantile(turn_rate, 0.94)
        turn_idx = np.where(turn_rate >= q)[0]
        if len(turn_idx) > 0:
            turn_idx = turn_idx[: min(120, len(turn_idx))]
            ax_c.scatter(
                x[turn_idx],
                y[turn_idx],
                s=22,
                c="#f59e0b",
                edgecolors="black",
                linewidths=0.35,
                alpha=0.85,
                zorder=6,
                label="High turn/spin frames",
            )
            ax_c.legend(loc="upper left", fontsize=8, frameon=True)

    ax_c.set_xticks([])
    ax_c.set_yticks([])
    ax_c.set_title("C", loc="left", fontweight="bold", fontsize=20)

    ax_d = fig.add_subplot(gs[1, 1])
    n_communities = int(comm_labels.max()) + 1

    xg, yg = np.meshgrid(
        np.linspace(x.min() - 1.0, x.max() + 1.0, 360),
        np.linspace(y.min() - 1.0, y.max() + 1.0, 360),
    )
    zg = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)

    centroids = np.zeros((n_communities, 2), dtype=float)
    for cid in range(n_communities):
        pts = embedding[comm_labels == cid]
        centroids[cid] = pts.mean(axis=0)

    dist_stack = np.stack([(xg - cx) ** 2 + (yg - cy) ** 2 for cx, cy in centroids], axis=-1)
    nearest = np.argmin(dist_stack, axis=-1)

    density_mask = zg > np.quantile(zg, 0.35)
    density_mask = gaussian_filter(density_mask.astype(float), sigma=2.0) > 0.2

    partition = np.where(density_mask, nearest, -1)

    palette = colors[: max(n_communities, 1)]
    for cid in range(n_communities):
        mask = partition == cid
        if not mask.any():
            continue
        ax_d.contourf(
            xg,
            yg,
            mask.astype(float),
            levels=[0.5, 1.5],
            colors=[palette[cid % len(palette)]],
            alpha=0.42,
        )
        ax_d.contour(
            xg,
            yg,
            mask.astype(float),
            levels=[0.5],
            colors="black",
            linewidths=2,
            alpha=0.9,
        )

    for cid in range(n_communities):
        cx, cy = centroids[cid]
        ax_d.scatter([cx], [cy], s=45, c="#dc2626", zorder=5)
        ax_d.text(
            cx,
            cy,
            name_map.get(cid, f"C{cid}"),
            fontsize=14,
            fontstyle="italic",
            fontweight="bold",
            color=palette[cid % len(palette)],
            ha="left",
            va="bottom",
        )

    for i in range(n_communities):
        for j in range(n_communities):
            p = comm_transition_probs[i, j]
            if p < 0.05 or i == j:
                continue
            x0, y0 = centroids[i]
            x1, y1 = centroids[j]
            ax_d.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    linewidth=1.0 + 11.0 * float(p),
                    alpha=0.88,
                    shrinkA=10,
                    shrinkB=10,
                    connectionstyle="arc3,rad=0.14",
                ),
                zorder=4,
            )

    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.set_title("D", loc="left", fontweight="bold", fontsize=20)

    fig.suptitle("Irish Dance Behavioral Space and Information-Bottleneck Communities", fontsize=17, fontweight="bold")
    fig.savefig(out_path)
    plt.close(fig)


def transition_matrix_from_sequence(seq: np.ndarray, n_states: int) -> np.ndarray:
    counts = np.zeros((n_states, n_states), dtype=float)
    for a, b in zip(seq[:-1], seq[1:]):
        counts[int(a), int(b)] += 1.0
    return row_normalize(counts)


def top_edges(mat: np.ndarray, top_k: int) -> List[Tuple[int, int, float]]:
    n = mat.shape[0]
    edges: List[Tuple[int, int, float]] = []
    for i in range(n):
        for j in range(n):
            if i != j and mat[i, j] > 0:
                edges.append((i, j, float(mat[i, j])))
    edges.sort(key=lambda t: t[2], reverse=True)
    return edges[:top_k]


def draw_directed_motif_graph(
    ax: plt.Axes,
    motif_ids: List[int],
    edge_list: List[Tuple[int, int, float]],
    title: str,
    edge_color: str = "#991b1b",
    show_labels: bool = True,
) -> None:
    n = len(motif_ids)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {mid: (np.cos(t), np.sin(t)) for mid, t in zip(motif_ids, theta)}

    for mid in motif_ids:
        x, y = pos[mid]
        ax.scatter([x], [y], s=230, c="#93c5fd", edgecolors="#1e3a8a", linewidths=1.4, zorder=4)
        ax.text(x, y, str(mid), ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a", zorder=5)

    for i, j, w in edge_list:
        x0, y0 = pos[i]
        x1, y1 = pos[j]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=edge_color,
                linewidth=1.0 + 15.0 * w,
                alpha=0.9,
                shrinkA=15,
                shrinkB=15,
                connectionstyle="arc3,rad=0.18",
            ),
            zorder=3,
        )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_labels:
        ax.text(
            0.5,
            -0.07,
            "Arrow width = transition probability",
            transform=ax.transAxes,
            ha="center",
            fontsize=9,
            color="#334155",
        )


def draw_intra_community_figure(
    motif_labels: np.ndarray,
    comm_labels: np.ndarray,
    walking_comm: int,
    community_label: str,
    out_path: Path,
) -> None:
    plt.style.use("seaborn-v0_8-white")

    indices = np.where(comm_labels == walking_comm)[0]
    if len(indices) < 8:
        raise ValueError("Not enough points in selected walking community for intra-community plot")

    motif_seq = motif_labels[indices]
    motif_ids = sorted(np.unique(motif_seq).tolist())
    id_to_local = {mid: i for i, mid in enumerate(motif_ids)}
    local_seq = np.array([id_to_local[m] for m in motif_seq], dtype=int)

    split = len(local_seq) // 2
    g1_seq = local_seq[:split]
    g2_seq = local_seq[split:]

    g1_tm = transition_matrix_from_sequence(g1_seq, len(motif_ids))
    g2_tm = transition_matrix_from_sequence(g2_seq, len(motif_ids))
    diff = g2_tm - g1_tm

    g1_edges_local = top_edges(g1_tm, top_k=10)
    g2_edges_local = top_edges(g2_tm, top_k=10)

    diff_edges = []
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            if i != j:
                diff_edges.append((i, j, float(diff[i, j])))
    diff_edges.sort(key=lambda t: abs(t[2]), reverse=True)
    diff_edges = [e for e in diff_edges if abs(e[2]) > 0.01][:12]

    def to_global(edges_local: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
        out: List[Tuple[int, int, float]] = []
        for i, j, w in edges_local:
            out.append((motif_ids[i], motif_ids[j], w))
        return out

    g1_edges = to_global(g1_edges_local)
    g2_edges = to_global(g2_edges_local)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=220)
    draw_directed_motif_graph(axes[0], motif_ids, g1_edges, "Group 1: Top Intra-Community Transitions", edge_color="#dc2626")
    draw_directed_motif_graph(axes[1], motif_ids, g2_edges, "Group 2: Top Intra-Community Transitions", edge_color="#16a34a")

    # Difference graph uses alternating color by sign of change.
    n = len(motif_ids)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {mid: (np.cos(t), np.sin(t)) for mid, t in zip(motif_ids, theta)}

    for mid in motif_ids:
        x, y = pos[mid]
        axes[2].scatter([x], [y], s=230, c="#93c5fd", edgecolors="#1e3a8a", linewidths=1.4, zorder=4)
        axes[2].text(x, y, str(mid), ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a", zorder=5)

    for i_local, j_local, w_signed in diff_edges:
        sign = diff[i_local, j_local]
        color = "#dc2626" if sign > 0 else "#16a34a"
        x0, y0 = pos[motif_ids[i_local]]
        x1, y1 = pos[motif_ids[j_local]]
        axes[2].annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=color,
                linewidth=1.0 + 18.0 * w_signed,
                alpha=0.92,
                shrinkA=15,
                shrinkB=15,
                connectionstyle="arc3,rad=0.18",
            ),
            zorder=3,
        )

    axes[2].set_title("Highest Transition Differences (G2 - G1)", fontsize=12, fontweight="bold")
    axes[2].set_aspect("equal")
    axes[2].set_xlim(-1.2, 1.2)
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    axes[2].text(
        0.5,
        -0.07,
        "Red: stronger in Group 2, Green: stronger in Group 1",
        transform=axes[2].transAxes,
        ha="center",
        fontsize=9,
        color="#334155",
    )

    fig.suptitle(
        f"Intra-Community Transition Graph for {community_label} Dance Community",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    results_root = project_root / "irish-dance-vame" / "results"
    figures_root = project_root / "irish-dance-vame" / "reports" / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)

    latent = np.load(results_root / "irish_dance_pose_dlc" / "VAME" / "latent_vectors.npy")
    motif_labels = np.load(results_root / "irish_dance_pose_dlc" / "VAME" / "hmm-8" / "8_hmm_label_irish_dance_pose_dlc.npy")
    comm_labels = np.load(
        results_root
        / "irish_dance_pose_dlc"
        / "VAME"
        / "hmm-8"
        / "community"
        / "cohort_community_label_irish_dance_pose_dlc.npy"
    )

    motif_transition_counts = np.load(results_root / "community_cohort" / "hmm-8" / "cohort_transition_matrix.npy")
    motif_transition_probs = row_normalize(motif_transition_counts)

    embedding = compute_embedding(latent)
    comm_transition_probs = community_transition_probs(comm_labels, int(comm_labels.max()) + 1)

    speed = estimate_speed(project_root / "results_dance_pose" / "irish_dance_pose_selected.csv", len(comm_labels))
    turn_rate = estimate_turning(project_root / "results_dance_pose" / "irish_dance_pose_selected.csv", len(comm_labels))
    name_map = behavioral_names_from_speed(comm_labels, speed, turn_rate)

    primary_path = figures_root / "irish_dance_information_bottleneck_communities.png"
    draw_primary_figure(
        embedding=embedding,
        motif_labels=motif_labels,
        comm_labels=comm_labels,
        motif_transition_probs=motif_transition_probs,
        comm_transition_probs=comm_transition_probs,
        name_map=name_map,
        turn_rate=turn_rate,
        out_path=primary_path,
    )

    # Use highest-motion multi-motif community as walking-like community.
    candidate_ids = []
    for cid in np.unique(comm_labels):
        motifs_here = np.unique(motif_labels[comm_labels == cid])
        if len(motifs_here) >= 3:
            candidate_ids.append(int(cid))

    if candidate_ids:
        comm_speed = {cid: np.nanmean(speed[comm_labels == cid]) for cid in candidate_ids}
        walking_comm = int(max(comm_speed, key=comm_speed.get))
    else:
        comm_speed = {int(cid): np.nanmean(speed[comm_labels == cid]) for cid in np.unique(comm_labels)}
        walking_comm = int(max(comm_speed, key=comm_speed.get))

    intra_path = figures_root / "irish_dance_intra_community_transitions.png"
    draw_intra_community_figure(
        motif_labels=motif_labels,
        comm_labels=comm_labels,
        walking_comm=walking_comm,
        community_label=name_map.get(walking_comm, "Turn/Spin-like"),
        out_path=intra_path,
    )

    print(f"Saved: {primary_path}")
    print(f"Saved: {intra_path}")
    print(f"Walking-like community id: {walking_comm} ({name_map.get(walking_comm, str(walking_comm))})")


if __name__ == "__main__":
    main()
