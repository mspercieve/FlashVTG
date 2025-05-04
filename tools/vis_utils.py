import sys
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from easydict import EasyDict as edict
import matplotlib.pyplot as plt


def get_chunk_ranges(total_frames, num_chunks=3):
    """
    전체 프레임 수에 맞춰 num_chunks(기본 3)로 등분한 구간 리스트 반환.
    각 구간은 (start, end) 형태이며, end는 포함.
    """
    chunk_size = total_frames // num_chunks
    ranges = []
    for i in range(num_chunks):
        start = i * chunk_size
        # 마지막 구간은 전체 프레임을 포함하도록 함.
        end = (total_frames - 1) if i == num_chunks - 1 else (i * chunk_size + chunk_size - 1)
        ranges.append((start, end))
    return ranges

def sample_frames_in_chunk(frames, chunk_start, chunk_end, sample_interval=5):
    """
    주어진 구간 [chunk_start, chunk_end] 내에서 sample_interval 간격으로 프레임 sampling.
    만약 마지막 프레임이 구간 내에 있고 sampling에서 누락되면 추가.
    """
    idxs = []
    f = chunk_start
    while f <= chunk_end:
        idxs.append(f)
        f += sample_interval
    if idxs[-1] < chunk_end:
        idxs.append(chunk_end)
    idxs = sorted(list(set(idxs)))
    sampled = frames[idxs]  # shape (n, C, H, W)
    return sampled, idxs

def intersection_interval(interval_a, interval_b):
    s = max(interval_a[0], interval_b[0])
    e = min(interval_a[1], interval_b[1])
    if s <= e:
        return (s, e)
    return None

def highlight_cells_red_outline(ax, table, cell_dict, row_start, row_end, col_start, col_end):
    xs, ys = [], []
    renderer = ax.figure.canvas.get_renderer()
    for row in range(row_start, row_end+1):
        for col in range(col_start, col_end+1):
            if (row, col) in cell_dict:
                cell = cell_dict[(row, col)]
                bbox = cell.get_window_extent(renderer)
                xs.extend([bbox.x0, bbox.x1])
                ys.extend([bbox.y0, bbox.y1])
    if not xs or not ys:
        return
    x_min = min(xs)
    x_max = max(xs)
    y_min = min(ys)
    y_max = max(ys)
    rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                         fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(rect)

def plot_chunk_figure(query, moment_gt, frames, token_list, attn_map,
                      chunk_range, chunk_idx, vis_path, vid):
    start_c, end_c = chunk_range
    fig = plt.figure(figsize=(28, 12), constrained_layout=True)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.12, 0.35, 0.53])

    # =============== (1) 상단: Query, GT moment, Chunk range 한 줄 ===============
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.axis("off")
    moment_str_list = []
    for sub in moment_gt:
        moment_str_list.append("~".join(str(int(x)) for x in sub))
    moment_str = " / ".join(moment_str_list)
    text_top = f"Query: {query}\tGT moment: {moment_str}\tChunk range: {start_c}~{end_c}"
    ax_top.text(0.5, 0.5, text_top, ha="center", va="center", fontsize=18)

    # =============== (2) 중간: 샘플링된 frames (가로 나열) ===============
    ax_mid = fig.add_subplot(gs[1, 0])
    ax_mid.axis("off")
    total_frames = frames.shape[0]
    sampled_frames, sampled_idx = sample_frames_in_chunk(frames, start_c, end_c, sample_interval=5)
    num_sampled = sampled_frames.shape[0]
    x_offset = 0.0
    gap = 0.02
    width_each = 1.0 / num_sampled - gap
    for i in range(num_sampled):
        f_img = np.transpose(sampled_frames[i], (1, 2, 0))
        ax_mid.imshow(f_img.astype(np.uint8),
                      extent=(x_offset, x_offset+width_each, 0, 1),
                      aspect='auto')
        ax_mid.text(x_offset + width_each/2, 0.95, f"Frame {sampled_idx[i]}",
                    ha="center", va="top", fontsize=14, color="white",
                    bbox=dict(boxstyle="round", facecolor="black", alpha=0.5))
        x_offset += (width_each + gap)
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(0, 1)

    # =============== (3) 하단: Attention Table (해당 구간의 frame 개수만큼) ===============
    ax_bot = fig.add_subplot(gs[2, 0])
    ax_bot.axis("off")
    subT = end_c - start_c + 1
    sub_attn = attn_map[start_c:end_c+1]  # shape (subT, L)
    if isinstance(sub_attn, torch.Tensor):
        sub_attn = sub_attn.detach().cpu().numpy()
    sub_attn = sub_attn.T  # (L, subT)
    L = len(token_list)

    header = ["index", "token"] + [str(f) for f in range(start_c, end_c+1)]
    table_data = [header]
    for i in range(L):
        row = [str(i), token_list[i]]
        for t_idx in range(subT):
            val = sub_attn[i, t_idx]
            # 값이 0이면 빈 문자열로 처리
            cell_val = "" if abs(val) < 1e-2 else f"{val:.2f}"
            row.append(cell_val)
        table_data.append(row)
    # 마지막 행: sum
    sum_row = ["sum", ""]
    for t_idx in range(subT):
        col_vals = sub_attn[:, t_idx]
        col_sum = np.sum(col_vals)
        sum_cell = "" if abs(col_sum) < 1e-2 else f"{col_sum:.2f}"
        sum_row.append(sum_cell)
    table_data.append(sum_row)

    # 테이블 생성: bbox로 축 전체 사용
    the_table = ax_bot.table(cellText=table_data,
                             cellLoc='center',
                             loc='upper left',
                             bbox=[0, 0, 1, 1])
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(12)
    ncol = len(header)
    for col in range(ncol):
        the_table.auto_set_column_width(col)
    the_table.scale(2.5, 1.4)
    fig.canvas.draw()  # 렌더러 업데이트

    # 빨간 박스: GT moment와 겹치는 부분에 대해
    cell_dict = the_table.get_celld()
    for interval in moment_gt:
        inter = intersection_interval(interval, (start_c, end_c))
        if inter is not None:
            s_i, e_i = inter
            col_start = (s_i - start_c) + 2
            col_end = (e_i - start_c) + 2
            # row 1부터 row L+1 (토큰 + sum 행)
            highlight_cells_red_outline(ax_bot, the_table, cell_dict,
                                        row_start=1, row_end=L+1,
                                        col_start=col_start, col_end=col_end)

    # 셀의 텍스트가 빈 문자열인 경우, 배경색을 옅은 회색으로 변경
    for key, cell in cell_dict.items():
        if key[0] > 0:  # 헤더 제외
            if cell.get_text().get_text().strip() == "":
                cell.set_facecolor("lightgray")

    chunk_save_dir = os.path.join(vis_path)
    os.makedirs(chunk_save_dir, exist_ok=True)
    save_filename = os.path.join(chunk_save_dir, f"chunk_{chunk_idx}.png")
    plt.savefig(save_filename, bbox_inches="tight")
    plt.close(fig)
    print("Saved chunk figure:", save_filename)


def visualize_similarity_matrix(tokens, t_sim, t_proj_sim, qd_t_proj_sim, query_text="", save_path=None):
    """
    tokens: list of str, length L
    t_sim, t_proj_sim, qd_t_proj_sim: torch.Tensor or np.ndarray of shape (L, L)
    query_text: original input query string
    save_path: optional path to save the image
    """
    if isinstance(t_sim, torch.Tensor):
        t_sim = t_sim.detach().cpu().numpy()
    if isinstance(t_proj_sim, torch.Tensor):
        t_proj_sim = t_proj_sim.detach().cpu().numpy()
    if isinstance(qd_t_proj_sim, torch.Tensor):
        qd_t_proj_sim = qd_t_proj_sim.detach().cpu().numpy()

    L = len(tokens)
    fig, ax = plt.subplots(figsize=(L * 0.5 + 2, L * 0.5 + 2))

    # 이미지 매트릭스 셋업
    ax.set_xticks(np.arange(L))
    ax.set_yticks(np.arange(L))
    ax.set_xticklabels(tokens, fontsize=12, rotation=45, ha="left")
    ax.set_yticklabels(tokens, fontsize=12)

    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # 셀 구분용 그리드
    ax.set_xticks(np.arange(L + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(L + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.invert_yaxis()

    # 셀 채우기
    for i in range(L):
        for j in range(L):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color="lightgray", zorder=0))
            else:
                ax.text(j, i - 0.25, f"{t_sim[i, j]:.3f}", ha="center", va="center", color="black", fontsize=11)
                ax.text(j, i,        f"{t_proj_sim[i, j]:.3f}", ha="center", va="center", color="red", fontsize=11)
                ax.text(j, i + 0.25, f"{qd_t_proj_sim[i, j]:.3f}", ha="center", va="center", color="blue", fontsize=11)

    # Query 문장 제목
    plt.title(f"Query: {query_text}", fontsize=14, pad=30)

    # 범례: 아래쪽에 선으로 표현
    legend_elements = [
        plt.Line2D([0], [0], color='black', lw=2, label='Llama'),
        plt.Line2D([0], [0], color='red', lw=2, label='Flash-VTG'),
        plt.Line2D([0], [0], color='blue', lw=2, label='QD-DETR'),
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=3, frameon=False, fontsize=11)

    plt.tight_layout()
    if save_path:
        save_filename = os.path.join(save_path, "sim_mat.png")
        plt.savefig(save_filename, dpi=300)
        plt.close()
    else:
        plt.show()

def visualize_phrase_clusters(query, tokens, sqan_attn, slot_attn, save_path=None):
    """
    query: str - input text query
    tokens: list of str - tokenized words from the query (length L)
    sqan_attn: np.ndarray [N, L] - SQAN attention scores
    slot_attn: np.ndarray [N, L] - Slot attention scores (after phrase refinement)
    save_path: str or None - if set, save the figure
    """
    N, L = sqan_attn.shape
    fig, axes = plt.subplots(2, 1, figsize=(min(20, L), 3 + N * 0.7 * 2), constrained_layout=True)

    titles = ['SQAN Attention (Initial Clustering)', 'Slot Attention (After Refinement)']
    attn_matrices = [sqan_attn, slot_attn]

    for ax, attn, title in zip(axes, attn_matrices, titles):
        table_data = []
        for n in range(N):
            row = [f"{attn[n, l]:.2f}" for l in range(L)]
            table_data.append(row)

        col_labels = [f"{i}:{tok}" for i, tok in enumerate(tokens)]
        row_labels = [f"Phrase {n+1}" for n in range(N)]

        ax.axis("off")
        table = ax.table(cellText=table_data,
                         rowLabels=row_labels,
                         colLabels=col_labels,
                         loc='center',
                         cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        ax.set_title(title, fontsize=12, pad=10)

    fig.suptitle(f"Query: {query}", fontsize=14, y=1.02)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved attention visualization to: {save_path}")
    else:
        plt.show()