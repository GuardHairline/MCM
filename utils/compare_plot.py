import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_two_acc_matrices(acc_matrix1, sessions1, acc_matrix2, sessions2, base_name1, base_name2):
    """
    绘制两个 acc_matrix 热力图进行比较，使用子图并排显示，
    坐标轴刻度采用各自 sessions 中的 session_name。
    同时，对于每个单元格，如果训练会话的 replay_sessions 中包含测试会话的 session_name，则在数值后加 "*"。
    """
    n1 = len(acc_matrix1)
    n2 = len(acc_matrix2)
    matrix1 = np.full((n1, n1), np.nan)
    matrix2 = np.full((n2, n2), np.nan)
    for i, row in enumerate(acc_matrix1):
        matrix1[i, :len(row)] = row
    for i, row in enumerate(acc_matrix2):
        matrix2[i, :len(row)] = row

    # 提取会话名称
    session_names1 = [s["session_name"] for s in sessions1][:n1]
    session_names2 = [s["session_name"] for s in sessions2][:n2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 第一幅图
    im1 = axes[0].imshow(matrix1, cmap='viridis', interpolation='nearest')
    axes[0].set_title(f'Acc Matrix: {base_name1}')
    axes[0].set_xlabel('Test Session')
    axes[0].set_ylabel('Train Session')
    axes[0].set_xticks(np.arange(n1))
    axes[0].set_yticks(np.arange(n1))
    axes[0].set_xticklabels(session_names1, rotation=45)
    axes[0].set_yticklabels(session_names1)
    fig.colorbar(im1, ax=axes[0], label='Accuracy (%)')
    # 在每个有效单元格中标注数值，并加上回放标记
    for i in range(n1):
        for j in range(n1):
            if not np.isnan(matrix1[i, j]):
                # 从训练会话（行 i）中提取 replay_sessions 列表
                replay_sessions = sessions1[i].get("details", {}).get("replay_sessions", [])
                # 从测试会话（列 j）中获取 session_name
                test_session_name = sessions1[j].get("session_name", "")
                if test_session_name in replay_sessions:
                    text_str = f"{matrix1[i, j]:.1f}*"
                else:
                    text_str = f"{matrix1[i, j]:.1f}"
                axes[0].text(j, i, text_str, ha="center", va="center", color="white", fontsize=8)

    # 第二幅图
    im2 = axes[1].imshow(matrix2, cmap='viridis', interpolation='nearest')
    axes[1].set_title(f'Acc Matrix: {base_name2}')
    axes[1].set_xlabel('Test Session')
    axes[1].set_ylabel('Train Session')
    axes[1].set_xticks(np.arange(n2))
    axes[1].set_yticks(np.arange(n2))
    axes[1].set_xticklabels(session_names2, rotation=45)
    axes[1].set_yticklabels(session_names2)
    fig.colorbar(im2, ax=axes[1], label='Accuracy (%)')
    for i in range(n2):
        for j in range(n2):
            if not np.isnan(matrix2[i, j]):
                replay_sessions = sessions2[i].get("details", {}).get("replay_sessions", [])
                test_session_name = sessions2[j].get("session_name", "")
                if test_session_name in replay_sessions:
                    text_str = f"{matrix2[i, j]:.1f}*"
                else:
                    text_str = f"{matrix2[i, j]:.1f}"
                axes[1].text(j, i, text_str, ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    os.makedirs("checkpoints/figures", exist_ok=True)
    image_path = os.path.join("checkpoints/figures", f"compare_acc_matrix_{base_name1}_vs_{base_name2}.png")
    plt.savefig(image_path)
    plt.show()


def plot_two_final_metrics_AA_AIA(sessions1, sessions2, base_name1, base_name2):
    """
    分别提取两个 JSON 文件中 sessions 的 AA 与 AIA 指标，
    绘制在一个图上比较。
    """

    def extract_AA_AIA(sessions):
        names = []
        AA, AIA = [], []
        for session in sessions:
            final_metrics = session.get("final_metrics", {})
            if final_metrics:
                names.append(session["session_name"])
                AA.append(final_metrics.get("AA", np.nan))
                AIA.append(final_metrics.get("AIA", np.nan))
        return names, AA, AIA

    names1, AA1, AIA1 = extract_AA_AIA(sessions1)
    names2, AA2, AIA2 = extract_AA_AIA(sessions2)

    x1 = np.arange(len(names1))
    x2 = np.arange(len(names2))

    plt.figure(figsize=(10, 6))
    plt.plot(x1, AA1, marker='o', linestyle='-', label=f'AA ({base_name1})')
    plt.plot(x1, AIA1, marker='o', linestyle='-', label=f'AIA ({base_name1})')
    plt.plot(x2, AA2, marker='s', linestyle='--', label=f'AA ({base_name2})')
    plt.plot(x2, AIA2, marker='s', linestyle='--', label=f'AIA ({base_name2})')

    plt.title('AA & AIA Comparison')
    plt.xlabel('Session Index')
    plt.ylabel('Metric Value')
    plt.xticks(ticks=x1, labels=names1, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("checkpoints/figures", exist_ok=True)
    image_path = os.path.join("checkpoints/figures", f"compare_AA_AIA_{base_name1}_vs_{base_name2}.png")
    plt.savefig(image_path)
    plt.show()


def plot_two_final_metrics_FM_BWT(sessions1, sessions2, base_name1, base_name2):
    """
    分别提取两个 JSON 文件中 sessions 的 FM 与 BWT 指标，
    绘制在一个图上比较。
    """

    def extract_FM_BWT(sessions):
        names = []
        FM, BWT = [], []
        for session in sessions:
            final_metrics = session.get("final_metrics", {})
            if final_metrics:
                names.append(session["session_name"])
                FM.append(final_metrics.get("FM", np.nan))
                BWT.append(final_metrics.get("BWT", np.nan))
        return names, FM, BWT

    names1, FM1, BWT1 = extract_FM_BWT(sessions1)
    names2, FM2, BWT2 = extract_FM_BWT(sessions2)

    x1 = np.arange(len(names1))
    x2 = np.arange(len(names2))

    plt.figure(figsize=(10, 6))
    plt.plot(x1, FM1, marker='o', linestyle='-', label=f'FM ({base_name1})')
    plt.plot(x1, BWT1, marker='o', linestyle='-', label=f'BWT ({base_name1})')
    plt.plot(x2, FM2, marker='s', linestyle='--', label=f'FM ({base_name2})')
    plt.plot(x2, BWT2, marker='s', linestyle='--', label=f'BWT ({base_name2})')

    plt.title('FM & BWT Comparison')
    plt.xlabel('Session Index')
    plt.ylabel('Metric Value')
    plt.xticks(ticks=x1, labels=names1, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs("checkpoints/figures", exist_ok=True)
    image_path = os.path.join("checkpoints/figures", f"compare_FM_BWT_{base_name1}_vs_{base_name2}.png")
    plt.savefig(image_path)
    plt.show()


def compare_results(json_file1, json_file2):
    """
    加载两个 JSON 文件的结果，比较它们的 acc_matrix 和 CL 指标，
    并分别生成对比图（热力图和两个 CL 指标图）。
    """
    with open(json_file1, "r", encoding="utf-8") as f:
        train_info1 = json.load(f)
    with open(json_file2, "r", encoding="utf-8") as f:
        train_info2 = json.load(f)

    acc_matrix1 = train_info1.get("acc_matrix", [])
    sessions1 = train_info1.get("sessions", [])
    acc_matrix2 = train_info2.get("acc_matrix", [])
    sessions2 = train_info2.get("sessions", [])

    base_name1 = os.path.splitext(os.path.basename(json_file1))[0]
    base_name2 = os.path.splitext(os.path.basename(json_file2))[0]

    plot_two_acc_matrices(acc_matrix1, sessions1, acc_matrix2, sessions2, base_name1, base_name2)
    plot_two_final_metrics_AA_AIA(sessions1, sessions2, base_name1, base_name2)
    plot_two_final_metrics_FM_BWT(sessions1, sessions2, base_name1, base_name2)


def main():
    json_file1 = "checkpoints/train_info_twitter2017_none_text.json"
    json_file2 = "checkpoints/train_info_twitter2017_replay_text.json"
    compare_results(json_file1, json_file2)


if __name__ == "__main__":
    main()
