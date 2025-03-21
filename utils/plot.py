import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_acc_matrix(acc_matrix, sessions, base_name):
    """
    acc_matrix 是一个列表列表，每一行的长度依次增加，表示
    第 i 次任务训练结束后在前 i 个任务上的测试准确率。
    我们构造一个 n x n 的矩阵，未定义的位置填为 NaN，
    然后用 imshow 绘制热力图，并在每个有效位置上标注数值。
    """
    n = len(acc_matrix)
    matrix = np.full((n, n), np.nan)
    for i, row in enumerate(acc_matrix):
        # 每行应填入 row 的所有元素（从左起）
        matrix[i, :len(row)] = row

    plt.figure(figsize=(6, 5))
    im = plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.title('Accuracy Matrix')
    # 提取前 n 个会话名称作为标签
    session_names = [s["session_name"] for s in sessions][:n]
    plt.xticks(ticks=np.arange(n), labels=session_names, rotation=45)
    plt.yticks(ticks=np.arange(n), labels=session_names)
    plt.xlabel('Test Task Index')
    plt.ylabel('Train Task Index')
    plt.colorbar(im, label='Accuracy (%)')
    # 在每个有效的单元格中标注数值
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                # 从训练会话（行 i）中提取 replay_sessions 列表
                replay_sessions = sessions[i].get("details", {}).get("replay_sessions", [])
                # 从测试会话（列 j）中获取 session_name
                test_session_name = sessions[j].get("session_name", "")
                # 如果测试会话的 session_name 出现在训练会话的 replay_sessions 列表中，则加上 *
                if test_session_name in replay_sessions:
                    text_str = f"{matrix[i, j]:.1f}*"
                else:
                    text_str = f"{matrix[i, j]:.1f}"
                plt.text(j, i, text_str, ha="center", va="center", color="white", fontsize=10)
    plt.tight_layout()

    # 创建保存图片的目录
    os.makedirs("checkpoints/figures", exist_ok=True)
    # 生成图片保存路径
    image_name = f"{base_name}_acc_matrix.png"
    image_path = os.path.join("checkpoints/figures", image_name)
    # 保存图片
    plt.savefig(image_path)

    plt.show()


def plot_final_metrics(sessions, base_name):
    """
    从 sessions 中提取每次训练结束后的 CL 指标（final_metrics）。
    注意第一个任务通常没有 CL 指标，所以我们只绘制有指标的后续任务。
    绘制 AA、AIA、FM、BWT 随任务编号的变化曲线。
    """
    task_indices = []
    AA = []
    AIA = []
    FM = []
    BWT = []
    # 遍历 sessions，假设 sessions 顺序即为训练顺序
    for idx, session in enumerate(sessions):
        final_metrics = session.get("final_metrics", {})
        # 如果指标不为空，则认为该任务训练后有 CL 指标（通常从第二个任务开始）
        if final_metrics:
            task_indices.append(idx + 1)  # 任务编号从1开始
            AA.append(final_metrics.get("AA", np.nan))
            AIA.append(final_metrics.get("AIA", np.nan))
            FM.append(final_metrics.get("FM", np.nan))
            BWT.append(final_metrics.get("BWT", np.nan))

    if not task_indices:
        print("没有找到有效的 final_metrics 数据，无法绘制 CL 指标图。")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(task_indices, AA, marker='o', label='AA')
    plt.plot(task_indices, AIA, marker='o', label='AIA')
    plt.plot(task_indices, FM, marker='o', label='FM')
    plt.plot(task_indices, BWT, marker='o', label='BWT')
    plt.title('Continual Learning Metrics over Tasks')
    plt.xlabel('Task Index')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 生成图片保存路径
    image_name = f"{base_name}_continual_learning_metrics.png"
    image_path = os.path.join("checkpoints/figures", image_name)
    # 保存图片
    plt.savefig(image_path)
    plt.show()

def main():
    os.makedirs("checkpoints", exist_ok=True)

    # 修改下面路径为你实际的 train_info.json 文件路径
    json_file = "checkpoints/train_info_twitterMix_replay_text.json"
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    with open(json_file, "r", encoding="utf-8") as f:
        train_info = json.load(f)

    acc_matrix = train_info.get("acc_matrix", [])
    sessions = train_info.get("sessions", [])

    # 绘制 acc_matrix 热力图
    plot_acc_matrix(acc_matrix, sessions, base_name)
    # 绘制 CL 指标随任务变化的趋势图
    plot_final_metrics(sessions, base_name)


if __name__ == "__main__":
    main()
