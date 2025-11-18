import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 为独立绘图工具配置简单的日志
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def plot_acc_matrix(acc_matrix, sessions, base_name, metric_suffix=''):
    """
    acc_matrix 是一个列表列表，每一行的长度依次增加，表示
    第 i 次任务训练结束后在前 i 个任务上的测试准确率。
    我们构造一个 n x n 的矩阵，未定义的位置填为 NaN，
    然后用 imshow 绘制热力图，并在每个有效位置上标注数值。
    
    ✨ 新增参数：
    metric_suffix: 指标后缀（如 '_chunk_f1', '_token_micro_f1_no_o'）
    """
    n = len(acc_matrix)
    matrix = np.full((n, n), np.nan)
    for i, row in enumerate(acc_matrix):
        # 每行应填入 row 的所有元素（从左起）
        # ✨ 处理None值
        matrix[i, :len(row)] = [v if v is not None else np.nan for v in row]

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
    # ✨ 生成图片保存路径（包含指标后缀）
    image_name = f"{base_name}_acc_matrix{metric_suffix}.png"
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


def plot_acc_matrix_from_config(config_file_path, train_info_file_path, save_dir="checkpoints/acc_matrix", 
                               plot_all_metrics=True):
    """
    从配置文件和训练信息文件自动绘制acc热力图
    
    该函数会：
    1. 从train_info读取acc_matrix和sessions数据
    2. 根据config_file_path提取文件名作为图片名称
    3. 绘制热力图并保存到指定目录
    4. 打印统计信息和CL指标
    
    ✨ 新增功能：
    - 支持绘制三种指标的热力图（acc, chunk_f1, token_micro_f1_no_o）
    
    Args:
        config_file_path: 配置文件路径（用于提取文件名）
        train_info_file_path: 训练信息JSON文件路径
        save_dir: 保存目录，默认为"checkpoints/acc_matrix"
        plot_all_metrics: 是否绘制所有三种指标的热力图，默认True
    
    Returns:
        output_files: 保存的图片路径列表，如果失败则返回None
    """
    print("\n" + "="*60)
    print("📊 开始绘制训练结果热力图...")
    print("="*60)
    
    try:
        # 检查train_info文件是否存在
        if not os.path.exists(train_info_file_path):
            print(f"⚠️  警告: 训练信息文件不存在: {train_info_file_path}")
            return None
        
        # 读取train_info
        with open(train_info_file_path, 'r', encoding='utf-8') as f:
            train_info = json.load(f)
        
        # ✨ 提取三种矩阵和sessions
        acc_matrix = train_info.get("acc_matrix", [])
        chunk_f1_matrix = train_info.get("chunk_f1_matrix", [])
        token_micro_f1_no_o_matrix = train_info.get("token_micro_f1_no_o_matrix", [])
        sessions = train_info.get("sessions", [])
        
        if not acc_matrix or not sessions:
            print("⚠️  警告: train_info中没有acc_matrix或sessions数据")
            return None
        
        # 从配置文件名提取base_name
        config_name = Path(config_file_path).stem  # 不含路径和扩展名
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        output_files = []
        
        # ✨ 定义要绘制的矩阵列表
        matrices_to_plot = [
            ("acc", acc_matrix, "Default (Acc)"),
        ]
        
        # 如果启用了所有指标的绘制，并且矩阵不为空
        if plot_all_metrics:
            if chunk_f1_matrix:
                matrices_to_plot.append(("chunk_f1", chunk_f1_matrix, "Chunk-level F1"))
            if token_micro_f1_no_o_matrix:
                matrices_to_plot.append(("token_micro_f1_no_o", token_micro_f1_no_o_matrix, "Token Micro F1 (no O)"))
        
        # ✨ 为每种指标绘制热力图
        for metric_key, matrix_data, metric_name in matrices_to_plot:
            print(f"\n📈 绘制 {metric_name} 热力图...")
            
            # 生成文件名
            suffix = "" if metric_key == "acc" else f"_{metric_key}"
            output_file = os.path.join(save_dir, f"{config_name}_acc_matrix{suffix}.png")
            
            # 绘制热力图
            n = len(matrix_data)
            matrix = np.full((n, n), np.nan)
            for i, row in enumerate(matrix_data):
                # ✨ 处理None值
                matrix[i, :len(row)] = [v if v is not None else np.nan for v in row]
            
            plt.figure(figsize=(8, 7))
            im = plt.imshow(matrix, cmap='viridis', interpolation='nearest', vmin=0, vmax=100)
            plt.title(f'{metric_name} Matrix\n{config_name}', fontsize=14, fontweight='bold')
            
            # 设置坐标轴标签
            session_names = [s["session_name"] for s in sessions][:n]
            plt.xticks(ticks=np.arange(n), labels=session_names, rotation=45, ha='right')
            plt.yticks(ticks=np.arange(n), labels=session_names)
            plt.xlabel('Test Task', fontsize=12)
            plt.ylabel('Train Task', fontsize=12)
            
            # 添加颜色条
            cbar = plt.colorbar(im, label=f'{metric_name} (%)')
            
            # 在每个单元格中标注数值
            for i in range(n):
                for j in range(n):
                    if not np.isnan(matrix[i, j]):
                        # 检查是否有replay标记
                        replay_sessions = sessions[i].get("details", {}).get("replay_sessions", [])
                        test_session_name = sessions[j].get("session_name", "")
                        
                        if test_session_name in replay_sessions:
                            text_str = f"{matrix[i, j]:.1f}*"
                        else:
                            text_str = f"{matrix[i, j]:.1f}"
                        
                        # 根据背景色选择文字颜色
                        text_color = "white" if matrix[i, j] < 50 else "black"
                        plt.text(j, i, text_str, ha="center", va="center", 
                                color=text_color, fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            output_files.append(output_file)
            
            # 打印统计信息
            print(f"✅ {metric_name} 热力图已保存: {output_file}")
            print(f"   任务数量: {n}")
            print(f"   平均准确率: {np.nanmean(matrix):.2f}%")
        
        # ✨ 打印所有指标的最终CL指标
        print(f"\n📈 最终持续学习指标汇总:")
        print(f"   配置文件: {config_file_path}")
        
        # 从最后一个session的final_metrics中提取
        if sessions and "final_metrics" in sessions[-1]:
            fm = sessions[-1]["final_metrics"]
            
            # 默认指标
            if "continual_metrics" in fm:
                cm = fm["continual_metrics"]
                print(f"\n  📊 默认指标 (acc):")
                print(f"     AA: {cm.get('AA', 0):.2f}%, FM: {cm.get('FM', 0):.2f}%, BWT: {cm.get('BWT', 0):.2f}%")
            
            # Chunk F1指标
            if "continual_metrics_chunk_f1" in fm:
                cm_chunk = fm["continual_metrics_chunk_f1"]
                print(f"\n  📊 Chunk-level F1:")
                print(f"     AA: {cm_chunk.get('AA', 0):.2f}%, FM: {cm_chunk.get('FM', 0):.2f}%, BWT: {cm_chunk.get('BWT', 0):.2f}%")
            
            # Token Micro F1 (no O)指标
            if "continual_metrics_token_micro_f1_no_o" in fm:
                cm_token = fm["continual_metrics_token_micro_f1_no_o"]
                print(f"\n  📊 Token Micro F1 (no O):")
                print(f"     AA: {cm_token.get('AA', 0):.2f}%, FM: {cm_token.get('FM', 0):.2f}%, BWT: {cm_token.get('BWT', 0):.2f}%")
        
        print("="*60)
        return output_files
        
    except Exception as e:
        print(f"❌ 绘图失败: {e}")
        import traceback
        traceback.print_exc()
        print("="*60)
        return None


def plot_accuracy_matrix_from_train_info(
    train_info_path: str,
    output_path: str = None,
    metric: str = "acc",
    show_values: bool = True,
    cmap: str = "viridis",
    value_range = (0, 100),
):
    """
    Render a continual-learning heatmap directly from a train_info.json file.

    Args:
        train_info_path: Path to the JSON produced after training.
        output_path: Where to save the figure. If None, derive from JSON path.
        metric: One of {'acc', 'chunk_f1', 'token_micro_f1_no_o'} or a custom key
                already present in train_info (e.g. 'acc_matrix').
        show_values: Whether to draw the numeric value inside each cell.
        cmap: Matplotlib colormap.
        value_range: Tuple (vmin, vmax); set to None for automatic scaling.
    """
    metric_alias = {
        "acc": ("acc_matrix", "Accuracy"),
        "chunk_f1": ("chunk_f1_matrix", "Chunk-level F1"),
        "token_micro_f1_no_o": ("token_micro_f1_no_o_matrix", "Token Micro F1 (no O)"),
    }

    metric_key, metric_title = metric_alias.get(metric, (metric, metric))

    if not os.path.exists(train_info_path):
        raise FileNotFoundError(f"train_info file not found: {train_info_path}")

    with open(train_info_path, "r", encoding="utf-8") as f:
        train_info = json.load(f)

    matrix_data = train_info.get(metric_key)
    sessions = train_info.get("sessions", [])

    if not matrix_data:
        raise ValueError(f"Metric '{metric_key}' not found or empty in {train_info_path}")
    if not sessions:
        raise ValueError(f"No 'sessions' information found in {train_info_path}")

    n = len(matrix_data)
    matrix = np.full((n, n), np.nan)

    for i, row in enumerate(matrix_data):
        if row is None:
            continue
        numeric_row = []
        for value in row:
            if value is None:
                numeric_row.append(np.nan)
            else:
                try:
                    numeric_row.append(float(value))
                except (TypeError, ValueError):
                    numeric_row.append(np.nan)
        matrix[i, :len(numeric_row)] = numeric_row

    if value_range is None:
        valid_values = matrix[~np.isnan(matrix)]
        if valid_values.size:
            vmin, vmax = float(np.min(valid_values)), float(np.max(valid_values))
        else:
            vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = value_range

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    fig.colorbar(im, ax=ax, label=f"{metric_title} (%)")

    session_names = []
    for idx in range(n):
        if idx < len(sessions):
            session_names.append(sessions[idx].get("session_name", f"T{idx+1}"))
        else:
            session_names.append(f"T{idx+1}")

    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(session_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(n))
    ax.set_yticklabels(session_names)
    ax.set_xlabel("Test Task")
    ax.set_ylabel("Train Task")
    ax.set_title(f"{metric_title} Matrix", fontsize=14, fontweight="bold")

    if show_values:
        for i in range(n):
            replay_sessions = []
            if i < len(sessions):
                replay_sessions = sessions[i].get("details", {}).get("replay_sessions", [])
            for j in range(n):
                value = matrix[i, j]
                if np.isnan(value):
                    continue
                if j < len(sessions):
                    test_session_name = sessions[j].get("session_name", "")
                else:
                    test_session_name = f"T{j+1}"
                value_str = f"{value:.1f}"
                if test_session_name in replay_sessions:
                    value_str += "*"
                # Choose text color based on normalized intensity for readability
                if vmax - vmin > 0:
                    normalized = (value - vmin) / (vmax - vmin)
                else:
                    normalized = 0.0
                text_color = "white" if normalized < 0.5 else "black"
                ax.text(j, i, value_str, ha="center", va="center",
                        color=text_color, fontsize=10, fontweight="bold")

    fig.tight_layout()

    if output_path is None:
        base = Path(train_info_path)
        default_name = f"{base.stem}_{metric_key}_heatmap.png"
        output_path = str(base.with_name(default_name))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Heatmap saved to {output_path}")
    return output_path


def main():
    os.makedirs("checkpoints", exist_ok=True)

    # 修改下面路径为你实际的 train_info.json 文件路径
    json_file = "checkpoints/251022/train_info_twitter2015_none_t2m_seq1.json"
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
