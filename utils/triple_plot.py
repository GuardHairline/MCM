import os
import json
import numpy as np
import matplotlib.pyplot as plt


def plot_metrics_for_three_files(file1, file2, file3, base_name1, base_name2, base_name3, metric_name, metric_label):
    """
    根据给定的指标名称（如 AA、AIA、FM、BWT），从三个文件中提取并比较该指标。
    :param file1, file2, file3: 输入的三个 JSON 文件
    :param base_name1, base_name2, base_name3: 对应文件的基本名称，用于图例标签
    :param metric_name: 要绘制的指标（如 AA、AIA、FM、BWT）
    :param metric_label: 对应指标的标签
    """

    def extract_metric(sessions, metric_name):
        names = []
        metric_values = []
        for session in sessions:
            final_metrics = session.get("final_metrics", {})
            if final_metrics:
                names.append(session["session_name"])
                metric_values.append(final_metrics.get(metric_name, np.nan))
        return names, metric_values

    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2, open(file3, "r",
                                                                                                  encoding="utf-8") as f3:
        data1 = json.load(f1)
        data2 = json.load(f2)
        data3 = json.load(f3)

    sessions1 = data1.get("sessions", [])
    sessions2 = data2.get("sessions", [])
    sessions3 = data3.get("sessions", [])

    names1, values1 = extract_metric(sessions1, metric_name)
    names2, values2 = extract_metric(sessions2, metric_name)
    names3, values3 = extract_metric(sessions3, metric_name)

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(values1, marker='o', linestyle='-', label=f'{base_name1} ({metric_label})', color='b')
    plt.plot(values2, marker='s', linestyle='--', label=f'{base_name2} ({metric_label})', color='g')
    plt.plot(values3, marker='^', linestyle='-.', label=f'{base_name3} ({metric_label})', color='r')

    plt.title(f'{metric_label} Comparison')
    plt.xlabel('Session Index')
    plt.ylabel(f'{metric_label} Value')
    plt.xticks(ticks=np.arange(len(names1)), labels=names1, rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    os.makedirs("checkpoints/figures", exist_ok=True)
    image_path = os.path.join("checkpoints/figures",
                              f"compare_{metric_name}_{base_name1}_vs_{base_name2}_vs_{base_name3}.png")
    plt.savefig(image_path)
    plt.show()


def compare_results_for_three_files(file1, file2, file3):
    """
    加载三个 JSON 文件的结果，比较它们的指标（AA, AIA, FM, BWT），并分别生成对比图。
    """
    base_name1 = os.path.splitext(os.path.basename(file1))[0]
    base_name2 = os.path.splitext(os.path.basename(file2))[0]
    base_name3 = os.path.splitext(os.path.basename(file3))[0]

    metrics = [
        ('AA', 'AA'),
        ('AIA', 'AIA'),
        ('FM', 'FM'),
        ('BWT', 'BWT')
    ]

    for metric_name, metric_label in metrics:
        plot_metrics_for_three_files(file1, file2, file3, base_name1, base_name2, base_name3, metric_name, metric_label)


def main():
    json_file1 = "checkpoints/train_info_twitterMix_none_text.json"
    json_file2 = "checkpoints/train_info_twitterMix_replay_text.json"
    json_file3 = "checkpoints/train_info_twitterMix_replay_2nd_text.json"

    compare_results_for_three_files(json_file1, json_file2, json_file3)


if __name__ == "__main__":
    main()
