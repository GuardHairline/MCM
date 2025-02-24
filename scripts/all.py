import subprocess
import os


def main():
    # 获取脚本所在目录的上一级目录作为项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)


    # 假设我们有不同的参数设置列表
    param_sets = [
        # 第一个参数集
        ["--task_name", "masc", "--num_labels", "3", "--strategy", "none", "--session_name", "masc_1", "--mode", "text_only"],
        # 第二个参数集
        ["--task_name", "mate", "--session_name", "mate_1", "--pretrained_model_path", "checkpoints/model_1.pt", "--output_model_path", "checkpoints/model_mate.pt", "--num_labels", "3", "--strategy", "none", "--epochs", "20", "--lr", "1e-5", "--mode", "text_only"],
        # 可以添加更多的参数集
        ["--task_name", "mner", "--session_name", "mner_1", "--pretrained_model_path", "checkpoints/model_mate.pt", "--output_model_path", "checkpoints/model_mner.pt", "--train_text_file", "data/MNER/twitter2015/train.txt", "--test_text_file", "data/MNER/twitter2015/test.txt", "dev_text_file", "data/MNER/twitter2015/dev.txt", "image_dir", "data/MNER/twitter2015/images", "num_labels", "9", "strategy", "none", "mode", "text_only"],

        ["--task_name", "mnre", "session_name", "mnre_1", "pretrained_model_path", "checkpoints/model_mner.pt", "output_model_path", "checkpoints/model_mnre.pt", "train_text_file", "data/MNRE/mix/mnre_txt/train.txt", "test_text_file", "data/MNRE/mix/mnre_txt/test.txt", "dev_text_file", "data/MNRE/mix/mnre_txt/dev.txt", "image_dir", "data/MNRE/mix/mnre_image", "num_labels", "23", "strategy", "none"],

        ["--task_name", "mabsa", "num_labels", "7", "strategy", "none", "pretrained_model_path", "checkpoints/model_mnre.pt", "output_model_path", "checkpoints/model_mabsa.pt", "session_name", "mabsa_1", "mode", "text_only"]
    ]


    for params in param_sets:
        # 构造完整的命令
        command = ["python", os.path.join("scripts", "train.py")] + params


        # 切换工作目录到项目目录
        original_dir = os.getcwd()
        os.chdir(project_dir)


        try:
            # 运行命令
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command {' '.join(command)}: {e}")


        # 切换回原来的工作目录
        os.chdir(original_dir)


if __name__ == "__main__":
    main()