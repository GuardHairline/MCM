import subprocess
import os


def main():
    # 获取脚本所在目录的上一级目录作为项目目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)


    # 假设我们有不同的参数设置列表
    param_sets = [
        # 第一个参数集
        ["--task_name", "task1", "--session_name", "session1", "--epochs", "10", "--lr", "1e-4"],
        # 第二个参数集
        ["--task_name", "task2", "--session_name", "session2", "--epochs", "20", "--lr", "1e-5"],
        # 可以添加更多的参数集
        ["--task_name", "task3", "--session_name", "session3", "--epochs", "15", "--lr", "5e-5"]
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