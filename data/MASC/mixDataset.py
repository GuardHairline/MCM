#!/usr/bin/env python3
import os
import shutil

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def concat_files(split):
    # 定义各数据集文件路径
    file2015 = os.path.join('twitter2015', f"{split}.txt")
    file2017 = os.path.join('twitter2017', f"{split}.txt")
    output_file = os.path.join('mix', f"{split}.txt")

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_path in [file2015, file2017]:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    lines = infile.readlines()
                    # 如果最后一行没有换行符，则加上换行符，避免拼接时行内容混在一起
                    if lines and not lines[-1].endswith('\n'):
                        lines[-1] = lines[-1] + '\n'
                    outfile.writelines(lines)
            else:
                print(f"警告：找不到 {file_path} 文件。")

    # 检查拼接后的文件行数是否为4的倍数
    with open(output_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        if len(lines) % 4 != 0:
            print(f"警告：{split}.txt 的行数 {len(lines)} 不是4的倍数，请检查数据格式！")
        else:
            print(f"{split}.txt 合并成功，总行数：{len(lines)}。")

def copy_images():
    # 定义源图片文件夹
    sources = [os.path.join('twitter2015', 'images'), os.path.join('twitter2017', 'images')]
    dest_dir = os.path.join('mix', 'images')
    for source in sources:
        if os.path.exists(source):
            for item in os.listdir(source):
                s = os.path.join(source, item)
                d = os.path.join(dest_dir, item)
                # 如果是文件则直接复制；如果是文件夹，可以根据需要使用 copytree
                if os.path.isfile(s):
                    try:
                        shutil.copy(s, d)
                    except Exception as e:
                        print(f"复制文件 {s} 时发生错误：{e}")
                elif os.path.isdir(s):
                    # 如果目标文件夹已存在，可以选择跳过或者重命名
                    try:
                        shutil.copytree(s, d)
                    except FileExistsError:
                        print(f"目录 {d} 已存在，跳过复制 {s}。")
                    except Exception as e:
                        print(f"复制目录 {s} 时发生错误：{e}")
        else:
            print(f"警告：找不到图片文件夹 {source}。")

def main():
    # 确保输出目录存在
    ensure_dir('mix')
    ensure_dir(os.path.join('mix', 'images'))

    # 拼接 train, dev, test 文件
    for split in ['train', 'dev', 'test']:
        concat_files(split)

    # 复制图片
    copy_images()

if __name__ == '__main__':
    main()
