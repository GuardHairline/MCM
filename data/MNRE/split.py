import os
import ast

def separate_dataset(file_path, path1, path2, out_path_2015, out_path_2017, out_path_other):
    """
    根据每条数据中的图片名称，将数据分离到 twitter2015、twitter2017 和其他。
    
    参数：
        file_path: 数据集文本文件路径，每行一个记录
        path1: twitter2015图片所在文件夹路径
        path2: twitter2017图片所在文件夹路径
        out_path_2015: 输出的twitter2015数据集文件
        out_path_2017: 输出的twitter2017数据集文件
        out_path_other: 输出的其他数据集文件
    """
    with open(file_path, 'r', encoding='utf-8') as f_in, \
         open(out_path_2015, 'w', encoding='utf-8') as f_2015, \
         open(out_path_2017, 'w', encoding='utf-8') as f_2017, \
         open(out_path_other, 'w', encoding='utf-8') as f_other:
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                # 使用 ast.literal_eval 处理数据（数据中的引号为单引号，不是标准 JSON 格式）
                record = ast.literal_eval(line)
            except Exception as e:
                print(f"解析出错：{line}\n错误信息：{e}")
                continue
            
            # 确保 img_id 为字符串类型
            img_id = str(record.get('img_id', ''))
            # 拼接图片在两个文件夹中的完整路径
            img_path_2015 = os.path.join(path1, img_id)
            img_path_2017 = os.path.join(path2, img_id)
            
            if os.path.exists(img_path_2015):
                f_2015.write(line + "\n")
            elif os.path.exists(img_path_2017):
                f_2017.write(line + "\n")
            else:
                f_other.write(line + "\n")

if __name__ == '__main__':
    # 修改以下路径为你的实际路径
    file_path = 'D:/Codes/MCL/MCM/data/MNRE/mnre_txt_org/txt/ours_val.txt'
    path1 = 'D:/Codes/MCL/MCM/data/MASC/twitter2015/images'
    path2 = 'D:/Codes/MCL/MCM/data/MASC/twitter2017/images'
    
    # 输出文件路径
    out_path_2015 = 'D:/Codes/MCL/MCM/data/MNRE/mnre_txt_org/twitter2015/dev.txt'
    out_path_2017 = 'D:/Codes/MCL/MCM/data/MNRE/mnre_txt_org/twitter2017/dev.txt'
    out_path_other = 'D:/Codes/MCL/MCM/data/MNRE/mnre_txt_org/other/dev.txt'
    
    separate_dataset(file_path, path1, path2, out_path_2015, out_path_2017, out_path_other)
    print("数据分离完成！")
