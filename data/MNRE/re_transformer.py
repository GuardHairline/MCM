def process_relation_data_from_txt(input_file, output_file):
    # 定义关系映射
    relation_map = {
        '/per/per/peer': -1,
        '/per/org/member_of': 0,
        '/loc/loc/contain': 1,
        '/per/misc/present in': 2,
        '/org/loc/locate_at': 3,
        '/per/loc/place of residence': 4,
        '/misc/loc/held on': 5,
        '/org/org/subsidiary': 6,
        '/per/per/alternate names': 7,
        '/per/per/couple': 8,
        '/per/misc/nationality': 9,
        '/misc/misc/part of': 10,
        '/org/org/alternate names': 11,
        '/per/loc/place of birth': 12,
        '/per/misc/awarded': 13,
        '/per/per/charges': 14,
        '/per/per/parent': 15,
        '/per/per/siblings': 16,
        '/per/per/neighbor': 17,
        '/per/per/alumni': 18,
        '/per/misc/religion': 19,
        '/per/misc/race': 20,
        None: 21
    }

    output_lines = []

    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 遍历每行数据
    for line in lines:
        try:
            # 将每行数据转化为字典格式
            entry = eval(line.strip())

            tokens = entry['token']
            head = entry['h']
            tail = entry['t']
            img_id = entry['img_id']
            relation = entry['relation']

            # 确保所有 tokens 都是字符串类型
            tokens = [str(token) for token in tokens]
            
            # 获取关系编码
            relation_code = relation_map.get(relation, 21)

            # 生成关系头的格式
            head_name = " ".join(tokens[head['pos'][0]:head['pos'][1]])
            tail_name = " ".join(tokens[tail['pos'][0]:tail['pos'][1]])

            # 处理关系头的掩码文本
            head_masked = tokens[:head['pos'][0]] + ["$T$"] + tokens[head['pos'][1]:]
            head_masked_text = "\t".join(head_masked)
            
            # 处理关系尾的掩码文本
            tail_masked = tokens[:tail['pos'][0]] + ["$T$"] + tokens[tail['pos'][1]:]
            tail_masked_text = "\t".join(tail_masked)

            # 将数据添加到输出列表
            output_lines.append(head_masked_text)
            output_lines.append(head_name)
            output_lines.append(str(relation_code))
            output_lines.append(str(img_id))#有一个数据是int格式，强制转化为str
            
            output_lines.append(tail_masked_text)
            output_lines.append(tail_name)
            output_lines.append(str(relation_code))
            output_lines.append(str(img_id))
        except Exception as e:
            print(f"Error processing line: {line.strip()} - {e}")
    try:
        # 保存到输出文件
        with open(output_file, "w", encoding="utf-8") as file:
            file.write("\n".join(output_lines))
        print(f"转换完成，结果已保存到 {output_file}")
    except Exception as e:
        print(f"Error writing to file: {e}")


# 输入和输出文件路径
input_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2015\ori_dev.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2015\dev.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)

# 输入和输出文件路径
input_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2015\ori_test.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2015\\test.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)# 输入和输出文件路径

input_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2015\ori_train.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2015\\train.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)# 输入和输出文件路径

# 输入和输出文件路径
input_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2017\ori_dev.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2017\dev.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)

# 输入和输出文件路径
input_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2017\ori_test.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2017\\test.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)# 输入和输出文件路径

input_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2017\ori_train.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\twitter2017\\train.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)# 输入和输出文件路径

input_file = 'D:\Codes\MCL\MCM\data\MNRE\\mix\ori_dev.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\mix\dev.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)# 输入和输出文件路径

# 输入和输出文件路径
input_file = 'D:\Codes\MCL\MCM\data\MNRE\\mix\ori_test.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\mix\\test.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)# 输入和输出文件路径

# 输入和输出文件路径
input_file = 'D:\Codes\MCL\MCM\data\MNRE\\mix\ori_train.txt'  # 输入文件路径（txt文本）
output_file = 'D:\Codes\MCL\MCM\data\MNRE\\mix\\train.txt'  # 输出文件路径
# 调用函数进行数据转换
process_relation_data_from_txt(input_file, output_file)