import os
from openpyxl import Workbook
from openpyxl.drawing.image import Image

def write_to_excel(input_file, images_folder, output_file):
    # 初始化Excel工作簿
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Data"

    # 设置表头
    headers = ["Text Data", "Entity Name", "Relation Code", "Image"]
    sheet.append(headers)

    # 读取输入数据文件
    with open(input_file, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # 每四行一组写入Excel
    for i in range(0, len(lines), 4):
        if i + 3 >= len(lines):
            print(f"数据行不足四行，跳过第 {i+1} 组。")
            break

        # 提取数据
        # 第一行：替换 \t 为空格，保留其他内容
        text_data = lines[i].rstrip('\n').replace('\t', ' ')  # 第一行
        entity_name = lines[i + 1].strip()  # 第二行
        relation_code = lines[i + 2].strip()  # 第三行
        image_id = lines[i + 3].strip()  # 第四行，无文件后缀

        # 检查图片是否存在并加载
        image_path = os.path.join(images_folder, image_id + ".jpg")
        if os.path.exists(image_path):
            excel_image = Image(image_path)
        else:
            excel_image = None
            print(f"图片 {image_path} 未找到，跳过插入。")

        # 写入表格
        row = [text_data, entity_name, relation_code]
        sheet.append(row)

        # 插入图片到对应行
        if excel_image:
            img_row = sheet.max_row
            img_col = len(row) + 1  # 图片插入到第4列
            sheet.add_image(excel_image, f"D{img_row}")

    # 保存Excel文件
    workbook.save(output_file)
    print(f"数据已保存到 {output_file}")


# 输入和输出路径
input_file = "dataset\\twitter2015\dev_transform_ner.txt"  # 输入文件路径
images_folder = "dataset\\twitter2015\images"  # 图片文件夹路径
output_file = "dataset\\twitter2015\output.xlsx"  # 输出Excel文件路径

# 调用函数
write_to_excel(input_file, images_folder, output_file)
