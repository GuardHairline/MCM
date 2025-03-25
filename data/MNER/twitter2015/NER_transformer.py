def process_data_file(input_file, output_file):
    # 读取文件内容
    with open(input_file, "r", encoding="utf-8") as file:
        input_lines = file.readlines()
    
    # 处理数据
    output_lines = process_data(input_lines)
    
    # 保存到输出文件
    with open(output_file, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines))
    print(f"转换完成，结果已保存到 {output_file}")

def process_data(input_lines):
    result = []
    img_id = None
    current_text = []
    current_tags = []

    # 实体类别映射
    entity_map = {"PER": "-1", "ORG": "0", "LOC": "1", "OTHER": "2"}

    for line in input_lines:
        if line.startswith("IMGID:"):
            # 如果有前面的数据，处理前一组
            if img_id is not None:
                result.extend(generate_output(current_text, current_tags, img_id, entity_map))
            # 新图片 ID
            img_id = line.strip().split(":")[1]
            current_text = []
            current_tags = []
        else:
            # 非空行则解析单词及其标签
            parts = line.strip().split("\t")
            if len(parts) == 2:
                word, tag = parts
                current_text.append(word)
                current_tags.append(tag)

    # 处理最后一组
    if img_id is not None:
        result.extend(generate_output(current_text, current_tags, img_id, entity_map))
    
    return result

def generate_output(words, tags, img_id, entity_map):
    output = []
    entities = extract_entities(words, tags)
    
    for entity_text, entity_type, start, end in entities:
        masked_text = words[:start] + ["$T$"] + words[end:]
        masked_text = "\t".join(masked_text)
        
        output.append(f"{masked_text}")
        output.append(f"{entity_text}")
        output.append(f"{entity_map[entity_type]}")
        output.append(f"{img_id}")
    
    return output

def extract_entities(words, tags):
    entities = []
    entity_text = []
    entity_type = None
    start_idx = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-"):
            # 保存上一个实体
            if entity_text:
                entities.append((" ".join(entity_text), entity_type, start_idx, i))
            # 开始新的实体
            entity_text = [words[i]]
            entity_type = tag[2:]
            start_idx = i
        elif tag.startswith("I-") and entity_type == tag[2:]:
            # 继续当前实体
            entity_text.append(words[i])
        else:
            # 非实体或新实体，保存前一个实体
            if entity_text:
                entities.append((" ".join(entity_text), entity_type, start_idx, i))
            entity_text = []
            entity_type = None
            start_idx = None

    # 处理最后一个实体
    if entity_text:
        entities.append((" ".join(entity_text), entity_type, start_idx, len(tags)))

    return entities

# 调用函数读取文件并保存结果
input_file = "dataset\\twitter2015\\train_100_samples.txt"
output_file = "dataset\\twitter2015\\train_100_samples_transform_ner.txt"
process_data_file(input_file, output_file)
