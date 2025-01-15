# scripts/inference.py
import torch
from PIL import Image
from models.base_model import BaseMultimodalModel
from models.task_heads.mabsa_head import MABSAHead
from train import Full_Model
from transformers import AutoTokenizer
from torchvision import transforms

def predict_sentiment(model_path, text, entity, image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载模型
    base_model = BaseMultimodalModel()
    head = MABSAHead(input_dim=base_model.fusion_output_dim, num_labels=3)
    full_model = Full_Model(base_model, head)
    full_model.load_state_dict(torch.load(model_path, map_location=device))
    full_model.to(device)
    full_model.eval()

    # 文本处理
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    input_text = f"{text} [SEP] {entity}"
    encoding = tokenizer(input_text,
                         truncation=True,
                         padding='max_length',
                         max_length=128,
                         return_tensors='pt')

    # 图像处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])
    try:
        with Image.open(image_path).convert('RGB') as img:
            img_tensor = transform(img).unsqueeze(0)
    except:
        img_tensor = torch.zeros((1, 3, 224, 224))

    # 推理
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids", None)
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        logits = full_model(input_ids, attention_mask, token_type_ids, img_tensor.to(device))
        preds = torch.argmax(logits, dim=-1).item()

    # 映射回-1,0,1 (假设 index: 0->neg, 1->neu, 2->pos)
    index_to_label = {0: -1, 1: 0, 2: 1}
    sentiment = index_to_label[preds]
    return sentiment

if __name__ == "__main__":
    model_path = "checkpoints/mabsa_model.pt"
    text = "这家餐厅的服务真的很棒，"
    entity = "服务"
    image_path = "data/mabsa/images/example.jpg"

    sentiment = predict_sentiment(model_path, text, entity, image_path)
    print("Predicted sentiment:", sentiment)
