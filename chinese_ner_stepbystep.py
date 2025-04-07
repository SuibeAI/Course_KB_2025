from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import torch.nn.functional as F

# 1. 加载模型与分词器
model_name = "uer/roberta-base-finetuned-cluener2020-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)
model.eval()

# 2. 输入文本
text = "小米公司位于北京，雷军是创始人之一。"

# 3. 分词与编码
inputs = tokenizer(text, return_tensors="pt")
print("\n🔹 分词结果 tokenizer.tokenize():")
print(tokenizer.tokenize(text))

print("\n🔹 编码结果 tokenizer():")
print(inputs)

# 4. 模型前向传播
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

print("\n🔹 模型输出 logits:")
print(logits.shape)

# 5. 取最大概率的类别
predictions = torch.argmax(logits, dim=-1)
print("\n🔹 预测的标签ID:")
print(predictions)

# 6. 映射标签 ID 到实际标签（BIO）
id2label = model.config.id2label
print("\n🔹 标签映射表:")
print(id2label)

labels = [id2label[label_id.item()] for label_id in predictions[0]]
print("\n🔹 标签映射结果（BIO）:")
for token, label in zip(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]), labels):
    print(f"{token}: {label}")

# 7. 实体提取：合并 B- 和 I- 标签
def extract_entities(tokens, labels):
    entities = []
    current_entity = None
    current_type = None

    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = token
            current_type = label[2:]
        elif label.startswith("I-") and current_entity:
            current_entity += token
        else:
            if current_entity:
                entities.append((current_entity, current_type))
                current_entity = None
                current_type = None
    if current_entity:
        entities.append((current_entity, current_type))
    return entities

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
entities = extract_entities(tokens, labels)

print("\n🔹 实体识别结果:")
for word, entity_type in entities:
    print(f"实体: {word}，类型: {entity_type}")