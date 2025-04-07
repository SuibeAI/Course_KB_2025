from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# 使用在CLUENER数据集上fine-tune的中文RoBERTa模型
model_name = "uer/roberta-base-finetuned-cluener2020-chinese"

# 加载分词器和模型
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# 创建NER管道
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# 输入句子（可自行更换）
text = "小米公司位于北京，雷军是创始人之一。"

# 推理
entities = ner_pipeline(text)

# 打印结果
for entity in entities:
    print(f"实体: {entity['word']}，类型: {entity['entity_group']}，置信度: {entity['score']:.2f}")