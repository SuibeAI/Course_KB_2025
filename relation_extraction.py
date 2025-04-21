from transformers import BertTokenizer, BartForConditionalGeneration

# 模型名称
model_name = 'fanxiao/CGRE_CNDBPedia-Generative-Relation-Extraction'

# 加载分词器和模型
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# 输入文本
text = "小米公司位于北京，雷军是创始人之一。"

# 编码输入文本
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

# 生成输出
outputs = model.generate(
    inputs["input_ids"],
    max_length=128,
    num_beams=4,
    early_stopping=True
)

# 解码输出
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("抽取结果：", result)