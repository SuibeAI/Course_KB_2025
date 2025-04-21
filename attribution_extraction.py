from transformers import BertTokenizer, BertForMaskedLM
import torch

# 使用英文模型，你也可以换成中文的 bert-base-chinese
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)
model.eval()

# 构造输入句子（实体+上下文属性）
sentence = "Albert Einstein was a physicist born in Germany. His nationality is [MASK]."
inputs = tokenizer(sentence, return_tensors="pt")

# 获取 [MASK] 的位置
mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

# 模型预测
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 获取预测结果
mask_token_logits = logits[0, mask_token_index, :]
top_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

print("Top 5 predicted nationalities:")
for token in top_tokens:
    print(f"> {tokenizer.decode([token])}")