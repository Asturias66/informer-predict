import torch
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# 加载预训练的BERT模型和分词器
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True, ignore_mismatched_sizes=True)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 输入句子
sentence = "The quick brown fox jumps over the lazy dog."

# 使用分词器对句子进行编码
inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

# 计算模型的输出（注意力权重）
with torch.no_grad():
    outputs = model(**inputs)
    attentions = outputs.attentions

# 获取第一个注意力头的注意力权重
attention_head_0 = attentions[0][0, 0].cpu().numpy()

# 创建一个热图以可视化注意力权重
plt.figure(figsize=(10, 5))
sns.heatmap(attention_head_0, annot=True, fmt='.2f', cmap='viridis',
            xticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]),
            yticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))

# 设置轴标签和标题
plt.xlabel('Keys')
plt.ylabel('Queries')
plt.title('Self-Attention Weights in BERT (Head 0)')

# 显示图形
plt.show()