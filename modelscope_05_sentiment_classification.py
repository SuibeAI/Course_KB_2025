from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.siamese_uie, 'iic/nlp_structbert_siamese-uie_chinese-base', model_revision='v1.0')


# 支持情感分类
output = semantic_cls(
	input='很满意，音质很好，发货速度快，值得购买', 
  	schema={
        '属性词': {
            "正向情感(情感词)": None, 
            "负向情感(情感词)": None, 
            "中性情感(情感词)": None
        }
    }
) 

print(output)

