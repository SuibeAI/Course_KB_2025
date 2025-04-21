from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.siamese_uie, 'iic/nlp_structbert_siamese-uie_chinese-base', model_revision='v1.0')


# 事件抽取 {事件类型（事件触发词）: {参数类型: None}}
output = semantic_cls(
	input='7月28日，天津泰达在德比战中以0-1负于天津天海。', 
  	schema={
        '胜负(事件触发词)': {
            '时间': None,
            '败者': None,
            '胜者': None,
            '赛事名称': None
        }
    }
) 

print(output)

