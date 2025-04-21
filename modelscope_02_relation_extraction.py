from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.siamese_uie, 'iic/nlp_structbert_siamese-uie_chinese-base', model_revision='v1.0')


# 关系抽取 {主语实体类型: {关系(宾语实体类型): None}}
output = semantic_cls(
	input='《七里香》是周杰伦的第五张音乐专辑，由周杰伦担任制作人，共收录《七里香》《将军》《止战之殇》等10首歌曲', 
  	schema = {
        '歌手': {
            '歌曲': None,
            '专辑': None,
        }
    }
) 

print(output)