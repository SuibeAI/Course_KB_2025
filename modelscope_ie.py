from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

semantic_cls = pipeline(Tasks.siamese_uie, 'iic/nlp_structbert_siamese-uie_chinese-base', model_revision='v1.0')

# 命名实体识别 {实体类型: None}
output = semantic_cls(
    input='小米科技有限责任公司（Xiaomi Corporation）成立于2010年3月3日，总部位于北京市海淀区安宁庄路小米科技园，创始人雷军，是一家主要从事智能手机、物联网和生活消费产品研发和销售业务，提供互联网服务，以及从事投资业务的中国投资控股公司。', 
    schema={
        '人物': None,
        '地理位置': None,
        '组织机构': None
    }
) 

print(output)

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

# 属性情感抽取 {属性词: {情感词: None}}
output = semantic_cls(
	input='很满意，音质很好，发货速度快，值得购买', 
  	schema={
        '属性词': {
            '情感词': None,
        }
    }
) 

print(output)

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
