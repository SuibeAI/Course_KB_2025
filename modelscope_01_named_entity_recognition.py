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
