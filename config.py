#数值型参数定义
PREDEFINED_PARAMS = {
    "spicy": {"不辣": 0, "微辣": 1, "中辣": 2, "重辣": 3, "爆辣": 4},
    "calories": {"低卡": 500, "中卡": 800, "高卡": 1000},
    "salty": {"清淡": 1, "适中": 2, "偏咸": 3}
}

#模糊参数定义
KG_PARAM_TYPES = ["cuisine", "ingredient", "flavor"]

# BERT模型意图标签
INTENT_LABELS = [
    "recommend",
    "search_ingredient",
    "search_flavor",
    "diet_restriction",
    "exit",
    "unknown"
]
