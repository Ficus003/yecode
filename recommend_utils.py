import surprise
from surprise import Dataset,KNNBasic

#训练协同过滤模型(user_based CF)
#返回：训练好的模型
def train_collaborative_filtering(ratings_df):
    """
    训练基于用户协同过滤推荐模型
    使用surprise推荐系统库
    :param ratings_df: DataFrame - 用户对菜品的评分
    :return: 训练好的模型
    """
    reader = surprise.Reader(rating_scale=(1,5))  #创建reader对象，指定评分范围1-5
    data = Dataset.load_from_df(ratings_df,reader) #surprise库dataset工具将pandas DataFrame转换为surprise格式
    trainset = data.build_full_trainset()  #不划分测试集，数据全部用于训练

    sim_options = {   #相似度配置
        'name':'cosine',   #使用余弦相似度计算相似性
        'user_based':True  #启用基于用户协同过滤，查找兴趣相似用户，推荐他们喜欢的菜(False:基于物品的协同过滤，查找相似菜品)
    }

    model= KNNBasic(sim_options = sim_options)  #KNNBasic surprise库实现协同过滤的模型，内部使用相似矩阵
    model.fit(trainset)   #把数据交给模型进行训练

    return model   #返回训练好的模型

#使用CF（训练好的KNNBasic）模型，推荐评分最高的前n(n=10)个菜
def collaborative_filtering(user_id,model,graph,ratings_data,n=10):

    all_dishes = set(get_all_dishes(graph))   #获取知识图谱中所有菜品'name'放入集合
    rated_dishes = set(get_rated_dishes(user_id, ratings_data))   #调用get_rated_dishes函数，找出用户(user_id)已经评分过的菜品，放入集合
    unrated_dishes = all_dishes - rated_dishes  #差集，得出未评分菜品集合

    predictions = []   #创建预测菜品列表
    for dish_id in unrated_dishes:   #遍历未评分菜品
        pred = model.predict(user_id,dish_id)  #预测用户对该菜品评分
        predictions.append((dish_id,pred.est)) #(菜品id，预测评分)放入predictions列表

    return sorted(predictions,key=lambda x: -x[1])[:n]   #对评分排序，从高到低，取前n项

#从知识图谱获取所有菜品id
def get_all_dishes(graph):

    query = "MATCH (d:Dish) RETURN d.name as dish_id"
    results = graph.run(query).data()
    return [r['dish_id'] for r in results]

#获取用户已评分菜品列表
def get_rated_dishes(user_id,ratings_data):
    from collections import defaultdict  #defaultdict存储用户评分记录
    user_ratings = defaultdict(list)     #字典

    for record in ratings_data:    #遍历ratings_data
        user_ratings[record['user_id']].append(record['dish_id']) #按照user_id分类，放入用户评分过的dish_id，放在user_ratings

    return user_ratings.get(user_id, [])  #从user_ratings取出对应的已经评分了的菜品列表，如果没有，就返回一个空列表[]

#通过调用知识图谱推荐符合用户偏好的菜品
def kg_recommend(user_profile,graph,n=10):
    query = """     
    MATCH (d:Dish)
    WHERE d.calories <= $max_cal
    AND ANY(flavor IN SPLIT(d.flavor, ',') WHERE toLower(flavor) IN $preferred_flavors)
    AND NOT EXISTS{
        MATCH (d)-[:包含]->(i:Ingredient)
        WHERE i.name IN $restrictions
        }
    RETURN d.name as dish_id,
            (d.spicy * $spicy_weight +
             (1000 - d.calories)/100 * $calories_weight)as kg_score
    ORDER BY kg_score DESC
    LIMIT $n
    """

    preferred_flavors = [f.lower() for f in user_profile['preferred_flavors']]  #大小写处理
    restrictions = [r.lower() for r in user_profile.get('restrictions', [])]

    params = {     #参数传入query
        'max_cal':user_profile['max_calories'],
        'preferred_flavors': preferred_flavors,
        'restrictions':restrictions,
        'spicy_weight':0.6,
        'calories_weight':0.4,
        'n':n
    }

    #调试输出
    print("KG查询参数：", params)

    try:
        results = graph.run(query,params).data()
        print("KG查询结果：",results)
        return [(r['dish_id'], r['kg_score']) for r in results]  #返回列表[('菜品1'，评分),('菜品2',评分)]
    except Exception as e:
        print("KG查询错误：", str(e))
        return []

#将CF评分与KG评分通过加权平均方法，合成最终推荐列表
def combine_results(cf_items, kg_items, alpha=0.7):  #alpha:权重系数

    #处理空列表
    if not cf_items and not kg_items:
        return []

    #归一化数值范围
    cf_scores = normalize([s for _, s in cf_items]) if cf_items else []
    kg_scores = normalize([s for _, s in kg_items]) if kg_items else []

    combined = {}

    #只有cf时
    if cf_items and not kg_items:
        return cf_items

    #只有kg时
    if kg_items and not cf_items:
        return kg_items

    #整合cf和kg的结果
    for (dish_id, _), cf_norm in zip(cf_items, cf_scores):  #遍历CF结果
        combined[dish_id] = alpha * cf_norm

    for (dish_id, _), kg_norm in zip(kg_items, kg_scores):  #遍历KG结果
        combined[dish_id] = combined.get(dish_id, 0) + (1 - alpha) * kg_norm  #知识图谱默认权重：(1-alpha)

    return sorted(combined.items(), key=lambda x: -x[1])  #从高到低返回数据

#归一化函数，调用后使CF评分与KG评分缩放在一个范围内
def normalize(scores):

    #处理空列表
    if not scores:
        return []

    min_s, max_s = min(scores), max(scores)
    #防止除0
    if max_s == min_s:
        return [0.5 for _ in scores]     #返回中性值

    return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]  #(当前分数-最小分数)/(最大值-最小值),+1e-6:防止计算机浮点误差

def get_user_profile(user_id,graph):
    try:
        flavor_query = """
        MATCH (u:User {id:$user_id})-[r:RATED]->(d:Dish)
        RETURN d.flavor as flavors
        """
        flavors = list({f for r in graph.run(flavor_query,user_id=user_id).data()
                        for f in r['flavor'].split(',')})
    except:
        flavors = ['咸鲜']

    return {
        'preferred_flavors': flavors or ['咸鲜'],
        'restrictions': get_diet_restrictions(user_id),
        'max_calories': 600
    }

def get_preferred_flavors(user_id):
    try:
        user_num = int(user_id.split('user')[-1])
        return ['辣','咸鲜']if user_id % 2 == 0 else ['清淡']
    except:
        return ['咸鲜']

def get_diet_restrictions(user_id):
    try:
        user_num = int(user_id.split('user')[-1])
        return ['花生'] if user_num % 3 == 0 else []
    except:
        return []


def hybrid_recommend(user_id,graph,cf_model,ratings_data,alpha=0.6):

    profile = get_user_profile(user_id,graph)

    cf_items = collaborative_filtering(user_id,cf_model,graph,ratings_data)

    kg_items = kg_recommend(profile,graph)

    return combine_results(cf_items,kg_items,alpha)