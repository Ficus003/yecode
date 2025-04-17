import pandas as pd
import surprise
from surprise import Dataset,KNNBasic
from neo4j_utils import KGQuery
from predictor import IntentPredictor
#训练协同过滤模型(user_based CF)
#返回：训练好的模型
def train_collaborative_filtering(user_profile):
    """
    训练基于用户协同过滤推荐模型
    使用surprise推荐系统库

    """
    ratings_df = pd.DataFrame(user_profile)
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
def collaborative_filtering(user_id,model,graph,user_profile,n=10):

    all_dishes = set(get_all_dishes(graph))   #获取知识图谱中所有菜品'name'放入集合
    rated_dishes = set(get_rated_dishes(user_id, user_profile))   #调用get_rated_dishes函数，找出用户(user_id)已经评分过的菜品，放入集合
    unrated_dishes = all_dishes - rated_dishes  #差集，得出未评分菜品集合

    predictions = []   #创建预测菜品列表
    for dish_id in unrated_dishes:   #遍历未评分菜品
        pred = model.predict(user_id, dish_id)  #预测用户对该菜品评分
        predictions.append((dish_id, pred.est)) #(菜品id，预测评分)放入predictions列表

    return sorted(predictions, key=lambda x: -x[1])[:n]   #对评分排序，从高到低，取前n项

#从知识图谱获取所有菜品id
def get_all_dishes(graph):

    query = "MATCH (d:Dish) RETURN d.name as dish_id"
    results = graph.run(query).data()
    return [r['dish_id'] for r in results]

#获取用户已评分菜品列表
def get_rated_dishes(user_id,user_profile):
    from collections import defaultdict  #defaultdict存储用户评分记录
    user_rated = defaultdict(list)     #字典

    for record in user_profile:    #遍历user_profile
        user_rated[record['user_id']].append(record['dish_id']) #按照user_id分类，放入用户评分过的dish_id，放在user_ratings

    return user_rated.get(user_id, [])  #从user_ratings取出对应的已经评分了的菜品列表，如果没有，就返回一个空列表[]

#通过调用知识图谱推荐符合用户偏好的菜品

#将CF评分与KG评分通过加权平均方法，合成最终推荐列表,kg_score为默认的1
def combine_results(cf_items, kg_items, cf_weight, n=10):  #alpha:权重系数

    #处理空列表
    if not cf_items and not kg_items:
        return []

    #归一化数值范围
    cf_scores = normalize([s for _, s in cf_items]) if cf_items else []
    kg_scores = 1 #默认值
    combined = {}

    #只有cf时
    if cf_items and not kg_items:
        return cf_items

    #只有kg时
    if kg_items and not cf_items:
        return kg_items

    #整合cf和kg的结果
    combined = [
        (dish, (1 - cf_weight) * kg_scores + cf_weight * cf_scores) for dish, score in cf_scores
    ] #知识图谱默认权重：(1-alpha)

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:n]

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

        rating_query = """
        MATCH (u:User)-[r:rated]->(d:Dish)
        WHERE u.user_id = $user_id
        RETURN u.user_id as user_id, d.name as dish_id, r.rating as rating
        """
        with graph.session() as session:
            result = graph.run(rating_query, user_id=user_id)
            user_profile = []
            for record in result:
                user_profile.append(
                    {
                        "user_id": record["user_id"],
                        "dish_id": record["dish_id"],
                        "rating": record["rating"]
                    }
                )
            return user_profile





def hybrid_recommend(user_id,graph,cf_model):

    profile = get_user_profile(user_id,graph)

    cf_items = collaborative_filtering(user_id,cf_model,graph,profile)
    #text =
    #param_type = IntentPredictor.extract_keyword()
    kg_items = KGQuery.fuzzy_query()


    return combine_results(cf_items, kg_items, cf_weight=0.7, n=10)