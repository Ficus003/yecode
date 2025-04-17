import pandas as pd
import surprise
from surprise import Dataset,KNNBasic
from collections import defaultdict
from neo4j_utils import KGQuery

class CollaborativeRecommender:
    def __init__(self, user_profile):
        self.user_profile = user_profile
        self.model = self.train_collaborative_filtering()

    #训练协同过滤模型(user_based CF)
    #返回：训练好的模型
    def train_collaborative_filtering(self):
        """
        训练基于用户协同过滤推荐模型
        使用surprise推荐系统库

        """
        ratings_df = pd.DataFrame(self.user_profile)
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
    def collaborative_filtering(self, user_id, graph, n=10):

        all_dishes = self.get_all_dishes(graph)   #获取知识图谱中所有菜品'name'放入集合
        rated_dishes = self.get_rated_dishes(user_id)   #调用get_rated_dishes函数，找出用户(user_id)已经评分过的菜品，放入集合
        unrated_dishes = set(all_dishes) - set(rated_dishes)  #差集，得出未评分菜品集合

        predictions = []   #创建预测菜品列表
        for dish_id in unrated_dishes:   #遍历未评分菜品
            pred = self.model.predict(user_id, dish_id)  #预测用户对该菜品评分
            predictions.append((dish_id, pred.est)) #(菜品id，预测评分)放入predictions列表

        return sorted(predictions, key=lambda x: -x[1])[:n]   #对评分排序，从高到低，取前n项

    #从知识图谱获取所有菜品id
    @staticmethod
    def get_all_dishes(graph):
        query = "MATCH (d:Dish) RETURN d.name as dish_id"
        results = graph.run(query).data()
        return [r['dish_id'] for r in results]

    #获取用户已评分菜品列表
    def get_rated_dishes(self, user_id):
        user_rated = defaultdict(list)     #defaultdict存储用户评分记录,字典
        for record in self.user_profile:    #遍历user_profile
            user_rated[record['user_id']].append(record['dish_id']) #按照user_id分类，放入用户评分过的dish_id，放在user_ratings
        return user_rated.get(user_id, [])  #从user_ratings取出对应的已经评分了的菜品列表，如果没有，就返回一个空列表[]


class ResultCombiner:


    # 归一化函数，调用后使CF评分缩放在（0，1）范围内
    @staticmethod
    def normalize(scores):
        # 处理空列表
        if not scores:
            return []
        min_s, max_s = min(scores), max(scores)
        # 防止除0
        if max_s == min_s:
            return [0.5 for _ in scores]  # 返回中性值
        return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]  # (当前分数-最小分数)/(最大值-最小值),+1e-6:防止计算机浮点误差


    #将CF评分与KG评分通过加权平均方法，合成最终推荐列表,kg_score为默认的1
    @staticmethod
    def combine(cf_items, kg_items, cf_weight=0.7, n=10):  #cf_weight,cf评分的权重系数

        #处理空列表
        if not cf_items and not kg_items:
            return []

        cf_dict = dict(cf_items)
        kg_dict = dict(kg_items)

        #归一化数值范围
        if cf_items:
            cf_norm_score = ResultCombiner.normalize(list(cf_dict.values()))
            cf_dict = dict(zip(cf_dict.keys(), cf_norm_score))

        if kg_items:
            kg_norm_score = ResultCombiner.normalize(list(kg_dict.values()))
            kg_dict = dict(zip(cf_dict.keys(), kg_norm_score))

        #只有cf时
        if cf_items and not kg_items:
            return sorted(cf_dict.items(), key=lambda x: x[1], reverse=True)[:n]

        #只有kg时
        if kg_items and not cf_items:
            return sorted(kg_dict.items(), key=lambda x: x[1], reverse=True)[:n]

        combined = []
        #整合cf和kg的结果
        for dish in set(cf_dict.keys()).union(kg_dict.keys()):
            cf_score = cf_dict.get(dish, 0)
            kg_score = kg_dict.get(dish, 0)
            final_score = cf_weight * cf_score + (1 - cf_weight) * kg_score
            combined.append((dish, final_score))
        #知识图谱权重：(1-cf_score)

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:n]


def get_user_profile(user_id,graph):
        rating_query = """
        MATCH (u:User)-[r:rated]->(d:Dish)
        WHERE u.user_id = $user_id
        RETURN u.user_id as user_id, d.name as dish_id, r.rating as rating
        """
        with graph.session() as session:
            result = graph.run(rating_query, user_id=user_id)
            return [{"user_id":record["user_id"], "dish_id":record["dish_id"], "rating":record["rating"]}
                    for record in result]

def hybrid_recommend(user_id, graph, user_text):

    profile = get_user_profile(user_id,graph)

    cf_model = CollaborativeRecommender(profile)
    cf_items = cf_model.collaborative_filtering(user_id, graph)

    kg_query = KGQuery()
    kg_items = kg_query.process_query(user_text)


    return ResultCombiner.combine(cf_items, kg_items, cf_weight=0.7, n=10)