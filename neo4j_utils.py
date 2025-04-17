import pandas as pd
from py2neo import Graph,Node,Relationship
from config import PREDEFINED_PARAMS
from predictor import IntentPredictor
#………………………………………………………………
#数据库连接
#………………………………………………………………

#auth是用户名与密码，name为数据库名，neo4j5.x版本要求必需定义数据库名
def safe_graph_connect():

    try:
        return Graph("bolt://localhost:7687",
                  auth=("neo4j", "12345678"),
                  name="neo4j")
    except Exception as e:
        print(f"【严重错误】数据库连接失败：{str(e)}")
        exit(1)

#………………………………………………………………
#知识图谱构建
#………………………………………………………………

def build_knowledge_graph(csv_path,graph):

    try:
        df = pd.read_csv(csv_path)
        graph.delete_all()

        for _,row in df.iterrows():
            dish = Node("Dish",name=row['name'],
                        flavor=row['flavor'],
                        calories=int(row['calories']),
                        spicy = int(row['spicy']),
                        salty = int(row['salty']))
            graph.create(dish)

            cuisine = Node("Cuisine",name=row['cuisine'])
            graph.merge(cuisine,"Cuisine","name")
            graph.create(Relationship(dish,"属于",cuisine))

            ingredients = [x.strip() for x in row['ingredients'].split('、')]
            for ing in ingredients:
                ingredient = Node("Ingredient",name=ing)
                graph.merge(ingredient,"Ingredient","name")
                graph.create(Relationship(dish,"包含",ingredient))
    except Exception as e:
        print(f"【ERROR】知识图谱构建失败：{str(e)}")
        raise
#创建用户画像图谱
def build_user_profile(rating_csv_path, graph):
    try:
        rating_df = pd.read_csv(rating_csv_path)
        for _, row in rating_df.iterrows():
            user = Node("User", user_id=row['user_id'])
            dish = Node("Dish",name=row['dish_id'])

            rated = Relationship(user, "评分", dish, rating=int(row['rating']))
            graph.merge(rated)

    except Exception as e:
        print(f"【ERROR】知识图谱构建失败：{str(e)}")
        raise

#封装知识图谱查询函数

class KGQuery:
    def __init__(self):
        self.graph = safe_graph_connect()

    def process_query(self, text):
        """
        判断参数'cuisine''flavor''ingredient'，返回对应模糊查询函数
        :param text: 用户输入的文本
        :return: 对应模糊查询函数
        """
        intent, param_type = IntentPredictor.parse(text)
        kg_items = []
        if intent == "recommend":
            temp_results = self.query_all_dishes()
            if "spicy" in param_type:
                temp_results = [dish for dish in temp_results if dish["spicy"] <= PREDEFINED_PARAMS["spicy"].get(param_type["spicy"])]
            if "calories" in param_type:
                calories_limit = PREDEFINED_PARAMS["calories"].get(param_type["calories"])
                temp_results = [dish for dish in temp_results if dish["calories"] <= calories_limit]
            if "salty" in param_type:
                salty_limit = PREDEFINED_PARAMS["salty"].get(param_type["salty"])
                temp_results = [dish for dish in temp_results if dish["salty"] <= salty_limit]

            if "cuisine" in param_type:
                cuisine_dishes = self.query_cuisine(param_type["cuisine"])
                temp_results = [dish for dish in temp_results if dish["dish"] in {d["dish"] for d in cuisine_dishes}]
            if "flavor" in param_type:
                flavor_dishes = self.query_flavor(param_type["flavor"])
                temp_results = [dish for dish in temp_results if dish["dish"] in {d["dish"] for d in flavor_dishes}]
            if "ingredient" in param_type:
                ingredient_dishes = self.query_ingredient(param_type["ingredient"])
                temp_results = [dish for dish in temp_results if dish["dish"] in {d["dish"] for d in ingredient_dishes}]

            if "exclude_ingredient" in param_type:
                exclude_dishes = self.query_exclude_ingredient(param_type["exclude_ingredient"])
                temp_results = [dish for dish in temp_results if dish["dish"] in {d["dish"] for d in exclude_dishes}]

            results = temp_results

            for dish in temp_results:
                score = 1.0
                if "spicy" in param_type:
                    desired = PREDEFINED_PARAMS["spicy"].get(param_type["spicy"])
                    score -= abs(dish["spicy"] - desired) * 0.2

                if "calories" in param_type:
                    desired = PREDEFINED_PARAMS["calories"].get(param_type["calories"])
                    score -= max(0,(dish["calories"] - desired) / 1000)

                if "salty" in param_type:
                    desired = PREDEFINED_PARAMS["salty"].get(param_type["salty"])
                    score -= abs(dish["salty"] - desired) * 0.1

                kg_items.append((dish["dish"],score))

        else:
            print("无法识别用户意图，返回空列表")
        return kg_items
#……………………………………………………………………
#查询功能
#……………………………………………………………………

    def query_all_dishes(self):
        query = """
        MATCH (d:Dish)
        RETURN d.name as dish,  d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
        """
        return self.graph.run(query).data()

    def query_cuisine(self, text):
        try:
            #返回名字最短的菜名
            query = """
            MATCH(d:Dish)-[:属于]->(c:Cuisine)
            WHERE toLower(c.name) CONTAINS toLower($text)
            RETURN d.name as dish,  d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            
            """
            return self.graph.run(query, text=text).data()
        except Exception as e:
            print(f"[错误]菜系查询失败：{str(e)}")
            return []


    def query_ingredient(self, text):
        try:
            query = """
            MATCH(d:Dish)-[:包含]->(i:Ingredient)
            WHERE toLower(i.name) CONTAINS toLower($text)
            RETURN d.name as dish,  d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            
            """
            return self.graph.run(query, text=text).data()

        except Exception as e:
            print(f"[错误]食材查询失败：{str(e)}")
            return []

    def query_exclude_ingredient(self, text):
        try:
            query = """
            MATCH (d:Dish)
            WHERE NOT EXISTS {
                MATCH (d)-[:包含]->(i:Ingredient)
                WHERE toLower(i.name) CONTAINS toLower($text)
            }
            RETURN d.name as dish,  d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            """
            return self.graph.run(query, text=text).data()
        except Exception as e:
            print(f"[错误]食材查询失败：{str(e)}")
            return []


    def query_flavor(self, text):
        try:
            #SPLIT以逗号分隔，把flavor字符分割成列表
            #ANY关键字检查是否有一个flavor与用户输入的文本对应
            text_list = [text.lower()]     #转换成列表
            query = """
            MATCH(d:Dish)
            WHERE ANY (flavor IN SPLIT(d.flavor, ',') WHERE toLower(flavor) IN $text)
            RETURN d.name as dish,  d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
           
            """
            return self.graph.run(query, text=text_list).data()

        except Exception as e:
            print(f"[错误]食材查询失败：{str(e)}")
            return []

    def query_spicy(self,spicy_level):
        spicy_value = PREDEFINED_PARAMS["spicy"].get(spicy_level)
        if spicy_value is None:
            print(f"[错误]未找到对应辣度: {spicy_level}")
            return []

        try:
            query = """
            MATCH (d:Dish)
            WHERE d.spicy = $spicy_value
            RETURN d.name as dish, d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            """
            return self.graph.run(query, spicy_value=spicy_value).data()
        except Exception as e:
            print(f"[错误]辣度查询失败：{str(e)}")
            return []


    def query_calories(self,calories):
        calories_value = PREDEFINED_PARAMS["calories"].get(calories)
        if calories_value is None:
            print(f"[错误]未找到对应卡路里: {calories}")
            return []

        try:
            query = """
            MATCH (d:Dish)
            WHERE d.calories <= $calories_value
            RETURN d.name as dish, d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            """
            return self.graph.run(query, calories_value=calories_value).data()
        except Exception as e:
            print(f"[错误]卡路里查询失败：{str(e)}")
            return []

    def query_salty(self,salty_level):
        salty_value = PREDEFINED_PARAMS["salty"].get(salty_level)
        if salty_value is None:
            print(f"[错误]未找到对应咸度: {salty_level}")
            return []

        try:
            query = """
            MATCH (d:Dish)
            WHERE d.salty <= $salty_value
            RETURN d.name as dish, d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            """
            return self.graph.run(query, salty_value=salty_value).data()
        except Exception as e:
            print(f"[错误]咸度查询失败：{str(e)}")
            return []

