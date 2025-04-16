import pandas as pd
from py2neo import Graph,Node,Relationship

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

class KHQuery:
    def __init__(self):
        self.graph = safe_graph_connect()

    def fuzzy_query(self, param_type, text):
        """
        判断参数'cuisine''flavor''ingredient'，返回对应模糊查询函数
        :param param_type: 查询类型, 'cuisine''flavor''ingredient'
        :param text: 用户输入的文本
        :return: 对应模糊查询函数
        """
        if param_type =="cuisine":
            return self.query_cuisine(text)
        elif param_type == "flavor":
            return self.query_flavor(text)
        elif param_type == "ingredient":
            return self.query_ingredient(text)
        return None
#……………………………………………………………………
#查询功能
#……………………………………………………………………

    def query_cuisine(self, text):
        try:
            #返回名字最短的菜名
            query = """
            MATCH(d:Dish)-[:属于]->(c:Cuisine)
            WHERE toLower(c.name) CONTAINS toLower($text)
            RETURN d.name as dish,  d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            ORDER BY length(d.name) ASC
            LIMIT 1
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
            ORDER BY length(d.name) ASC
            LIMIT 1
            """
            return self.graph.run(query, text=text).data()

        except Exception as e:
            print(f"[错误]食材查询失败：{str(e)}")
            return []

    def query_flavor(self,text):
        try:
            #SPLIT以逗号分隔，把flavor字符分割成列表
            #ANY关键字检查是否有一个flavor与用户输入的文本对应
            query = """
            MATCH(d:Dish)
            ANY (flavor IN SPLIT(d.flavor, ',') WHERE toLower(flavor) IN $text)
            RETURN d.name as dish,  d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
            ORDER BY length(d.name) ASC
            LIMIT 1
            """
            return self.graph.run(query, text=text).data()

        except Exception as e:
            print(f"[错误]食材查询失败：{str(e)}")
            return []