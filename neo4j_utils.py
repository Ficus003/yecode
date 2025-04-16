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





#……………………………………………………………………
#查询功能
#……………………………………………………………………

def recommend_by_cuisine(cuisine,graph):
    try:
        query = """
        MATCH (d:Dish)-[:属于]->(c:Cuisine)
        WHERE toLower(c.name) CONTAINS toLower($cuisine)
        RETURN d.name as dish, d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
        """
        return graph.run(query, cuisine=cuisine).data()
    except Exception as e:
        print(f"[错误]菜系查询失败：{str(e)}")
        return []

def recommend_by_ingredient(ingredient,graph):
    try:
        query = """
        MATCH (d:Dish)-[:包含]->(i:Ingredient)
        WHERE toLower(i.name) CONTAINS toLower($ingredient)
        RETURN d.name as dish, d.flavor as flavor, d.calories as calories, d.spicy as spicy, d.salty as salty
        """
        return graph.run(query, ingredient=ingredient).data()
    except Exception as e:
        print(f"[错误]食材查询失败：{str(e)}")
        return []

