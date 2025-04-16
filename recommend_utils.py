import surprise
from surprise import Dataset,KNNBasic

#训练协同过滤模型(user_based CF)
#返回：训练好的模型
def train_collaborative_filtering(ratings_df):
    reader = surprise.Reader(rating_scale=(1,5))
    data = Dataset.load_from_df(ratings_df,reader)
    trainset = data.build_full_trainset()

    sim_options = {
        'name':'cosine',
        'user_based':True
    }

    model= KNNBasic(sim_options = sim_options)
    model.fit(trainset)

    return model

#
def collaborative_filtering(user_id,model,graph,ratings_data,n=10):

    all_dishes = set(get_all_dishes(graph))
    rated_dishes = set(get_rated_dishes(user_id, ratings_data))
    unrated_dishes = all_dishes - rated_dishes

    predictions = []
    for dish_id in unrated_dishes:
        pred = model.predict(user_id,dish_id)
        predictions.append((dish_id,pred.est))

    return sorted(predictions,key=lambda x: -x[1])[:n]

def get_all_dishes(graph):

    query = "MATCH (d:Dish) RETURN d.name as dish_id"
    results = graph.run(query).data()
    return [r['dish_id'] for r in results]

def get_rated_dishes(user_id,ratings_data):
    from collections import defaultdict
    user_ratings = defaultdict(list)

    for record in ratings_data:
        user_ratings[record['user_id']].append(record['dish_id'])

    return user_ratings.get(user_id, [])

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

    preferred_flavors = [f.lower() for f in user_profile['preferred_flavors']]
    restrictions = [r.lower() for r in user_profile.get('restrictions', [])]

    params = {
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
        return [(r['dish_id'], r['kg_score']) for r in results]
    except Exception as e:
        print("KG查询错误：", str(e))
        return []

def combine_results(cf_items, kg_items, alpha=0.7):

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
    for (dish_id, _), cf_norm in zip(cf_items, cf_scores):
        combined[dish_id] = alpha * cf_norm

    for (dish_id, _), kg_norm in zip(kg_items, kg_scores):
        combined[dish_id] = combined.get(dish_id, 0) + (1 - alpha) * kg_norm

    return sorted(combined.items(), key=lambda x: -x[1])

def normalize(scores):

    #处理空列表
    if not scores:
        return []

    min_s, max_s = min(scores), max(scores)
    #防止除0
    if max_s == min_s:
        return [0.5 for _ in scores]     #返回中性值

    return [(s - min_s) / (max_s - min_s + 1e-6) for s in scores]

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