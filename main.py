import csv
from recommend_utils import  *


from neo4j_utils import *
from init_dishes_data import *

#………………………………………………………………
#数据管理
#………………………………………………………………



def save_dishes_to_csv(dishes,filename):    #写入数据到csv
    #dishes:菜品列表，每个菜品为字典格式，在init_sample_data()中
    #filename:csv文件名（’dishes_data.csv‘）

    with open(filename,'w',newline='',encoding='utf-8-sig')as f:
        #newline='':避免在windows系统下出现空行
        #encoding='utf-8-sig':防止中文乱码

        fieldnames = ['name','cuisine','ingredients','flavor','calories','spicy','salty']
        writer = csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()    #写入表头
        writer.writerows(dishes)

def load_dishes_from_csv(filname):
    with open(filname,mode='r',encoding='utf-8-sig')as file:
        reader = csv.DictReader(file)
        return list(reader)

def add_dish(dishes,dish_info):
    dishes.append(dish_info)
    return dishes




#………………………………………………………………………………
#交互功能
#………………………………………………………………………………

def print_results(results,title):

    if not results:
        print(f"\n 未找到{title}，请尝试其他关键词")
        return

    print(f"\n==={title}===")
    for idx,item in enumerate(results,1):
        print(f"{idx}.{item['dish']}，风味：{item['flavor']}，热量：{item['calories']}，辣度：{item['spicy']}，含盐指数：{item['salty']}")

def parse_command(user_input):

    if user_input.startswith("推荐 "):
        return "cuisine",user_input[3:].strip()
    elif user_input.startswith("查询 "):
        return "ingredient",user_input[3:].strip()
    return None,None

def main_loop(graph):
    print("===菜品推荐系统===")
    print("指令格式:\n- 推荐 [菜系]\n- 查询 [食材]\n- 退出")

    while True:
        try:
            user_input = input("\n请输入指令:").strip().lower()

            if not user_input:
                print("请按格式输入，如：’推荐 粤菜‘或’退出‘")
                continue

            if user_input == "退出":
                print("感谢使用！")
                break

            command_type,param = parse_command(user_input)

            if not param:
                print("请输入有效参数，如：’推荐 川菜‘")
                continue

            if command_type =="cuisine":
                results = recommend_by_cuisine(param,graph)
                print_results(results,f"{param}菜品推荐")
            elif command_type =="ingredient":
                results = recommend_by_ingredient(param,graph)
                print_results(results,f"包含{param}的菜品")
            else:
                print("无效指令")

        except KeyboardInterrupt:
            print("\n检测到中断指令，安全退出……")
            break
        except Exception as e:
            print(f"【系统异常】请重试，错误详情：{str(e)}")

def main():

    graph = safe_graph_connect()
    dishes = init_sample_data()

    save_dishes_to_csv(dishes,'dishes_data.csv')
    build_knowledge_graph('dishes_data.csv',graph)

    ratings_data = [
        {'user_id': 'user1', 'dish_id': '麻婆豆腐', 'rating': 5},
        {'user_id': 'user1', 'dish_id': '回锅肉', 'rating': 4},
        {'user_id': 'user2', 'dish_id': '清炒时蔬', 'rating': 3}
    ]

    #模型训练
    ratings_df = pd.DataFrame(ratings_data)
    cf_model = train_collaborative_filtering(ratings_df)

    #测试推荐
    user_id = 'user1'
    profile = get_user_profile(user_id, graph)
    recommendations = hybrid_recommend(user_id, graph, cf_model, ratings_data)
    print("所有菜品", get_all_dishes(graph)[:5])
    print()
    print("user1已评分：", get_rated_dishes('user1', ratings_data))
    print()
    cf_test = collaborative_filtering('user1', cf_model, graph, ratings_data)
    print("cf推荐：", cf_test)
    print()
    print("混和推荐结果：", recommendations)

    main_loop(graph)

    #loaded_dishes = load_dishes_from_csv('dishes_data.csv')
    #print("加载的菜品：",loaded_dishes)

    #new_dishes = {'name':'蒜泥白肉', 'cuisine':'川菜', 'ingredients':'五花肉、蒜泥、辣椒油', 'flavor':'蒜香微辣'}
    #dishes = add_dish(dishes,new_dishes)
    #save_dishes_to_csv(dishes,'dishes_data.csv')

if __name__ == "__main__":
    main()
