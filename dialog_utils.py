from datetime import datetime
class DialogState:
    def __init__(self):
        self.session_id = self._generate_session_id()
        self.current_intent = None
        self.slots = {}
        self.history = []
        self.recommended_dishes = []
        self.preferences = {}
        self.context = []
        self.created_at = datetime.now()
        self.last_updated = datetime.now()

    def _generate_session_id(self):
        return f"sess_{int(datetime.now().timestamp())}"

    def update(self, role, text, intent=None, slots=None,recommended=None):
        """
        更新对话状态

        :param role: str - 说话的角色：'user' or 'system'
        :param text: str - 发言的内容
        :param intent:str - 识别到的意图
        :param slots: dict - 提取的槽位信息
        :param recommended:list[str] - 本次推荐菜品列表
        :return:none
        """

        if role not in ['user', 'system']:
            raise ValueError("role必须是'user'或'system'")

        self.history.append((role,text,intent))

        if role == 'user' and intent:
            self.current_intent = intent

        if slots:
            self.slots.update(slots)

        if recommended:
            self.recommended_dishes.extend(recommended)

        self.context.append(f"{role}: {text}")
        if len(self.context) > 3:
            self.context.pop(0)

        self.last_updated = datetime.now()


    def reset(self, keep_preferences=False):
        """
        重置对话状态
        :param keep_preferences:bool - 是否保留用户偏好
        :return: none
        """
        preferences = self.preferences if keep_preferences else {}
        self.__init__()
        self.preferences = preferences


    def get_context(self, n=3, with_intent=False):
        """
        获取最近n轮对话上下文

        :param n: int - 获取轮次
        :param with_intent: bool - 是否包含意图信息
        :return: list[str] - 格式化后的对话上下文

        example:
        输入:n=2,with_intent=True
        输出:["user:推荐川菜(intent:recommend_cuisine)",
             "system:您偏好什么辣度？"]

        """
        recent_history = self.history[-n:] if self.history else []

        if with_intent:
            return [
                f"{role}:{text}(intent:{intent})" if intent else f"{role}:{text}"
                for role, text, intent in recent_history
            ]
        else:
            return [f"{role}:{text}" for role, text, _ in recent_history]

    def is_recommended_before(self, dish_name):

        return dish_name in self.recommended_dishes

    def get_slot(self, slot_name, default=None):
        return self.slots.get(slot_name, default)

