from .node import MCTSNode, INF  # 导入MCTSNode类和INF常量
from .config import MCTSConfig  # 导入MCTSConfig类
from env.base_env import BaseGame  # 导入BaseGame类
from other_algo.heuristic import go_heuristic_evaluation  # 导入启发式评估函数

import numpy as np  # 导入numpy库

class UCTMCTSConfig(MCTSConfig):  # 定义UCTMCTSConfig类，继承自MCTSConfig
    def __init__(
        self,
        n_rollout:int = 1,  # 每次搜索的模拟次数，默认为1
        heuristic_weight: float = 0.3,  # 控制启发式权重，默认为0.3
        *args, **kwargs  # 其他参数
    ):
        MCTSConfig.__init__(self, *args, **kwargs)  # 调用父类的构造函数
        self.n_rollout = n_rollout  # 设置n_rollout属性
        self.heuristic_weight = heuristic_weight  # 设置heuristic_weight属性

class UCTMCTS:  # 定义UCTMCTS类
    def __init__(self, init_env: BaseGame, config: UCTMCTSConfig, root: MCTSNode = None):  # 构造函数
        self.config = config  # 设置config属性
        self.root = root  # 设置root属性
        if root is None:  # 如果root为空
            self.init_tree(init_env)  # 初始化树
        self.root.cut_parent()  # 切断根节点的父节点

    def init_tree(self, init_env: BaseGame):  # 初始化树的方法
        env = init_env.fork()  # 复制环境以避免副作用
        self.root = MCTSNode(
            action=None, env=env, reward=0,  # 创建根节点
        )

    def get_subtree(self, action: int):  # 获取子树的方法
        if self.root.has_child(action):  # 如果根节点有该动作的子节点
            new_root = self.root.get_child(action)  # 获取该子节点
            return UCTMCTS(new_root.env, self.config, new_root)  # 返回以该子节点为根的新树
        else:
            return None  # 否则返回None

    def uct_action_select(self, node: MCTSNode) -> int:  # 基于UCB选择最佳动作的方法
        total_visits = np.sum(node.child_N_visit)  # 计算子节点的总访问次数
        if total_visits == 0:  # 如果总访问次数为0
            total_visits = 1e-8  # 设置为一个很小的数以避免除零错误
        log_visits = np.log(total_visits)  # 计算总访问次数的对数

        best_uct = -np.inf  # 初始化最佳UCT值为负无穷
        best_action = -1  # 初始化最佳动作为-1
        for action in range(node.n_action):  # 遍历所有动作
            if node.action_mask[action] == 0:  # 如果动作不可用
                continue  # 跳过该动作
            n = node.child_N_visit[action]  # 获取该动作的访问次数
            if n == 0:  # 如果访问次数为0
                return action  # 优先选择未访问的合法动作
            q = node.child_V_total[action] / n  # 计算该动作的平均价值
            uct = q + self.config.C * np.sqrt(log_visits / n)  # 计算UCT值
            if uct > best_uct:  # 如果当前UCT值大于最佳UCT值
                best_uct = uct  # 更新最佳UCT值
                best_action = action  # 更新最佳动作
        return best_action  # 返回最佳动作

    def backup(self, node: MCTSNode, value: float) -> None:  # 回溯更新节点值的方法
        current = node  # 从当前节点开始
        while current.parent is not None:  # 直到根节点
            action = current.action  # 获取当前节点的动作
            current.parent.child_N_visit[action] += 1  # 更新父节点的访问次数
            current.parent.child_V_total[action] += value  # 更新父节点的总价值
            value = -value  # 反转价值
            current = current.parent  # 移动到父节点

    def rollout(self, node: MCTSNode) -> float:  # 模拟游戏直到结束的方法
        env = node.env.fork()  # 复制环境
        current_player = env.current_player  # 获取当前玩家
        while not env.ended:  # 直到游戏结束
            valid_actions = np.where(env.action_mask == 1)[0]  # 获取所有合法动作
            if len(valid_actions) == 0:  # 如果没有合法动作
                break  # 退出循环
            action = np.random.choice(valid_actions)  # 随机选择一个合法动作
            _, reward, done = env.step(action)  # 执行动作
        final_reward = reward if env.current_player == current_player else -reward  # 计算最终奖励
        return final_reward  # 返回最终奖励
        # 启发式搜索部分
        # heuristic_value = go_heuristic_evaluation(env)
        # combined_reward = (
        #     (1 - self.config.heuristic_weight) * final_reward + 
        #     self.config.heuristic_weight * heuristic_value
        # )
        # return combined_reward

    def pick_leaf(self) -> MCTSNode:  # 选择叶子节点进行扩展的方法
        current = self.root  # 从根节点开始
        while True:
            if current.done:  # 如果当前节点已完成
                return current  # 返回当前节点
            action = self.uct_action_select(current)  # 选择最佳动作
            if action == -1:  # 如果没有合法动作
                return current  # 返回当前节点
            if not current.has_child(action):  # 如果当前节点没有该动作的子节点
                return current.add_child(action)  # 添加子节点并返回
            current = current.get_child(action)  # 移动到子节点

    def get_policy(self, node: MCTSNode = None) -> np.ndarray:  # 返回搜索后的树的策略的方法
        if node is None:  # 如果节点为空
            node = self.root  # 设置为根节点
        masked_visits = node.child_N_visit * node.action_mask  # 计算每个动作的访问次数
        total = np.sum(masked_visits)  # 计算总访问次数
        if total == 0:  # 如果总访问次数为0
            policy = node.action_mask.astype(np.float32)  # 设置策略为动作掩码
            policy /= np.sum(policy)  # 归一化策略
        else:
            policy = masked_visits / total  # 计算策略
        return policy  # 返回策略

    def search(self):  # 搜索树的方法
        for _ in range(self.config.n_search):  # 搜索n_search次
            leaf = self.pick_leaf()  # 选择叶子节点
            if leaf.done:  # 如果叶子节点已完成
                value = leaf.reward  # 获取叶子节点的奖励
            else:
                value = 0  # 初始化奖励为0
                for _ in range(self.config.n_rollout):  # 模拟n_rollout次
                    value += self.rollout(leaf)  # 累加模拟的奖励
                value /= self.config.n_rollout  # 计算平均奖励
            self.backup(leaf, value)  # 回溯更新节点值
        return self.get_policy()  # 返回搜索后的树的策略