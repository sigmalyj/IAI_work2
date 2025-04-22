from .node import MCTSNode, INF  # 导入 MCTS 节点类和无穷大常量
from .config import MCTSConfig  # 导入 MCTS 配置类
from env.base_env import BaseGame  # 导入基础游戏环境类

from model.linear_model_trainer import NumpyLinearModelTrainer  # 导入线性模型训练器
import numpy as np  # 导入 NumPy 库


class PUCTMCTS:
    """
    基于 PUCT（Polynomial Upper Confidence Trees）的蒙特卡洛树搜索实现。
    """

    def __init__(self, init_env: BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root: MCTSNode = None):
        """
        初始化 PUCTMCTS 对象。

        参数：
        - init_env: 初始游戏环境。
        - model: 用于预测策略和价值的模型。
        - config: MCTS 配置对象。
        - root: 可选的根节点，如果未提供，则初始化一个新树。
        """
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)  # 初始化搜索树
        self.root.cut_parent()  # 切断根节点与父节点的连接（如果有）

    def init_tree(self, init_env: BaseGame):
        """
        初始化搜索树。

        参数：
        - init_env: 初始游戏环境。
        """
        env = init_env.fork()  # 复制游戏环境
        obs = env.observation  # 获取当前观察
        self.root = MCTSNode(
            action=None, env=env, reward=0  # 创建根节点
        )
        # 计算并保存预测的策略
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)  # 设置子节点的先验概率

    def get_subtree(self, action: int):
        """
        获取指定动作后的子树。

        参数：
        - action: 动作编号。

        返回：
        - 子树的根节点对应的 PUCTMCTS 对象。
        """
        if self.root.has_child(action):  # 如果根节点有对应的子节点
            new_root = self.root.get_child(action)  # 获取子节点
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)  # 返回新的子树
        else:
            return None

    def puct_action_select(self, node: MCTSNode):
        """
        基于 PUCT 公式选择最佳动作。

        参数：
        - node: 当前节点。

        返回：
        - 最佳动作编号。
        """
        total_visits = np.sum(node.child_N_visit)  # 所有子节点的访问次数总和
        if total_visits == 0:
            total_visits = 1e-8  # 避免除以零
        log_visits = np.log(total_visits + 1)  # 计算对数访问次数

        best_puct = -np.inf  # 初始化最佳 PUCT 值
        best_action = -1  # 初始化最佳动作
        for action in range(node.n_action):  # 遍历所有动作
            if node.action_mask[action] == 0:  # 跳过无效动作
                continue
            n = node.child_N_visit[action]  # 子节点的访问次数
            q = node.child_V_total[action] / n if n > 0 else 0  # 子节点的平均价值
            p = node.child_priors[action]  # 子节点的先验概率
            puct = q + self.config.C * p * np.sqrt(log_visits) / (1 + n)  # 计算 PUCT 值
            if puct > best_puct:  # 更新最佳 PUCT 值和动作
                best_puct = puct
                best_action = action
        return best_action

    def backup(self, node: MCTSNode, value):
        """
        回溯更新节点的访问次数和价值。

        参数：
        - node: 叶节点。
        - value: 叶节点的价值。
        """
        current = node
        while current.parent is not None:  # 从叶节点回溯到根节点
            action = current.action
            current.parent.child_N_visit[action] += 1  # 更新访问次数
            current.parent.child_V_total[action] += value  # 更新总价值
            value = -value  # 对手的价值取反
            current = current.parent

    def pick_leaf(self):
        """
        选择要扩展的叶节点。

        返回：
        - 叶节点。
        """
        current = self.root
        while True:
            if current.done:  # 如果当前节点是终止节点
                return current
            action = self.puct_action_select(current)  # 选择最佳动作
            if action == -1 or not current.has_child(action):  # 如果动作无效或子节点不存在
                return current.add_child(action)  # 创建并返回新子节点
            current = current.get_child(action)  # 移动到子节点

    def get_policy(self, node: MCTSNode = None):
        """
        获取搜索树的策略分布。

        参数：
        - node: 可选的节点，默认为根节点。

        返回：
        - 策略分布（基于访问次数）。
        """
        if node is None:
            node = self.root
        masked_visits = node.child_N_visit * node.action_mask  # 仅考虑有效动作的访问次数
        total = np.sum(masked_visits)  # 总访问次数
        if total == 0:  # 如果没有访问过任何动作
            policy = node.action_mask.astype(np.float32)  # 使用动作掩码作为策略
            policy /= np.sum(policy)  # 归一化
        else:
            policy = masked_visits / total  # 计算策略分布
        return policy

    def search(self):
        """
        执行 MCTS 搜索。

        返回：
        - 根节点的策略分布。
        """
        for _ in range(self.config.n_search):  # 重复搜索指定次数
            leaf = self.pick_leaf()  # 选择叶节点
            if leaf.done:  # 如果叶节点是终止节点
                value = leaf.reward  # 使用叶节点的奖励作为价值
            else:
                obs = leaf.env.compute_canonical_form_obs(leaf.env.observation, leaf.env.current_player)  # 获取规范化观察
                policy, value = self.model.predict(obs)  # 使用模型预测策略和价值
                leaf.set_prior(policy)  # 设置叶节点的先验概率
            self.backup(leaf, value)  # 回溯更新
        return self.get_policy(self.root)  # 返回根节点的策略分布