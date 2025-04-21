from .node import MCTSNode, INF
from .config import MCTSConfig
from env.base_env import BaseGame

from model.linear_model_trainer import NumpyLinearModelTrainer
import numpy as np


class PUCTMCTS:
    def __init__(self, init_env:BaseGame, model: NumpyLinearModelTrainer, config: MCTSConfig, root:MCTSNode=None):
        self.model = model
        self.config = config
        self.root = root
        if root is None:
            self.init_tree(init_env)
        self.root.cut_parent()
    
    def init_tree(self, init_env:BaseGame):
        env = init_env.fork()
        obs = env.observation
        self.root = MCTSNode(
            action=None, env=env, reward=0
        )
        # compute and save predicted policy
        child_prior, _ = self.model.predict(env.compute_canonical_form_obs(obs, env.current_player))
        self.root.set_prior(child_prior)
    
    def get_subtree(self, action:int):
        # return a subtree with root as the child of the current root
        # the subtree represents the state after taking action
        if self.root.has_child(action):
            new_root = self.root.get_child(action)
            return PUCTMCTS(new_root.env, self.model, self.config, new_root)
        else:
            return None
    
    def puct_action_select(self, node: MCTSNode):
        # select the best action based on PUCB when expanding the tree
        total_visits = np.sum(node.child_N_visit)
        if total_visits == 0:
            total_visits = 1e-8  # Avoid division by zero
        log_visits = np.log(total_visits + 1)

        best_puct = -np.inf
        best_action = -1
        for action in range(node.n_action):
            if node.action_mask[action] == 0:  # Skip invalid actions
                continue
            n = node.child_N_visit[action]
            q = node.child_V_total[action] / n if n > 0 else 0
            p = node.child_priors[action]
            puct = q + self.config.C * p * np.sqrt(log_visits) / (1 + n)
            if puct > best_puct:
                best_puct = puct
                best_action = action
        return best_action

    def backup(self, node: MCTSNode, value):
        # backup the value of the leaf node to the root
        # update N_visit and V_total of each node in the path
        current = node
        while current.parent is not None:
            action = current.action
            current.parent.child_N_visit[action] += 1
            current.parent.child_V_total[action] += value
            value = -value  # Alternate value for the opponent
            current = current.parent

    def pick_leaf(self):
        # select the leaf node to expand
        # the leaf node is the node that has not been expanded
        # create and return a new node if game is not ended
        current = self.root
        while True:
            if current.done:
                return current
            action = self.puct_action_select(current)
            if action == -1 or not current.has_child(action):
                return current.add_child(action)
            current = current.get_child(action)

    def get_policy(self, node: MCTSNode = None):
        # return the policy of the tree(root) after the search
        # the policy comes from the visit count of each action
        if node is None:
            node = self.root
        masked_visits = node.child_N_visit * node.action_mask
        total = np.sum(masked_visits)
        if total == 0:
            policy = node.action_mask.astype(np.float32)
            policy /= np.sum(policy)
        else:
            policy = masked_visits / total
        return policy

    def search(self):
        for _ in range(self.config.n_search):
            leaf = self.pick_leaf()
            if leaf.done:
                value = leaf.reward
            else:
                obs = leaf.env.compute_canonical_form_obs(leaf.env.observation, leaf.env.current_player)
                policy, value = self.model.predict(obs)
                leaf.set_prior(policy)
            self.backup(leaf, value)
        return self.get_policy(self.root)