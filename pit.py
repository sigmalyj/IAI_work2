from env import *
from players import *
from mcts.uct_mcts import UCTMCTSConfig
from tqdm import trange, tqdm
import logging
import numpy as np
import os
import argparse

def print_devide_line(n=50):
    return "--" * n

def generate_log_filename(player1, player2):
    log_dir = 'log'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if isinstance(game, GobangGame):
        size_info = f"{game.n}x{game.m}_{game.n_in_row}"
    else:
        size_info = f"C{config.C}_r{config.n_rollout}_s{config.n_search}"
    filename = f"{game.__class__.__name__}_{size_info}_{player1.__class__.__name__}_vs_{player2.__class__.__name__}.log"
    return os.path.join(log_dir, filename)

def pit(game:BaseGame, player1:BasePlayer, player2:BasePlayer, log_output:bool=True):
    game.reset()
    if log_output:
        logging.info(f"start playing {game.__class__.__name__}")
        logging.info(print_devide_line())
    reward = 0
    
    for player in [player1, player2]:
        if player.__class__.__name__ == 'UCTPlayer':
            player.clear()
            
    while True:
        a1 = player1.play(game)
        _, reward, done = game.step(a1)
        if player2.__class__.__name__ == 'UCTPlayer':
            player2.opp_play(a1)
        if log_output:
            logging.info(f"Player 1 ({player1}) move: {a1}")
            logging.info(game.to_string())
            logging.info(print_devide_line())
        if done:
            break
        a2 = player2.play(game)
        _, reward, done = game.step(a2)
        if player1.__class__.__name__ == 'UCTPlayer':
            player1.opp_play(a2)
        if log_output:
            logging.info(f"Player 2 ({player2}) move: {a2}")
            logging.info(game.to_string())
            logging.info(print_devide_line())
        if done:
            reward *= -1
            break
    if log_output:
        if reward == 1:
            logging.info(f"Player 1 ({player1}) win")
        elif reward == -1:
            logging.info(f"Player 2 ({player2}) win")
        else:
            logging.info("Draw")
    return reward
        
def multi_match(game:BaseGame, player1:BasePlayer, player2:BasePlayer, n_match=100):
    print("n_rollout:", config.n_rollout)
    print("n_search:", config.n_search)
    print(f"GameType:{game.__class__.__name__}")
    print(f"Player 1:{player1}  Player 2:{player2}")
    logging.info("n_rollout: %d", config.n_rollout)
    logging.info("n_search: %d", config.n_search)
    logging.info(f"GameType:{game.__class__.__name__}")
    logging.info(f"Player 1:{player1}  Player 2:{player2}")
    n_p1_win, n_p2_win, n_draw = 0, 0, 0
    T = trange(n_match)
    for _ in T:
        reward = pit(game, player1, player2, log_output=True)  # 启用日志输出
        if reward == 1:
            n_p1_win += 1
        elif reward == -1:
            n_p2_win += 1
        else:
            n_draw += 1
        T.set_description_str(f"P1 win: {n_p1_win} ({n_p1_win}) P2 win: {n_p2_win} ({n_p2_win}) Draw: {n_draw} ({n_draw})") 
    print(f"Player 1 ({player1}) win: {n_p1_win} ({n_p1_win/n_match*100:.2f}%)")
    print(f"Player 2 ({player2}) win: {n_p2_win} ({n_p2_win/n_match*100:.2f}%)")
    print(f"Draw: {n_draw} ({n_draw/n_match*100:.2f}%)")
    print(f"Player 1 not lose: {n_p1_win+n_draw} ({(n_p1_win+n_draw)/n_match*100:.2f}%)")
    print(f"Player 2 not lose: {n_p2_win+n_draw} ({(n_p2_win+n_draw)/n_match*100:.2f}%)")
    logging.info(f"Player 1 ({player1}) win: {n_p1_win} ({n_p1_win/n_match*100:.2f}%)")
    logging.info(f"Player 2 ({player2}) win: {n_p2_win} ({n_p2_win/n_match*100:.2f}%)")
    logging.info(f"Draw: {n_draw} ({n_draw/n_match*100:.2f}%)")
    logging.info(f"Player 1 not lose: {n_p1_win+n_draw} ({(n_p1_win+n_draw)/n_match*100:.2f}%)")
    logging.info(f"Player 2 not lose: {n_p2_win+n_draw} ({(n_p2_win+n_draw)/n_match*100:.2f}%)")
    return n_p1_win, n_p2_win, n_draw
        
        
def search_best_C():
    from matplotlib import pyplot as plt
    p2nl = []
    cs = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.5, 5.0]
    n_match = 100
    for c in cs:
        config = UCTMCTSConfig()
        config.C = c
        config.n_rollout = 7
        config.n_search = 64
        player1 = AlphaBetaPlayer()
        player2 = UCTPlayer(config, deterministic=True)
        game = TicTacToeGame()
        p1w, p2w, drw = multi_match(game, player1, player2, n_match=n_match)
        p2nl.append((p2w+drw)/n_match)
    plt.plot(cs, p2nl)
    plt.savefig('p2nl.png')
        
if __name__ == '__main__':
    #####################
    # Modify code below #
    #####################
    
    # set seed to reproduce the result
    # np.random.seed(0)
        
    # game = GobangGame(3, 3) # Tic-Tac-Toe game
    # game = GobangGame(5, 4) # 5x5 Gobang (4 stones in line to win) 
    game = GoGame(7)
    
    parser = argparse.ArgumentParser(description='Run pit.py with configurable parameters.')
    parser.add_argument('--C', type=float, default=1.0, help='UCT constant.')
    parser.add_argument('--n_rollout', type=int, default=7, help='Number of rollouts for MCTS.')
    parser.add_argument('--n_search', type=int, default=64, help='Number of searches for MCTS.')
    args = parser.parse_args()

    # config for MCTS
    config = UCTMCTSConfig()
    # config.C = 0.5
    # config.C = 0.7
    config.C = 1.0
    # config.C = 2.5
    # config.C = 5.0
    config.n_rollout = args.n_rollout
    # config.n_rollout = 7
    # config.n_rollout = 13 # 模拟次数
    config.n_search = args.n_search
    # config.n_search = 64 # 搜索次数
    # config.n_search = 400
    
    # player initialization    
    # player1 = HumanPlayer()
    # player1 = RandomPlayer()
    # player1 = AlphaBetaPlayer()
    # player1 = AlphaBetaHeuristicPlayer()
    player1 = UCTPlayer(config, deterministic=True)
    # player1 = PUCTPlayer(config, deterministic=True)
    
    # player2 = HumanPlayer()
    player2 = RandomPlayer()
    # player2 = AlphaBetaPlayer()
    # player2 = AlphaBetaHeuristicPlayer()
    # player2 = UCTPlayer(config, deterministic=True)
    
    # 设置日志记录
    log_filename = generate_log_filename(player1, player2)
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(message)s')
    
    # single match
    # pit(game, player1, player2)
    
    multi_match(game, player1, player2, n_match=5)
    multi_match(game, player2, player1, n_match=5)
    # multi_match(game, player1, player2, n_match=100)
    
    #####################