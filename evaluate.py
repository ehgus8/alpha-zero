from ai.nn import Net
import utils
import test
from games import Gomoku, TicTacToe, Connect4
import json

with open('config.json') as f:
    config = json.load(f)

TEST_CFG = config['test']


def safe_int_input(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print('숫자를 입력하세요.')

def test_mode():
    Game = Gomoku
    num_transformer_layers = 2
    best_model = Net(Game.rows, patch_size=5, embed_dim=256, action_dim=Game.action_dim,num_heads=8, channels=Game.feature_dim, depth=num_transformer_layers)
    contender_model = Net(Game.rows, patch_size=5, embed_dim=256, action_dim=Game.action_dim,num_heads=8, channels=Game.feature_dim, depth=num_transformer_layers)
    best_model_num = safe_int_input('select best_model version: ')
    best_model = utils.load_model(best_model, f'model_v{best_model_num}.pth')
    contender_model_num = safe_int_input('select contender_model version: ')
    contender_model = utils.load_model(contender_model, f'model_v{contender_model_num}.pth')
    best_model_mcts_iter = TEST_CFG['mcts_iter']
    contender_model_mcts_iter = TEST_CFG['mcts_iter']
    test.compare(Game, best_model, contender_model, best_model_mcts_iter, contender_model_mcts_iter, TEST_CFG['iterations'], sampling=False,early_stopping=False)

def play_against_agent_mode():
    Game = Gomoku
    human_turn = safe_int_input('select turn first(0) second(1): ')
    num_transformer_layers = 1
    model_version = safe_int_input('select model version: ')
    older_model = Net(Game.rows, patch_size=5, embed_dim=512, action_dim=Game.action_dim,num_heads=8, channels=Game.feature_dim, depth=num_transformer_layers)
    older_model = utils.load_model(older_model, f'model_v{model_version}.pth')
    test.play_against_agent(Game, older_model, TEST_CFG['mcts_iter'], human_turn) 