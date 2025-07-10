import numpy as np
import torch
import json
from flask import Flask, render_template, request, jsonify

from ai.nn import Net
from games.gomoku import Gomoku
from ai import Node, MCTS
import utils

app = Flask(__name__)

# --- Load Config ---
with open('config.json') as f:
    config = json.load(f)
TRAIN_CFG = config['train']
TEST_CFG = config['test']
# --------------------

# --- Global variables ---
Game = Gomoku
model = None
game_state = None
human_player_idx = 0 # 0 for black, 1 for white
# ------------------------


def load_model():
    """Load the latest trained model."""
    global model
    model = Net(Game.rows, 
                patch_size=TRAIN_CFG['patch_size'], 
                embed_dim=TRAIN_CFG['embed_dim'], 
                action_dim=Game.action_dim,
                num_heads=TRAIN_CFG['num_heads'], 
                depth=TRAIN_CFG['num_transformer_layers'], 
                channels=Game.feature_dim)
    model, _ = utils.load_latest_model(model, Game)
    model.eval()


@app.route('/')
def index():
    """Render the main game page."""
    return render_template('index.html')


@app.route('/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    global game_state, human_player_idx

    data = request.get_json()
    player_color = data.get('player_color', 'black')
    human_player_idx = 0 if player_color == 'black' else 1
    
    game_state = Game()

    if human_player_idx == 1:
        # AI (Black) makes the first move
        ai_player = 0
        center_action = (Game.rows // 2, Game.cols // 2)
        game_state.make_move(game_state.board, ai_player, center_action)
        game_state.move_count += 1

    return jsonify({
        'board': game_state.board.tolist(),
        'rows': Game.rows,
        'cols': Game.cols
    })


@app.route('/move', methods=['POST'])
def move():
    """Handle a player's move and get the AI's response."""
    global game_state, human_player_idx
    if game_state is None:
        return jsonify({'error': 'Game not started'}), 400

    data = request.get_json()
    row, col = data['row'], data['col']
    player_action = (row, col)

    # Player's move
    if game_state.board[0, row, col] != 0 or game_state.board[1, row, col] != 0:
        return jsonify({'error': 'Invalid move'}), 400
    
    game_state.make_move(game_state.board, human_player_idx, player_action)
    game_state.move_count += 1
    winner = game_state.check_winner(game_state.board, human_player_idx, player_action)

    if winner != -1:
        return jsonify({'board': game_state.board.tolist(), 'winner': winner, 'ai_move': None})

    # AI's move
    ai_player = 1 - human_player_idx
    root = Node(None, None, ai_player, game_state.move_count)
    MCTS.mcts(model, game_state.board, root, Game, mcts_iterations=TEST_CFG['mcts_iter'], dirichlet=False)
    
    chosen_child = root.max_visit_child()
    ai_action = chosen_child.prevAction
    game_state.make_move(game_state.board, ai_player, ai_action)
    game_state.move_count += 1
    winner = game_state.check_winner(game_state.board, ai_player, ai_action)
    
    return jsonify({
        'board': game_state.board.tolist(),
        'winner': winner,
        'ai_move': ai_action
    })

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0') 