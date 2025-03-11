import game
import numpy as np
import torch
import utils

class MCTS:
    @staticmethod
    def mcts(model, board, root, Game: game.Game, mcts_iterations, dirichlet = True):
        """
        Perform Monte Carlo Tree Search.
        Select -> Expand -> Simulate -> Backup

        Args:
            board (np.ndarray): The current board state.
            current_player (int): The current player.
            model (nn.Module): The neural network model.
        """
        for _ in range(mcts_iterations):
            node = root
            trace = [root]
            while node.children:
                node = node.select('network' if model else 'normal')
                trace.append(node)

                Game.make_move(board, 1 - node.currentPlayer, node.prevAction)
            if node.prevAction:
                winner = Game.check_winner(board, 1 - node.currentPlayer, node.prevAction)
                if winner != -1:
                    result = 1 if winner == 1 - node.currentPlayer else -1
                    node.backup(trace, result, board, Game)
                    continue
                elif node.move_count == Game.state_dim:
                    result = 0
                    node.backup(trace, result, board, Game)
                    continue
            valid_moves = Game.get_valid_moves(board)
            if model:
                # forward
                policy_logits, value = model(
                    torch.from_numpy(board).unsqueeze(0)
                    # torch.tensor(board, dtype=torch.float32).unsqueeze(0)
                )
                policy_logits = policy_logits.squeeze(0).detach().numpy()
                policy_softmax = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
                if (not node.prevAction) and dirichlet:
                    policy_softmax = utils.add_dirichlet_noise(policy_softmax)
                # Expand
                node.expand(valid_moves, policy_softmax, Game)
                result = -value.item()
            else:
                node.expand(valid_moves, None, Game)
                result = MCTS.simulate(Game, board, node)
            node.backup(trace, result, board, Game)

    @staticmethod
    def simulate(Game: game.Game, board, node):
        """
        Simulation (rollout) step of MCTS.
        """
        sim_board = board.copy()
        current_player = node.currentPlayer
        move_count = node.move_count
        winner = -1
        while winner == -1 and move_count < Game.state_dim:
            valid_moves = Game.get_valid_moves(sim_board)

            action = valid_moves[np.random.randint(len(valid_moves))]

            current_player = Game.make_move(sim_board, current_player, action)
            move_count += 1

            winner = Game.check_winner(sim_board, 1 - current_player, action)

        if winner != -1:
            return 1 if winner == (1 - node.currentPlayer) else -1
        return 0