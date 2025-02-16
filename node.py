import numpy as np
import utils
import game

class Node:
    def __init__(self, parent: 'Node', prevAction: tuple, currentPlayer: int, move_count: int, prior: float = 0.0):
        """
        Initialize a new node.

        Args:
            parent (Node): The parent node.
            prevAction: The action that led to this node.
            currentPlayer: The player who is currently making a move.
            move_count: The number of moves made so far. it is different from the depth of the tree.

        """
        self.parent = parent
        self.visit = 0
        self.value = 0 # The value of the node that means the value of edge between parent and this node.
        self.ucb = np.inf
        self.prior = prior
        self.move_count = move_count
        self.prevAction = prevAction
        self.currentPlayer = currentPlayer
        self.children = []

    def select(self):
        """
        Select the child node with the highest UCB value.
        But Actually, this function sorts the children by UCB value in ascending order.
        """

        utils.calcUcbOfChildrenFromParent(self)
        self.children.sort(key=lambda x: x.ucb, reverse=True)

    def expand(self, valid_moves: list[tuple[int, int]], policy_distribution, Game):
        """
        Expand the node by adding all possible valid moves as children.
        """

        for move in valid_moves: 
            if policy_distribution is not None:
                self.children.append(Node(self, move, 1 - self.currentPlayer, self.move_count + 1, policy_distribution[Game.get_action_idx(move)]))
            else:
                self.children.append(Node(self, move, 1 - self.currentPlayer, self.move_count + 1))
        

    def backup(self, trace: list['Node'], value: float, board: np.ndarray, Game):
        """
        Backpropagate the value of the simulation result
        by updating the value and visit count of each node in the trace.
        The value is negated at each level of the tree.
        """

        for node in trace[::-1]:
            node.visit += 1
            node.value += value
            value *= -1
            if node.parent:
                Game.undo_move(board, node.currentPlayer, node.prevAction)
    
    def max_visit_child(self):
        """
        Return the child with the highest visit count.
        """

        return max(self.children, key=lambda x: x.visit)

    def sample_child(self, Game):
        """
        Sample a child node according to the visit count.
        """

        prob_dist = utils.get_probablity_distribution_of_children(self, Game)
        children = [None] * Game.action_dim
        for child in self.children:
            children[Game.get_action_idx(child.prevAction)] = child
        
        return np.random.choice(children, p=prob_dist)

    def to_string(self, Game: game.Game):
        prob_dist = utils.get_probablity_distribution_of_children(self.parent, Game)
        return f"Node: {self.prevAction}, Value: {self.value}, Visit: {self.visit}, P(Visit): {prob_dist[Game.get_action_idx(self.prevAction)]}, UCB: {self.ucb}"
    