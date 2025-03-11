import numpy as np
import math
import logging
import os
import torch

def add_dirichlet_noise(priors, epsilon=0.25, alpha=0.03):
        """
            주어진 prior 확률 분포에 Dirichlet noise를 추가합니다.
            
            매개변수:
            priors: np.array, MCTS에서 사용되는 행동의 prior 확률들.
            epsilon: float, 원래 prior에 대한 가중치 (예: 0.25).
            alpha: float, Dirichlet 분포의 concentration parameter (예: 0.03).
            
            반환:
            noise가 추가된 새로운 prior 확률 분포.
        """
        noise = np.random.dirichlet([alpha] * len(priors))
        new_priors = (1 - epsilon) * priors + epsilon * noise
        return new_priors

def calcUcbOfChildrenFromParent(node: 'Node', mode: str = 'normal'):
    
    for child in node.children:
        if child.visit == 0:
                # child.ucb = np.inf
            continue
        if mode != 'normal':
            # if child.visit == 0:
            #     q = 0
            # else:
            #     q = (child.value / child.visit)
            c_puct = 1
            child.ucb = (child.value / child.visit) + c_puct * child.prior * math.sqrt(math.log(node.visit) / (child.visit+1))
        else:
            
            child.ucb = (child.value / child.visit) + math.sqrt(2 * math.log(node.visit) / child.visit)
    return

def get_probablity_distribution_of_children(node: 'Node', Game):
    """
    Get the probablity distribution of children by the number of visits.
    """
    visit_counts = np.zeros(Game.action_dim)
    for child in node.children:
        visit_counts[Game.get_action_idx(child.prevAction)] = child.visit

    probablity_distribution = visit_counts/np.sum(visit_counts)
    return probablity_distribution

def load_model(model: 'Net', name: str):
    """
    Load the model from the path.
    """
    model_path = os.path.join(os.path.dirname(__file__), f'models/{name}')
    model.load_state_dict(torch.load(model_path))
    return model

def get_game_logger(game_name):
    logger = logging.getLogger(game_name)
    logger.setLevel(logging.DEBUG)

    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    file_path = os.path.join(log_dir, f'{game_name}.log')
    
    if not logger.handlers:
        file_handler = logging.FileHandler(file_path, mode='a') 
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
    logger.debug('\n')
    logger.debug('start')
    return logger