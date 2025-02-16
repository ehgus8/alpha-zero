import numpy as np
import math
import logging
import os
import torch


def calcUcbOfChildrenFromParent(node: 'Node', mode: str = 'normal'):
    
    for child in node.children:
        if child.visit == 0:
            child.ucb = np.inf
            continue
        if mode != 'normal':
            child.ucb = (child.value / child.visit) + child.prior * math.sqrt(math.log(node.visit) / child.visit)
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
    logger.debug('start')
    return logger