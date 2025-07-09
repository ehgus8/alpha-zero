import cProfile, pstats
from games.gomoku import Gomoku
from ai.nn import Net
import torch


def main():
    """Run a single self-play game for profiling."""
    Game = Gomoku
    # 간단한 작은 모델(깊이 1) 사용
    model = Net(Game.rows, patch_size=5, embed_dim=128, action_dim=Game.action_dim,
                num_heads=4, depth=1, channels=Game.feature_dim)
    game = Game()
    with torch.no_grad():
        game.self_play(model, mcts_iter=50, display=False)


if __name__ == "__main__":
    profile_path = "profile.out"
    cProfile.run("main()", profile_path)
    stats = pstats.Stats(profile_path)
    stats.sort_stats("cumtime").print_stats(30) 