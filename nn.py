import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class Net(nn.Module):
    def __init__(self, state_dim, action_dim, dropout=0.0):
        """
        Args:
            num_transformer_layers: Transformer 블록의 수
            dropout: dropout 비율
        """
        super(Net, self).__init__()
        
        channels = 256
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.res_block1 = ResidualBlock(channels)
        
        self.policy_fc = nn.Linear(state_dim * channels, state_dim * 2)
        self.value_fc = nn.Linear(state_dim * channels, state_dim * 1)
        # --- Policy Head ---
        self.policy_head = nn.Linear(state_dim * 2, action_dim)
        # --- Value Head ---
        self.value_head = nn.Linear(state_dim * 1, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.res_block1(x)
        x = x.view(x.size(0), -1)

        policy_x = F.relu(self.policy_fc(x))
        value_x = F.relu(self.value_fc(x))

        policy_logits = self.policy_head(policy_x)
        value_out = torch.tanh(self.value_head(value_x))
        
        return policy_logits, value_out

# --- Residual Block 정의 ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Args:
            channels: input and output channels same for skip connection
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity   # skip connection
        out = self.relu(out)
        return out
