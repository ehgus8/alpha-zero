import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

#CrossAttentionNetVariableCandidates
class Net(nn.Module):
    def __init__(self, state_dim, action_dim, num_transformer_layers=4, dropout=0.0):
        """
        Args:
            num_transformer_layers: Transformer 블록의 수
            dropout: dropout 비율
        """
        super(Net, self).__init__()
        
        embed_dim = 128 # f(s)와 g(s') 임베딩 차원
        # --- f(s) branch: 현재 상태 s를 임베딩 ---
        self.representation = RepresentationNet(state_dim, 3, embed_dim)
        
        # Dynamic 네트워크: g(s, a)
        self.dynamic = DynamicsNet(hidden_dim=embed_dim, action_dim=action_dim)
        
        # --- Transformer 블록 (cross-attention) ---
        self.transformer_layers = nn.ModuleList([
            CrossAttentionTransformerBlock(embed_dim=embed_dim, num_heads=8, dropout=dropout)
            for _ in range(num_transformer_layers)
        ])
        
        self.policy_fc = nn.Linear(embed_dim, state_dim * 2)
        self.value_fc = nn.Linear(embed_dim, state_dim * 1)
        # --- 최종 예측 Head ---
        self.policy_head = nn.Linear(state_dim * 2, action_dim)
        self.value_head = nn.Linear(state_dim * 1, 1)
    
    def forward(self, s, return_attention=False):
        """
        Args:
            s: 현재 상태 (tensor), shape: (B, 3, H, W)
            s_candidates_list: 길이 B의 리스트, 각 원소는 (N_i, 3, H, W) 텐서 
                               (여기서 N_i는 샘플마다 달라질 수 있음)
        Returns:
            policy: policy logits, shape: (B, 9)
            value: value 예측, shape: (B, 1) (출력 범위: -1 ~ 1)
        """
        B = s.size(0)
        
        # --- f(s) 계산 ---
        f = self.representation(s)
        
        # 2. 가능한 모든 행동에 대해 dynamic 네트워크로 후보 은닉 상태 g(s, a)를 계산
        action_dim = self.dynamic.action_dim  # 예: 9
        device = f.device
        # 모든 행동에 대한 one-hot: (action_dim, action_dim)
        candidate_actions = torch.eye(action_dim, device=device)  
        # 이를 배치마다 사용: (B, action_dim, action_dim)
        candidate_actions = candidate_actions.unsqueeze(0).expand(B, -1, -1)
        # f(s)를 각 후보에 대해 반복: (B, action_dim, hidden_dim)
        f_repeated = f.unsqueeze(1).expand(-1, action_dim, -1)
        # flatten하여 dynamic 네트워크에 넣기
        f_flat = f_repeated.contiguous().view(B * action_dim, -1)              # (B * action_dim, hidden_dim)
        candidate_actions_flat = candidate_actions.contiguous().view(B * action_dim, -1)  # (B * action_dim, action_dim)
        # 동적 모델로 후보 은닉 상태 계산: g(s, a)
        g_flat = self.dynamic(f_flat, candidate_actions_flat)  # (B * action_dim, hidden_dim)
        # 다시 (B, action_dim, hidden_dim) 형태로 reshape
        g_candidates = g_flat.view(B, action_dim, -1)
        
        # 3. Cross-Attention: f(s) (query)와 g_candidates (key, value)를 결합
        query = f.unsqueeze(0)                   # (1, B, hidden_dim)
        key   = g_candidates.transpose(0, 1)       # (action_dim, B, hidden_dim)
        value = key                              # (action_dim, B, hidden_dim)

        attn_maps = []  # 어텐션 맵들을 저장할 리스트
        for layer in self.transformer_layers:
            if return_attention:
                query, attn_weights = layer(query, key, value, key_padding_mask=None, return_attention=True)
                attn_maps.append(attn_weights)
            else:
                query = layer(query, key, value, key_padding_mask=None)
        attn_output = query.squeeze(0)           # (B, hidden_dim)
        
        # 4. 최종 예측: 정책과 가치
        policy_fc = self.policy_fc(attn_output)
        value_fc = self.value_fc(attn_output)

        policy = self.policy_head(policy_fc)   # (B, action_dim)
        value_out = torch.tanh(self.value_head(value_fc))  # (B, 1), -1 ~ 1
        
        if return_attention:
            return policy, value_out, attn_maps
        else:
            return policy, value_out

    
class RepresentationNet(nn.Module):
    """
    관측 -> 은닉 상태 변환, 맨 처음에 한 번만 사용 됨.
    """
    
    def __init__(self, state_dim, in_channels=3, hidden_dim=256):
        super(RepresentationNet, self).__init__()
        # --- f(s) branch: 현재 상태 s를 임베딩 ---
        self.f_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        # self.f_bn1   = nn.BatchNorm2d(hidden_dim)
        # self.f_res = ResidualBlock(channels=channel)
        self.f_fc1   = nn.Linear(in_channels  * state_dim, hidden_dim)
    
    def forward(self, s):
        """
        Args:
            s: 현재 상태 (tensor), shape: (B, 3, H, W)
        Returns:
            현재 상태 임베딩 벡터
        """
        B = s.size(0)
        
        # --- f(s) 계산 ---
        f = F.relu(self.f_conv1(s))
        # f = F.relu(self.f_bn1(f))
        # f = self.f_res(f)
        f = f.view(B, -1)  # flatten
        f = self.f_fc1(f) # 결과: (B, embed_dim)

        return f

###############################################################################
# 2. Dynamics Network
###############################################################################
class DynamicsNet(nn.Module):
    """
    은닉 상태 (임베딩 벡터)와 행동 -> 다음 은닉 상태
    (MuZero에서 사용하는 dynamic 모델과 유사한 역할)
    """
    def __init__(self, hidden_dim, action_dim):
        """
        Args:
            hidden_dim: 은닉 상태의 크기
            action_dim: 가능한 행동의 개수
        """
        super(DynamicsNet, self).__init__()
        self.action_dim = action_dim  # 나중에 후보 행동 생성에 사용
        self.fc_state1 = nn.Linear(hidden_dim + action_dim, 128)
        self.fc_state2 = nn.Linear(128, hidden_dim)
    
    def forward(self, hidden_state, action_onehot):
        """
        Args:
            hidden_state: [B, hidden_dim] 또는 [B * action_dim, hidden_dim]
            action_onehot: [B, action_dim] 또는 [B * action_dim, action_dim]
        Returns:
            next_hidden_state: 같은 shape로 반환
        """
        x = torch.cat([hidden_state, action_onehot], dim=1)
        hs = F.relu(self.fc_state1(x))
        next_hidden_state = self.fc_state2(hs)
        return next_hidden_state
    


# --- Transformer Block (Cross-Attention Block) 정의 ---
class CrossAttentionTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, ff_hidden_dim=None):
        """
        Args:
            embed_dim: 임베딩 차원
            num_heads: Multi-Head Attention의 헤드 수
            dropout: dropout 비율
            ff_hidden_dim: Feed-forward hidden layer 차원 (기본값: embed_dim * 4)
        """
        super(CrossAttentionTransformerBlock, self).__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = embed_dim * 4
        
        # Multi-Head Attention 층 (여기서는 query와 key/value 간 cross attention)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # Residual Connection 후 Layer Normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, query, key, value, key_padding_mask=None, return_attention=False):
        """
        Args:
            query: (seq_len, batch, embed_dim) – 여기서는 현재 상태 임베딩 (길이 1)
            key: (seq_len_k, batch, embed_dim) – 후보 상태 임베딩 (패딩된 시퀀스)
            value: (seq_len_k, batch, embed_dim) – 후보 상태 임베딩
            key_padding_mask: (batch, seq_len_k) – 패딩된 위치 True 처리
        Returns:
            업데이트된 query: (seq_len, batch, embed_dim)
        """
        # Cross-Attention
        attn_output, attn_weights = self.mha(
            query, key, value, key_padding_mask=key_padding_mask, need_weights=True
        )
        query = query + self.dropout1(attn_output)
        query = self.norm1(query)
        
        # Feed-Forward Network
        ff_output = self.ff(query)
        query = query + self.dropout2(ff_output)
        query = self.norm2(query)
        if return_attention:
            return query, attn_weights
        else:
            return query

# --- Residual Block 정의 ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        """
        Args:
            channels: 입력과 출력 채널 수 (skip connection 때문에 동일해야 함)
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
