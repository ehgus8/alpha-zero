import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, channels):
        super().__init__()
        # patch_size는 2k+1, stride를 1로, padding = patch_size // 2로 설정
        self.patch_embed = nn.Conv2d(
            channels, embed_dim, kernel_size=patch_size, stride=1, padding=patch_size // 2
        )

    def forward(self, x):
        # 입력 x의 shape: (B, channels, img_size, img_size)
        # 출력 shape: (B, embed_dim, img_size, img_size)
        x = self.patch_embed(x)
        # flatten spatial dimensions: (B, embed_dim, img_size*img_size) -> (B, img_size*img_size, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViTEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, channels):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, embed_dim, channels)
        # stride가 1로 설정되어 출력 feature map의 크기는 원본과 동일함
        num_patches = img_size * img_size
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
    
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embedding
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Self-Attention 단계
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output

        # MLP 단계
        x_norm = self.norm2(x)
        x = x + self.mlp(x_norm)
        return x

class Net(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, action_dim, num_heads, depth, channels, dropout=0.0):
        super().__init__()
        self.embedding = ViTEmbedding(img_size, patch_size, embed_dim, channels)
        self.blocks = nn.ModuleList([
            AttentionBlock(embed_dim, num_heads, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.policy_head = nn.Linear(embed_dim, action_dim)
        self.value_head = nn.Linear(embed_dim, 1)
        
    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        # cls_token을 사용해 최종 예측 (정책과 가치)
        cls_token_final = x[:, 0]
        policy_logits = self.policy_head(cls_token_final)
        values = torch.tanh(self.value_head(cls_token_final))
        return policy_logits, values
