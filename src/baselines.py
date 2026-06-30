import torch
import torch.nn as nn
import math

class TransformerModel(nn.Module):
    """
    PRO-88: Parameter-matched Transformer baseline.
    Uses standard self-attention (with causal masking) and a continuous projection head.
    """
    def __init__(self, state_dim=3, action_dim=1, n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, n_embd)
        
        # Positional embedding for Transformer
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, n_embd))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.output_head = nn.Linear(n_embd, state_dim)
        
    def forward(self, states, actions, targets=None):
        B, T, _ = states.size()
        x_input = torch.cat([states, actions], dim=-1) # B, T, state_dim + action_dim
        x = self.input_proj(x_input)
        
        # Add positional embedding
        x = x + self.pos_encoder[:, :T, :]
        
        # Create causal mask for transformer
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=states.device)
        
        out = self.transformer(x, mask=mask, is_causal=True)
        predictions = self.output_head(out)
        
        loss = None
        if targets is not None:
            loss = nn.functional.mse_loss(predictions, targets)
            
        return predictions, loss

class GRUModel(nn.Module):
    """
    PRO-89: GRU floor baseline (has recurrent memory but no attention/plasticity).
    """
    def __init__(self, state_dim=3, action_dim=1, n_embd=128, n_layer=2):
        super().__init__()
        self.input_proj = nn.Linear(state_dim + action_dim, n_embd)
        self.gru = nn.GRU(
            input_size=n_embd,
            hidden_size=n_embd,
            num_layers=n_layer,
            batch_first=True
        )
        self.output_head = nn.Linear(n_embd, state_dim)
        
    def forward(self, states, actions, targets=None):
        x_input = torch.cat([states, actions], dim=-1)
        x = self.input_proj(x_input)
        
        out, _ = self.gru(x)
        predictions = self.output_head(out)
        
        loss = None
        if targets is not None:
            loss = nn.functional.mse_loss(predictions, targets)
            
        return predictions, loss

class MLPModel(nn.Module):
    """
    PRO-89: MLP floor baseline (memoryless, predicts s_{t+1} only from s_t and a_t).
    """
    def __init__(self, state_dim=3, action_dim=1, n_embd=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, state_dim)
        )
        
    def forward(self, states, actions, targets=None):
        x_input = torch.cat([states, actions], dim=-1)
        predictions = self.net(x_input)
        
        loss = None
        if targets is not None:
            loss = nn.functional.mse_loss(predictions, targets)
            
        return predictions, loss

def get_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
