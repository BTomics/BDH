import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import bdh

class BDHPolicy(nn.Module):
    """
    PRO-96: Continuous BDH policy for behavior cloning.
    Accepts sequences of states and outputs sequences of actions.
    """
    def __init__(self, config: bdh.BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        self.attn = bdh.Attention(config)
        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # Policy mode: input is state, output is action
        self.input_proj = nn.Linear(config.state_dim, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))
        
        # Output action projection (clamped to action limits [-2.0, 2.0] at eval time)
        self.output_head = nn.Linear(D, config.action_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, states, targets=None, freeze_timestep=None):
        C = self.config
        B, T, _ = states.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Project state to embedding dimension
        x = self.input_proj(states).unsqueeze(1)  # B, 1, T, D
        x = self.ln(x)

        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)

            yKV = self.attn(
                Q=x_sparse, K=x_sparse, V=x,
                freeze_timestep=freeze_timestep
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

        # Output continuous action
        predictions = 2.0 * torch.tanh(self.output_head(x.view(B, T, D)))
        
        loss = None
        if targets is not None:
            loss = F.mse_loss(predictions, targets)

        return predictions, loss

    def forward_with_activations(self, states, freeze_timestep=None):
        C = self.config
        B, T, _ = states.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.input_proj(states).unsqueeze(1)
        x = self.ln(x)

        all_activations = []

        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N
            all_activations.append(x_sparse)

            yKV = self.attn(
                Q=x_sparse, K=x_sparse, V=x,
                freeze_timestep=freeze_timestep
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            yMLP = xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            y = self.ln(yMLP)
            x = self.ln(x + y)

        predictions = 2.0 * torch.tanh(self.output_head(x.view(B, T, D)))
        return predictions, all_activations

class TransformerPolicy(nn.Module):

    """
    Transformer policy baseline for behavior cloning.
    """
    def __init__(self, state_dim=3, action_dim=1, n_embd=96, n_head=4, n_layer=3, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, n_embd)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 1000, n_embd))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd, nhead=n_head, dim_feedforward=n_embd * 4,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.output_head = nn.Linear(n_embd, action_dim)
        
    def forward(self, states, targets=None):
        B, T, _ = states.size()
        x = self.input_proj(states) + self.pos_encoder[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=states.device)
        out = self.transformer(x, mask=mask, is_causal=True)
        predictions = 2.0 * torch.tanh(self.output_head(out))
        
        loss = None
        if targets is not None:
            loss = F.mse_loss(predictions, targets)
        return predictions, loss

class GRUPolicy(nn.Module):
    """
    GRU policy baseline for behavior cloning.
    """
    def __init__(self, state_dim=3, action_dim=1, n_embd=128, n_layer=2):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, n_embd)
        self.gru = nn.GRU(input_size=n_embd, hidden_size=n_embd, num_layers=n_layer, batch_first=True)
        self.output_head = nn.Linear(n_embd, action_dim)
        
    def forward(self, states, targets=None):
        x = self.input_proj(states)
        out, _ = self.gru(x)
        predictions = 2.0 * torch.tanh(self.output_head(out))
        
        loss = None
        if targets is not None:
            loss = F.mse_loss(predictions, targets)
        return predictions, loss
