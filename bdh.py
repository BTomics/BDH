# Copyright 2025 Pathway Technology, Inc.

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn


@dataclasses.dataclass
class BDHConfig:
    n_layer: int = 6
    n_embd: int = 256
    dropout: float = 0.1
    n_head: int = 4
    mlp_internal_dim_multiplier: int = 128
    state_dim: int = 3
    action_dim: int = 1


def get_freqs(n, theta, dtype):
    def quantize(t, q=2):
        return (t / q).floor() * q

    return (
        1.0
        / (theta ** (quantize(torch.arange(0, n, 1, dtype=dtype)) / n))
        / (2 * math.pi)
    )


class Attention(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        self.freqs = torch.nn.Buffer(
            get_freqs(N, theta=2**16, dtype=torch.float32).view(1, 1, 1, N)
        )

    @staticmethod
    def phases_cos_sin(phases):
        phases = (phases % 1) * (2 * math.pi)
        phases_cos = torch.cos(phases)
        phases_sin = torch.sin(phases)
        return phases_cos, phases_sin

    @staticmethod
    def rope(phases, v):
        v_rot = torch.stack((-v[..., 1::2], v[..., ::2]), dim=-1).view(*v.size())
        phases_cos, phases_sin = Attention.phases_cos_sin(phases)
        return (v * phases_cos).to(v.dtype) + (v_rot * phases_sin).to(v.dtype)

    def forward(self, Q, K, V, freeze_timestep=None):
        assert self.freqs.dtype == torch.float32
        assert K is Q
        _, _, T, _ = Q.size()

        r_phases = (
            torch.arange(
                0,
                T,
                device=self.freqs.device,
                dtype=self.freqs.dtype,
            ).view(1, 1, -1, 1)
        ) * self.freqs
        QR = self.rope(r_phases, Q)
        KR = QR

        # Current attention
        scores = (QR @ KR.mT)
        
        # PRO-87: Causal mask with optional freezing of Hebbian updates (fast-weights)
        mask = torch.tril(torch.ones((T, T), device=Q.device), diagonal=-1)
        if freeze_timestep is not None:
            freeze_mask = torch.zeros((T, T), device=Q.device)
            freeze_mask[:, :freeze_timestep] = 1.0
            mask = mask * freeze_mask
            
        scores = scores * mask
        return scores @ V


class BDH(nn.Module):
    def __init__(self, config: BDHConfig):
        super().__init__()
        self.config = config
        nh = config.n_head
        D = config.n_embd
        N = config.mlp_internal_dim_multiplier * D // nh
        
        self.decoder = nn.Parameter(torch.zeros((nh * N, D)).normal_(std=0.02))
        self.encoder = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        self.attn = Attention(config)

        self.ln = nn.LayerNorm(D, elementwise_affine=False, bias=False)
        
        # PRO-81: Continuous input projection (state_dim + action_dim -> n_embd)
        self.input_proj = nn.Linear(config.state_dim + config.action_dim, D)
        self.drop = nn.Dropout(config.dropout)
        self.encoder_v = nn.Parameter(torch.zeros((nh, D, N)).normal_(std=0.02))

        # PRO-82: Linear next-state regression head (n_embd -> state_dim)
        self.output_head = nn.Linear(D, config.state_dim)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, states, actions, targets=None, freeze_timestep=None):
        C = self.config

        B, T, _ = states.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        # Concatenate state and action along the feature dimension
        x_input = torch.cat([states, actions], dim=-1) # B, T, state_dim + action_dim
        
        # Project to embedding dimension
        x = self.input_proj(x_input).unsqueeze(1)  # B, 1, T, D

        # LayerNorm
        x = self.ln(x)  # B, 1, T, D

        for level in range(C.n_layer):
            x_latent = x @ self.encoder

            x_sparse = F.relu(x_latent)  # B, nh, T, N

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                freeze_timestep=freeze_timestep
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )  # B, 1, T, D
            y = self.ln(yMLP)
            x = self.ln(x + y)

        # Reshape to (B, T, D) and project to state_dim
        predictions = self.output_head(x.view(B, T, D))
        
        loss = None
        if targets is not None:
            # PRO-82: Compute MSE loss for next-state regression
            loss = F.mse_loss(predictions, targets)

        return predictions, loss

    def forward_with_activations(self, states, actions, freeze_timestep=None):
        C = self.config
        B, T, _ = states.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x_input = torch.cat([states, actions], dim=-1)
        x = self.input_proj(x_input).unsqueeze(1)
        x = self.ln(x)

        all_activations = []

        for level in range(C.n_layer):
            x_latent = x @ self.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N
            all_activations.append(x_sparse)

            yKV = self.attn(
                Q=x_sparse,
                K=x_sparse,
                V=x,
                freeze_timestep=freeze_timestep
            )
            yKV = self.ln(yKV)

            y_latent = yKV @ self.encoder_v
            y_sparse = F.relu(y_latent)
            xy_sparse = x_sparse * y_sparse
            xy_sparse = self.drop(xy_sparse)

            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.decoder
            )
            y = self.ln(yMLP)
            x = self.ln(x + y)

        predictions = self.output_head(x.view(B, T, D))
        return predictions, all_activations


    @torch.no_grad()
    def generate_rollout(self, initial_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Autoregressively predict the sequence of next states.
        initial_state: (B, 1, state_dim)
        actions: (B, T, action_dim)
        Returns: predicted_states (B, T, state_dim)
        """
        B, T, _ = actions.size()
        predicted_states = []
        
        for t in range(T):
            if len(predicted_states) > 0:
                # Concatenate initial state and all predicted states up to now
                history_states = torch.cat([initial_state] + predicted_states, dim=1) # (B, t+1, state_dim)
                history_actions = actions[:, :t+1, :] # (B, t+1, action_dim)
                preds, _ = self(history_states, history_actions)
                next_state_pred = preds[:, -1:, :]
            else:
                preds, _ = self(initial_state, actions[:, :1, :])
                next_state_pred = preds[:, -1:, :]
                
            predicted_states.append(next_state_pred)
            
        return torch.cat(predicted_states, dim=1)
