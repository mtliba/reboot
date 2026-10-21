"""Re-implementations of published SOC estimators, trained on OUR data with the
same splits, windows and Huber loss as every other baseline.

Each model takes the (B, W, C=7) scaled window and internally keeps only the
raw current / voltage / temperature channels (indices 0, 1, 2), because none of
the original papers use derived channels. Output: SOC in [0, 1], shape (B,).

Architectures follow the papers as closely as their published descriptions
allow (window length differs: ours is W=100 samples):

* ChemaliDNN  — Chemali et al., J. Power Sources 400 (2018): 7 fully connected
  layers, input [V_t, T_t, mean(I), mean(V)] over the trailing window, 32
  neurons per hidden layer.
* ChemaliLSTM — Chemali et al., IEEE TIE 65(8) (2018): single LSTM layer with
  500 hidden units on [V, I, T], fully connected output.
* YangGRU     — Yang et al., Energy 175 (2019): single GRU layer with 1000 units,
  Dense(50, ReLU), Dense(1), as in the authors' released code.
* HannanTransformer — Hannan et al., Sci. Rep. 11 (2021): transformer encoder,
  d_model 128, 16 heads, learned positional encoding, dropout 0.1/0.2, linear
  output (their masked-reconstruction pretraining corresponds to our
  L_mask-only ablation; here the architecture is trained supervised).
* LiCNNBiLSTMAM — Li et al., Sensors (2026): Conv1d(64 filters) -> Bi-LSTM
  (64 per direction) -> Bahdanau (tanh) attention -> FC-ReLU -> FC-sigmoid.
"""
from __future__ import annotations

import torch
import torch.nn as nn

RAW = (0, 1, 2)          # I, V, T in the scaled window
IDX_I, IDX_V, IDX_T = 0, 1, 2


def _raw(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, list(RAW)]                                   # (B, W, 3)


class ChemaliDNN(nn.Module):
    def __init__(self, in_channels: int = 7, hidden: int = 32, n_hidden_layers: int = 5):
        super().__init__()
        layers, d = [], 4
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):                                       # (B, W, C)
        feats = torch.stack([x[:, -1, IDX_V], x[:, -1, IDX_T], x[:, :, IDX_I].mean(1), x[:, :, IDX_V].mean(1)], dim=-1)
        return self.net(feats).squeeze(-1)


class ChemaliLSTM(nn.Module):
    def __init__(self, in_channels: int = 7, hidden: int = 500, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(3, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def forward(self, x):
        h, _ = self.lstm(_raw(x))
        return self.out(h[:, -1]).squeeze(-1)


class YangGRU(nn.Module):
    """Released code (github.com/FYX-master/SOC_estimation_with_GRU-RNN, train.py):
    GRU(1000) -> Dense(50, ReLU) -> Dense(1); Adam 1e-4, MAE loss, batch 72."""
    def __init__(self, in_channels: int = 7, hidden: int = 1000, num_layers: int = 1, dense: int = 50, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(3, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.out = nn.Sequential(nn.Linear(hidden, dense), nn.ReLU(), nn.Linear(dense, 1), nn.Sigmoid())

    def forward(self, x):
        h, _ = self.gru(_raw(x))
        return self.out(h[:, -1]).squeeze(-1)


class HannanTransformer(nn.Module):
    def __init__(self, in_channels: int = 7, d_model: int = 128, nhead: int = 16, num_layers: int = 4,
                 ffn_dim: int = 256, dropout_ffn: float = 0.2, dropout_res: float = 0.1, max_len: int = 512):
        super().__init__()
        self.proj = nn.Linear(3, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model)); nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=ffn_dim, dropout=dropout_res,
                                           batch_first=True, norm_first=False)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout_ffn)
        self.out = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.proj(_raw(x)) + self.pos[:, : x.size(1)]
        h = self.encoder(h)
        return self.out(self.drop(h.mean(1))).squeeze(-1)


class LiCNNBiLSTMAM(nn.Module):
    def __init__(self, in_channels: int = 7, filters: int = 64, kernel_size: int = 3, hidden: int = 64, fc: int = 64):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(3, filters, kernel_size, padding=kernel_size // 2), nn.ReLU())
        self.bilstm = nn.LSTM(filters, hidden, batch_first=True, bidirectional=True)
        self.att_w = nn.Linear(2 * hidden, 2 * hidden)
        self.att_v = nn.Linear(2 * hidden, 1, bias=False)
        self.head = nn.Sequential(nn.Linear(2 * hidden, fc), nn.ReLU(), nn.Linear(fc, 1), nn.Sigmoid())

    def forward(self, x):
        h = self.conv(_raw(x).transpose(1, 2)).transpose(1, 2)   # (B, W, filters)
        h, _ = self.bilstm(h)                                    # (B, W, 2*hidden)
        a = torch.softmax(self.att_v(torch.tanh(self.att_w(h))).squeeze(-1), dim=1)   # (B, W)
        ctx = (a.unsqueeze(-1) * h).sum(1)
        return self.head(ctx).squeeze(-1)


LITERATURE_MODELS = {
    "chemali_dnn": ChemaliDNN, "chemali_lstm": ChemaliLSTM, "yang_gru": YangGRU,
    "hannan_transformer": HannanTransformer, "li_cnnbilstm": LiCNNBiLSTMAM,
}
