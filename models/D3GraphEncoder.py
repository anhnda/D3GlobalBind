import torch
from torch import nn
import numpy as np


class D3PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model=512, device=None):
        super(D3PositionalEncoder, self).__init__()

        self.d_model = d_model
        print("Dmodel size: ", self.d_model)
        self.device = device
        self.zeros = torch.zeros(self.d_model).to(device)
        self.pad = nn.ConstantPad1d((0, 1), 0)


    def forward(self, x):
        assert x.shape[1] == 3
        nSeg = self.d_model // 3
        scales = torch.linspace(-1, 1, nSeg)
        xs = [x * sc for sc in scales]
        xs = torch.cat(xs, dim=1)[:, :self.d_model]
        # Padding 0
        xs = self.pad(xs.t()).t()
        return xs


def generate_square_subsequent_mask(sz: int):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

class D3GraphEncoder(torch.nn.Module):
    def __init__(self, d_model=512, n_layer=6, n_head=6, d_ff=2048, dropout_rate=0.1, device=None):
        super(D3GraphEncoder, self).__init__()
        self.d_model = d_model
        self.n_layer = n_layer
        self.dropout_rate = dropout_rate
        self.device = device
        self.pad = nn.ConstantPad1d((0, 1), 0)
        self.pe = D3PositionalEncoder(d_model, device).to(device)
        self.dropout = nn.Dropout(dropout_rate)
        transformerEncoderLayer = nn.TransformerEncoderLayer(d_model, n_head, d_ff, dropout_rate).to(device)
        self.transformerEncoder = nn.TransformerEncoder(transformerEncoderLayer, num_layers=n_layer).to(device)

        self.globalFeatureLayer = nn.Linear(d_model, d_model).to(device)
        self.globalBindingCentroid = nn.Linear(d_model, 3).to(device)
        self.globalBindingDirection = nn.Linear(d_model, 3).to(device)

    def forward(self, features, coords):
        features = self.pad(features.t()).t()
        coords = self.pe(coords)
        features = features + coords
        features = torch.unsqueeze(features, 1).to(self.device)
        features = self.dropout(features)
        src_mask = generate_square_subsequent_mask(features.shape[0]).to(self.device)
        out_all = self.transformerEncoder(features, src_mask)
        globalInfoVector = out_all.squeeze()[-1, :]
        globalBindingFeature = self.globalFeatureLayer(globalInfoVector)
        globalBindingCentroid = self.globalBindingCentroid(globalInfoVector)
        globalBindingDirection = self.globalBindingDirection(globalInfoVector)
        return globalBindingFeature, globalBindingCentroid, globalBindingDirection
