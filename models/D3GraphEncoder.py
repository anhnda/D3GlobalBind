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

    def forward(self, x, n_padding=0):
        assert x.shape[1] == 3
        nSeg = self.d_model // 3
        scales = torch.linspace(-1, 1, nSeg)
        xs = [x * sc for sc in scales]
        xs = torch.cat(xs, dim=1)[:, :self.d_model]
        # Padding 0
        if n_padding == 0:
            pad = self.pad
        else:
            pad = nn.ConstantPad1d((0, n_padding + 1), 0)
        xs = pad(xs.t()).t()

        return xs


class D3GraphEncoder(torch.nn.Module):
    def __init__(self, d_model=512, n_d3graph_layer=5, n_d3graph_head=5, d3_ff_size=2048, d3_graph_dropout_rate=0.1,
                 device=None):
        super(D3GraphEncoder, self).__init__()
        self.d_model = d_model
        self.n_layer = n_d3graph_layer
        self.dropout_rate = d3_graph_dropout_rate
        self.device = device
        self.pad = nn.ConstantPad1d((0, 1), 0)
        self.pe = D3PositionalEncoder(d_model, device).to(device)
        self.dropout = nn.Dropout(d3_graph_dropout_rate)
        transformerEncoderLayer = nn.TransformerEncoderLayer(d_model, n_d3graph_head, d3_ff_size,
                                                             d3_graph_dropout_rate).to(device)
        self.transformerEncoder = nn.TransformerEncoder(transformerEncoderLayer, num_layers=n_d3graph_layer).to(device)

        self.globalFeatureLayer = nn.Linear(d_model, d_model).to(device)
        self.globalBindingCentroid = nn.Linear(d_model, 3).to(device)
        self.globalBindingDirection = nn.Linear(d_model, 3).to(device)

    def forward(self, features, coords):
        if type(features) == list and len(features[0].shape) == 2:
            batch_size = len(features)
            seq_lengths = [feature.shape[0] + 1 for feature in features]
            max_seq_length = max(seq_lengths)
            src_mask = torch.zeros((max_seq_length, max_seq_length), device=self.device).type(torch.bool)

            pe_features = []
            src_key_padding_mask = torch.zeros((batch_size, max_seq_length), device=self.device).type(torch.bool)
            for i, feature in enumerate(features):
                padded_size = max_seq_length - feature.shape[0] - 1
                padded_nn = nn.ConstantPad1d((0, padded_size + 1), 0)
                padded_feature = padded_nn(feature.t()).t()
                padded_pos = self.pe(coords[i], padded_size)
                pe_features.append(padded_feature + padded_pos)
                src_key_padding_mask[i, feature.shape[0] + 1:] = True
            pe_features = torch.stack(pe_features).transpose(1, 0).to(self.device)

            out_all = self.transformerEncoder(pe_features, src_mask, src_key_padding_mask).transpose(1, 0)
            out_selected = []
            for i, out in enumerate(out_all):
                out_s = out[features[i].shape[0]]
                out_selected.append(out_s)
            out_selected = torch.stack(out_selected).to(self.device)
            globalBindingFeatures = self.globalFeatureLayer(out_selected)
            globalBindingDirections = self.globalBindingDirection(out_selected)
            globalBindingCentroid = self.globalBindingCentroid(out_selected)
            return globalBindingFeatures, globalBindingDirections, globalBindingCentroid
        else:
            features = self.pad(features.t()).t()
            coords = self.pe(coords)
            features = features + coords
            features = torch.unsqueeze(features, 1).to(self.device)
            features = self.dropout(features)
            src_mask = torch.zeros((features.shape[0], features.shape[0]), device=self.device).type(torch.bool)

            out_all = self.transformerEncoder(features, src_mask)
            globalInfoVector = out_all.squeeze()[-1, :]
            globalBindingFeature = self.globalFeatureLayer(globalInfoVector)
            globalBindingCentroid = self.globalBindingCentroid(globalInfoVector)
            globalBindingDirection = self.globalBindingDirection(globalInfoVector)
            return globalBindingFeature, globalBindingCentroid, globalBindingDirection
