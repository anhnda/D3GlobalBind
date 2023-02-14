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


class D3PositionalEncoder11(torch.nn.Module):
    def __init__(self, d_model=512, device=None):
        super(D3PositionalEncoder11, self).__init__()

        self.d_model = d_model
        print("Dmodel size: ", self.d_model)
        self.device = device
        self.zeros = torch.zeros(self.d_model).to(device)
        self.pad = nn.ConstantPad1d((0, 1), 0)

    def forward(self, x, n_padding=0):
        assert x.shape[1] == 3
        nSeg = self.d_model // 3
        scales = torch.linspace(-1, 1, nSeg)
        xs = [x for sc in scales]
        xs = torch.cat(xs, dim=1)[:, :self.d_model]
        # Padding 0
        if n_padding == 0:
            pad = self.pad
        else:
            pad = nn.ConstantPad1d((0, n_padding + 1), 0)
        xs = pad(xs.t()).t()

        return xs


class D3PositionalEncoder2(torch.nn.Module):
    def __init__(self, d_model=512, device=None):
        super(D3PositionalEncoder2, self).__init__()
        self.d_model = d_model
        print("Dmodel size: ", self.d_model)
        self.device = device
        self.zeros = torch.zeros(self.d_model).to(device)
        self.pad = nn.ConstantPad1d((0, 1), 0)
        self.nn1 = nn.Linear(3, d_model * 2)
        self.nn2 = nn.Linear(d_model * 2, d_model)
        self.actx = nn.LeakyReLU()

    def forward(self, x, n_padding=0):
        assert x.shape[1] == 3

        xs = self.nn2(self.actx(self.nn1(x)))

        # Padding 0
        if n_padding == 0:
            pad = self.pad
        else:
            pad = nn.ConstantPad1d((0, n_padding + 1), 0)
        xs = pad(xs.t()).t()

        return xs


class D3PositionalEncoder3(torch.nn.Module):
    def __init__(self, d_model=512, device=None):
        super(D3PositionalEncoder3, self).__init__()
        self.d_model = d_model
        print("Dmodel size: ", self.d_model)
        self.device = device
        self.zeros = torch.zeros(self.d_model).to(device)
        self.pad = nn.ConstantPad1d((0, 1), 0)
        # self.nn1 = nn.Linear(3, d_model)
        # self.nn2 = nn.Linear(d_model, d_model)
        # self.actx = nn.LeakyReLU()

    def forward(self, x, n_padding=0):
        assert x.shape[1] == 3
        n_seg = self.d_model // 6
        xs = []
        for i in range(n_seg):
            t = torch.vstack([torch.sin(x[:, 0] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 0] / pow(10000, 6 * i / self.d_model)),
                              torch.sin(x[:, 1] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 1] / pow(10000, 6 * i / self.d_model)),
                              torch.sin(x[:, 2] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 2] / pow(10000, 6 * i / self.d_model)),
                              ]).t()
            # print(t.shape)
            xs.append(t)
        # print(xs[0].shape)
        xs = torch.cat(xs, dim=1)
        # Padding 0
        if n_padding == 0:
            pad = self.pad
        else:
            pad = nn.ConstantPad1d((0, n_padding + 1), 0)
        xs = pad(xs.t()).t()

        return xs


class D3PositionalEncoder4(torch.nn.Module):
    def __init__(self, d_model=512, device=None):
        super(D3PositionalEncoder4, self).__init__()
        self.d_model = d_model
        print("Dmodel size: ", self.d_model)
        self.device = device
        self.zeros = torch.zeros(self.d_model).to(device)
        self.pad = nn.ConstantPad1d((0, 1), 0)
        # self.nn1 = nn.Linear(3, d_model)
        # self.nn2 = nn.Linear(d_model, d_model)
        # self.actx = nn.LeakyReLU()

    def forward(self, x, n_padding=0):
        assert x.shape[1] == 3
        n_seg = self.d_model // 6
        xs = []
        x = x * 1500
        for i in range(n_seg):
            t = torch.vstack([torch.sin(x[:, 0] / pow(10000, 6 * i / self.d_model)),
                              torch.sin(x[:, 1] / pow(10000, 6 * i / self.d_model)),
                              torch.sin(x[:, 2] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 0] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 1] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 2] / pow(10000, 6 * i / self.d_model)),
                              ]).t()
            # print(t.shape)
            xs.append(t)
        # print(xs[0].shape)
        xs = torch.cat(xs, dim=1)
        # Padding 0
        if n_padding == 0:
            pad = self.pad
        else:
            pad = nn.ConstantPad1d((0, n_padding + 1), 0)
        xs = pad(xs.t()).t()

        return xs


class D3PositionalEncoder5(torch.nn.Module):
    def __init__(self, d_model=512, device=None):
        super(D3PositionalEncoder5, self).__init__()
        self.d_model = d_model
        print("Dmodel size: ", self.d_model)
        self.device = device
        self.zeros = torch.zeros(self.d_model).to(device)
        # self.pad = nn.ConstantPad1d((0, 1), 0)
        self.lpad = torch.rand((1, d_model), requires_grad=True).to(self.device)
        # self.nn1 = nn.Linear(3, d_model)
        # self.nn2 = nn.Linear(d_model, d_model)
        # self.actx = nn.LeakyReLU()

    def forward(self, x, n_padding=0):
        assert x.shape[1] == 3
        n_seg = self.d_model // 6
        xs = []
        x = x * 1500
        for i in range(n_seg):
            t = torch.vstack([torch.sin(x[:, 0] / pow(10000, 6 * i / self.d_model)),
                              torch.sin(x[:, 1] / pow(10000, 6 * i / self.d_model)),
                              torch.sin(x[:, 2] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 0] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 1] / pow(10000, 6 * i / self.d_model)),
                              torch.cos(x[:, 2] / pow(10000, 6 * i / self.d_model)),
                              ]).t()
            # print(t.shape)
            xs.append(t)
        # print(xs[0].shape)
        xs = torch.cat(xs, dim=1)
        # Padding 0
        if n_padding == 0:
            pad = self.lpad
        else:
            print("Error: No support I195")
            exit(-1)
        xs = torch.cat([xs, pad], dim=0)

        return xs


class D3GraphEncoder(torch.nn.Module):
    def __init__(self, d_model=512, n_d3graph_layer=5, n_d3graph_head=5, d3_ff_size=2048, d3_graph_dropout_rate=0.1,
                 device=None):
        super(D3GraphEncoder, self).__init__()
        self.d_model = d_model
        self.n_layer = n_d3graph_layer
        self.dropout_rate = d3_graph_dropout_rate
        self.device = device
        # self.pad = nn.ConstantPad1d((0, 1), 0)
        self.fe0 = torch.rand((1, d_model), requires_grad=True).to(device)
        self.pe = D3PositionalEncoder4(d_model, device).to(device)
        self.dropout = nn.Dropout(d3_graph_dropout_rate)
        transformerEncoderLayer = nn.TransformerEncoderLayer(d_model, n_d3graph_head, d3_ff_size,
                                                             d3_graph_dropout_rate).to(device)
        self.transformerEncoder = nn.TransformerEncoder(transformerEncoderLayer, num_layers=n_d3graph_layer).to(device)

        # self.globalFeatureLayer = nn.Linear(d_model, d_model).to(device)
        # self.globalBindingCentroid = nn.Linear(d_model, 3).to(device)
        # self.globalBindingDirection = nn.Linear(d_model, 3).to(device)

        self.globalFeatureLayer1 = nn.Linear(d_model, d_model).to(device)
        self.globalFeatureLayer2 = nn.Linear(d_model, d_model).to(device)
        self.globalBindingCentroid1 = nn.Linear(d_model, d_model).to(device)
        self.globalBindingCentroid2 = nn.Linear(d_model, 3).to(device)
        self.globalBindingDirection1 = nn.Linear(d_model, d_model).to(device)
        self.globalBindingDirection2 = nn.Linear(d_model, 3).to(device)
        # self.d3_act = torch.nn.ReLU()
        self.d3_act = torch.nn.LeakyReLU()

    def pad_feature(self, features):
        return torch.cat([features, self.fe0], dim=0)

    def forward(self, features, coords):
        if type(features) == list and len(features[0].shape) == 2:
            print("Not revised yet.")
            exit(-1)
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
            # globalBindingFeatures = self.globalFeatureLayer(out_selected)
            # globalBindingDirections = self.globalBindingDirection(out_selected)
            # globalBindingCentroids = self.globalBindingCentroid(out_selected)
            globalBindingFeatures = self.globalFeatureLayer2(self.d3_act(self.globalFeatureLayer1(out_selected)))
            globalBindingCentroids = self.globalBindingCentroid2(self.d3_act(self.globalBindingCentroid1(out_selected)))
            globalBindingDirections = self.globalBindingDirection2(
                self.d3_act(self.globalBindingDirection2(out_selected)))
            return globalBindingFeatures, globalBindingDirections, globalBindingCentroids
        else:
            # features = self.pad(features.t()).t()
            features = self.pad_feature(features)
            coords = self.pe(coords)
            features = features + coords
            features = torch.unsqueeze(features, 1).to(self.device)
            features = self.dropout(features)
            src_mask = torch.zeros((features.shape[0], features.shape[0]), device=self.device).type(torch.bool)

            out_all = self.transformerEncoder(features, src_mask)
            globalInfoVector = out_all.squeeze()[-1, :]
            #
            # globalBindingFeature = self.globalFeatureLayer(globalInfoVector)
            # globalBindingCentroid = self.globalBindingCentroid(globalInfoVector)
            # globalBindingDirection = self.globalBindingDirection(globalInfoVector)

            globalBindingFeature = self.globalFeatureLayer2(self.d3_act(self.globalFeatureLayer1(globalInfoVector)))
            globalBindingCentroid = self.globalBindingCentroid2(
                self.d3_act(self.globalBindingCentroid1(globalInfoVector)))
            globalBindingDirection = self.globalBindingDirection2(
                self.d3_act(self.globalBindingDirection1(globalInfoVector)))

            return globalBindingFeature, globalBindingCentroid, globalBindingDirection
