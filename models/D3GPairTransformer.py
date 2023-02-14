import torch
from torch import nn
import numpy as np

#
# class D3PositionalEncoder(torch.nn.Module):
#     def __init__(self, d_model=512, device=None):
#         super(D3PositionalEncoder, self).__init__()
#
#         self.d_model = d_model
#         print("Dmodel size: ", self.d_model)
#         self.device = device
#         self.zeros = torch.zeros(self.d_model).to(device)
#         self.pad = nn.ConstantPad1d((0, 1), 0)
#
#     def forward(self, x, n_padding=0):
#         assert x.shape[1] == 3
#         nSeg = self.d_model // 3
#         scales = torch.linspace(-1, 1, nSeg)
#         xs = [x * sc for sc in scales]
#         xs = torch.cat(xs, dim=1)[:, :self.d_model]
#         # Padding 0
#         if n_padding == 0:
#             pad = self.pad
#         else:
#             pad = nn.ConstantPad1d((0, n_padding + 1), 0)
#         xs = pad(xs.t()).t()
#
#         return xs
#
#
# class D3PositionalEncoder11(torch.nn.Module):
#     def __init__(self, d_model=512, device=None):
#         super(D3PositionalEncoder11, self).__init__()
#
#         self.d_model = d_model
#         print("Dmodel size: ", self.d_model)
#         self.device = device
#         self.zeros = torch.zeros(self.d_model).to(device)
#         self.pad = nn.ConstantPad1d((0, 1), 0)
#
#     def forward(self, x, n_padding=0):
#         assert x.shape[1] == 3
#         nSeg = self.d_model // 3
#         scales = torch.linspace(-1, 1, nSeg)
#         xs = [x for sc in scales]
#         xs = torch.cat(xs, dim=1)[:, :self.d_model]
#         # Padding 0
#         if n_padding == 0:
#             pad = self.pad
#         else:
#             pad = nn.ConstantPad1d((0, n_padding + 1), 0)
#         xs = pad(xs.t()).t()
#
#         return xs
#
#
# class D3PositionalEncoder2(torch.nn.Module):
#     def __init__(self, d_model=512, device=None):
#         super(D3PositionalEncoder2, self).__init__()
#         self.d_model = d_model
#         print("Dmodel size: ", self.d_model)
#         self.device = device
#         self.zeros = torch.zeros(self.d_model).to(device)
#         self.pad = nn.ConstantPad1d((0, 1), 0)
#         self.nn1 = nn.Linear(3, d_model * 2)
#         self.nn2 = nn.Linear(d_model * 2, d_model)
#         self.actx = nn.LeakyReLU()
#
#     def forward(self, x, n_padding=0):
#         assert x.shape[1] == 3
#
#         xs = self.nn2(self.actx(self.nn1(x)))
#
#         # Padding 0
#         if n_padding == 0:
#             pad = self.pad
#         else:
#             pad = nn.ConstantPad1d((0, n_padding + 1), 0)
#         xs = pad(xs.t()).t()
#
#         return xs
#
#
# class D3PositionalEncoder3(torch.nn.Module):
#     def __init__(self, d_model=512, device=None):
#         super(D3PositionalEncoder3, self).__init__()
#         self.d_model = d_model
#         print("Dmodel size: ", self.d_model)
#         self.device = device
#         self.zeros = torch.zeros(self.d_model).to(device)
#         self.pad = nn.ConstantPad1d((0, 1), 0)
#         # self.nn1 = nn.Linear(3, d_model)
#         # self.nn2 = nn.Linear(d_model, d_model)
#         # self.actx = nn.LeakyReLU()
#
#     def forward(self, x, n_padding=0):
#         assert x.shape[1] == 3
#         n_seg = self.d_model // 6
#         xs = []
#         for i in range(n_seg):
#             t = torch.vstack([torch.sin(x[:, 0] / pow(10000, 6 * i / self.d_model)),
#                               torch.cos(x[:, 0] / pow(10000, 6 * i / self.d_model)),
#                               torch.sin(x[:, 1] / pow(10000, 6 * i / self.d_model)),
#                               torch.cos(x[:, 1] / pow(10000, 6 * i / self.d_model)),
#                               torch.sin(x[:, 2] / pow(10000, 6 * i / self.d_model)),
#                               torch.cos(x[:, 2] / pow(10000, 6 * i / self.d_model)),
#                               ]).t()
#             # print(t.shape)
#             xs.append(t)
#         # print(xs[0].shape)
#         xs = torch.cat(xs, dim=1)
#         # Padding 0
#         pad = None
#         if n_padding == 0:
#             pad = self.pad
#         elif n_padding != -1:
#             pad = nn.ConstantPad1d((0, n_padding + 1), 0)
#         if pad is not None:
#             xs = pad(xs.t()).t()
#
#         return xs


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
        pad = None
        if n_padding == 1:
            pad = self.pad
        elif n_padding > 1:
            pad = nn.ConstantPad1d((0, n_padding), 0)
        if pad is not None:
            xs = pad(xs.t()).t()
        return xs

#
# class D3PositionalEncoder5(torch.nn.Module):
#     def __init__(self, d_model=512, device=None):
#         super(D3PositionalEncoder5, self).__init__()
#         self.d_model = d_model
#         print("Dmodel size: ", self.d_model)
#         self.device = device
#         self.zeros = torch.zeros(self.d_model).to(device)
#         # self.pad = nn.ConstantPad1d((0, 1), 0)
#         self.lpad = torch.rand((1, d_model), requires_grad=True).to(self.device)
#         # self.nn1 = nn.Linear(3, d_model)
#         # self.nn2 = nn.Linear(d_model, d_model)
#         # self.actx = nn.LeakyReLU()
#
#     def forward(self, x, n_padding=0):
#         assert x.shape[1] == 3
#         n_seg = self.d_model // 6
#         xs = []
#         x = x * 1500
#         for i in range(n_seg):
#             t = torch.vstack([torch.sin(x[:, 0] / pow(10000, 6 * i / self.d_model)),
#                               torch.sin(x[:, 1] / pow(10000, 6 * i / self.d_model)),
#                               torch.sin(x[:, 2] / pow(10000, 6 * i / self.d_model)),
#                               torch.cos(x[:, 0] / pow(10000, 6 * i / self.d_model)),
#                               torch.cos(x[:, 1] / pow(10000, 6 * i / self.d_model)),
#                               torch.cos(x[:, 2] / pow(10000, 6 * i / self.d_model)),
#                               ]).t()
#             # print(t.shape)
#             xs.append(t)
#         # print(xs[0].shape)
#         xs = torch.cat(xs, dim=1)
#         # Padding 0
#         if n_padding == 0:
#             pad = self.lpad
#         else:
#             print("Error: No support I195")
#             exit(-1)
#         xs = torch.cat([xs, pad], dim=0)
#
#         return xs


class D3GPairTransformer(torch.nn.Module):
    def __init__(self, d_model=512, n_d3graph_layer=5, n_d3graph_head=5, d3_ff_size=2048, d3_graph_dropout_rate=0.1,
                 device=None):
        super(D3GPairTransformer, self).__init__()
        self.d_model = d_model
        self.n_layer = n_d3graph_layer
        self.dropout_rate = d3_graph_dropout_rate
        self.device = device
        self.pad = nn.ConstantPad1d((0, 1), 0)
        self.fe0 = torch.rand((6, d_model), requires_grad=True).to(device)
        self.pe = D3PositionalEncoder4(d_model, device).to(device)

        self.transformer = nn.Transformer(d_model=d_model, n_head=n_d3graph_head, dim_feedforward=d3_ff_size,
                                          dropout=d3_graph_dropout_rate).to(device)

        # self.globalFeatureLayer = nn.Linear(d_model, d_model).to(device)
        # self.globalBindingCentroid = nn.Linear(d_model, 3).to(device)
        # self.globalBindingDirection = nn.Linear(d_model, 3).to(device)

        self.globalFeatureLayer1 = nn.Linear(d_model, d_model).to(device)
        self.globalFeatureLayer2 = nn.Linear(d_model, d_model).to(device)
        self.globalBindingCentroid1 = nn.Linear(d_model, d_model).to(device)
        self.globalBindingDirection1 = nn.Linear(d_model, d_model).to(device)
        self.globalBindingDirection2 = nn.Linear(d_model, 3).to(device)
        # self.d3_act = torch.nn.ReLU()
        self.d3_act = torch.nn.LeakyReLU()

    def pad_x(self, x, n_padding=0, learn_pad=False):
        if learn_pad:
            return torch.cat([x, self.fe0[:n_padding, :]], dim=0)
        else:
            pad = None
            if n_padding == 1:
                pad = self.pad
            elif n_padding > 1:
                pad = nn.ConstantPad1d((0, n_padding), 0)
            if pad is not None:
                x = pad(x.t()).t()

        return x

    def mix_pe_feature(self, features, coords, n_padding=0, learn_pad=False):

        features = self.pad_x(features, n_padding, learn_pad)
        coords = self.pe(coords, n_padding)

        pe_features = features + coords
        return pe_features

    def forward(self, features_pro, coords_pro, features_lig, coords_lig):
        pe_feature_pro = self.mix_pe_feature(features_pro, coords_pro, n_padding=6, learn_pad=False)
        pe_feature_lig = self.mix_pe_feature(features_lig, coords_lig, n_padding=0, learn_pad=False)

        pe_feature_pro = torch.unsqueeze(pe_feature_pro, 1).to(self.device)
        pe_feature_lig = torch.unsqueeze(pe_feature_lig, 1).to(self.device)
        pe_feature_pro = self.dropout(pe_feature_pro)
        pe_feature_lig = self.dropout(pe_feature_lig)

        src_mask = torch.zeros((pe_feature_pro.shape[0], pe_feature_pro.shape[0]), device=self.device).type(torch.bool)
        tgt_mask = torch.zeros((pe_feature_lig.shape[0], pe_feature_lig.shape[0]), device=self.device).type(torch.bool)

        out = self.transformer(pe_feature_pro, pe_feature_lig, src_mask=src_mask, tgt_mask=tgt_mask)
        out = torch.squeeze(out)
        globalBindingFeature1 = self.globalFeatureLayer2(self.d3_act(self.globalFeatureLayer1(out[-6, :])))
        globalBindingFeature2 = self.globalFeatureLayer2(self.d3_act(self.globalFeatureLayer1(out[-5, :])))

        globalBindingCentroid1 = self.globalBindingCentroid2(
            self.d3_act(self.globalBindingCentroid1(self.globalFeatureLayer1(out[-4, :]))))

        globalBindingCentroid2 = self.globalBindingCentroid2(
            self.d3_act(self.globalBindingCentroid1(self.globalFeatureLayer1(out[-3, :]))))

        globalBindingDirection1 = self.globalBindingDirection2(
            self.d3_act(self.globalBindingDirection1(self.globalFeatureLayer1(out[-2, :]))))

        globalBindingDirection2 = self.globalBindingDirection2(
            self.d3_act(self.globalBindingDirection1(self.globalFeatureLayer1(out[-1, :]))))

        return globalBindingFeature1, globalBindingFeature2, globalBindingCentroid1, globalBindingCentroid2, globalBindingDirection1, globalBindingDirection2