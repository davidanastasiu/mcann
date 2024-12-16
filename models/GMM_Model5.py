import time  # unused?
import os  # unused?
import sys  # unused?
import math
import torch.nn.functional as F  # unused?
import numpy as np  # unused?
import torch
import torch.nn as nn
import torch.optim as optim  # unused?
import pandas as pd  # unused?
import random
from sklearn.metrics import mean_absolute_percentage_error  # unused?
import logging

logging.basicConfig(filename="Encoder_LSTM.log", filemode="w", level=logging.DEBUG)
random.seed("a")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class EncoderLSTM(nn.Module):
    def __init__(self, opt):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.input_len = opt.input_len
        self.output_len = opt.output_len
        self.seq_w = opt.seq_weight
        atten_dim = opt.atten_dim

        self.lstm0 = nn.LSTM(
            2,
            self.hidden_dim,
            self.layer_dim,
            dropout=0.1,
            bidirectional=False,
            batch_first=True,
        )
        self.lstm1 = nn.LSTM(
            2,
            self.hidden_dim,
            self.layer_dim,
            dropout=0.1,
            bidirectional=False,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            2,
            self.hidden_dim,
            self.layer_dim,
            dropout=0.1,
            bidirectional=False,
            batch_first=True,
        )
        self.attn0 = nn.MultiheadAttention(atten_dim, 4)
        self.attn1 = nn.MultiheadAttention(atten_dim, 4)
        self.attn2 = nn.MultiheadAttention(atten_dim, 4)
        self.attn3 = nn.MultiheadAttention(atten_dim, 4)
        self.attn4 = nn.MultiheadAttention(atten_dim, 4)
        self.attn5 = nn.MultiheadAttention(atten_dim, 4)
        self.L_out00 = nn.Linear(1, atten_dim)
        self.L_out10 = nn.Linear(atten_dim, 1)
        self.L_out01 = nn.Linear(1, atten_dim)
        self.L_out11 = nn.Linear(atten_dim, 1)
        self.L_out02 = nn.Linear(1, atten_dim)
        self.L_out12 = nn.Linear(atten_dim, 1)
        self.bn = nn.BatchNorm1d(self.output_len)
        self.L_out0 = nn.Linear(1, int(atten_dim / 2))
        self.L_out1 = nn.Linear(1, int(atten_dim / 2))
        self.L_out2 = nn.Linear(1, int(atten_dim / 2))

        self.ebb = PositionalEmbedding(int(atten_dim / 2), self.output_len)

    def forward(self, x):

        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(device)
        h = []
        c = []
        x0 = x[:, :, 0:2]
        relu = nn.ReLU()
        m = nn.Softmax(dim=2)
        tanh = nn.Tanh()

        ww0 = x[:, :, 5:8]
        ww1 = x[:, :, 2:5]
        ww = torch.add(ww0, ww1 * self.seq_w)
        ww = ww[:, -1 * self.output_len:, :]
        wwe = self.ebb(ww)
        wwe = wwe.repeat(ww0.size(0), 1, 1)

        # embeding -> self-attention -> add&norm -> self-attention -> add&norm -> linear -> softmax
        ww0 = ww[:, :, 0:1]
        ww0 = tanh(self.L_out0(ww0))
        ww0 = torch.cat((ww0, wwe), dim=2)
        ww00, _ = self.attn0(ww0, ww0, ww0)
        ww0 = self.bn(ww0 + ww00)
        ww00, _ = self.attn3(ww0, ww0, ww0)
        ww0 = self.bn(ww0 + ww00)
        ww0 = self.L_out10(relu(ww0))

        ww1 = ww[:, :, 1:2]
        ww1 = tanh(self.L_out1(ww1))
        ww1 = torch.cat((ww1, wwe), dim=2)
        ww11, _ = self.attn1(ww1, ww1, ww1)
        ww1 = self.bn(ww1 + ww11)
        ww11, _ = self.attn4(ww1, ww1, ww1)
        ww1 = self.bn(ww1 + ww11)
        ww1 = self.L_out11(relu(ww1))

        ww2 = ww[:, :, 2:3]
        ww2 = self.L_out2(ww2)
        ww2 = torch.cat((ww2, wwe), dim=2)
        ww22, _ = self.attn2(ww2, ww2, ww2)
        ww2 = self.bn(ww2 + ww22)
        ww22, _ = self.attn5(ww2, ww2, ww2)
        ww2 = self.bn(ww2 + ww22)
        ww2 = self.L_out12(relu(ww2))

        ww = torch.cat((ww0, ww1), dim=2)
        ww = torch.cat((ww, ww2), dim=2)
        # softmax
        ww = m(ww)

        out, (hn, cn) = self.lstm0(x0, (h0, c0))
        h.append(hn)
        c.append(cn)
        out, (hn, cn) = self.lstm1(x0, (h0, c0))
        h.append(hn)
        c.append(cn)
        out, (hn, cn) = self.lstm2(x0, (h0, c0))
        h.append(hn)
        c.append(cn)

        return h, c, ww


class DecoderLSTM(nn.Module):
    def __init__(self, opt):
        super(DecoderLSTM, self).__init__()
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer
        self.output_len = opt.output_len
        self.m = nn.ELU()
        atten_dim = opt.atten_dim
        self.tanh = nn.Tanh()

        self.lstm0 = nn.LSTM(
            2,
            self.hidden_dim,
            self.layer_dim,
            dropout=0.1,
            bidirectional=False,
            batch_first=True,
        )
        self.lstm1 = nn.LSTM(
            2,
            self.hidden_dim,
            self.layer_dim,
            dropout=0.1,
            bidirectional=False,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            2,
            self.hidden_dim,
            self.layer_dim,
            dropout=0.1,
            bidirectional=False,
            batch_first=True,
        )
        self.bn0 = nn.BatchNorm1d(opt.output_len)
        self.bn1 = nn.BatchNorm1d(opt.output_len)
        self.bn2 = nn.BatchNorm1d(opt.output_len)

        self.L_out0 = nn.Linear(1, self.hidden_dim)
        self.L_out1 = nn.Linear(self.hidden_dim, atten_dim)
        self.L_out2 = nn.Linear(self.hidden_dim, 1)
        self.L_out3 = nn.Linear(self.hidden_dim, 1)
        self.L_out4 = nn.Linear(self.hidden_dim, 1)
        self.L_out5 = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, encoder_h, encoder_c, ww):
        # Initialize hidden and cell state with zeros
        h0 = encoder_h[0]
        c0 = encoder_c[0]
        h1 = encoder_h[1]
        c1 = encoder_c[1]
        h2 = encoder_h[2]
        c2 = encoder_c[2]
        relu = nn.ReLU()  # unused?
        hn = h0
        cn = c0
        out0, (hnn, cnn) = self.lstm0(x, (hn, cn))
        out0 = torch.squeeze(self.L_out3(self.tanh(out0)))

        hn = h1
        cn = c1
        out1, (hnn, cnn) = self.lstm1(x, (hn, cn))
        out1 = torch.squeeze(self.L_out4(self.tanh(out1)))

        hn = h2
        cn = c2
        out2, (hnn, cnn) = self.lstm2(x, (hn, cn))
        out2 = torch.squeeze(self.L_out5(self.tanh(out2)))

        w0 = torch.squeeze(ww[:, :, 0:1], dim=2)
        w1 = torch.squeeze(ww[:, :, 1:2], dim=2)
        w2 = torch.squeeze(ww[:, :, 2:], dim=2)

        out = torch.unsqueeze(
            (torch.mul(out0, w0) + torch.mul(out1, w1) + torch.mul(out2, w2)), 2
        )
        out = torch.squeeze(out)

        return out
