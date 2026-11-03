"""mLSTM workload predictor: faithful copy of upstream PreServe.

Source: PreServe ``experiments/RQ1/workload_predictor/mLSTM_predict.py`` (classes
``mLSTMCell``, ``mLSTM``, ``mLSTMModel``, ``TimeSeriesDataset``, ``set_seed``, and the training
loop of ``workload_predictor.__init__``). Only logging and device selection were removed
(training runs on CPU).

Faithful-copy quirk, kept on purpose: ``mLSTMCell.forward`` computes the multiplicative state
``m = tanh(W_m(x) * U_m(h))`` but never uses it in the gates, so the cell is effectively a
standard LSTM cell. W_m / U_m get no gradient and stay at their initial values.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class mLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_ih = nn.Linear(input_size, 4 * hidden_size)
        self.W_hh = nn.Linear(hidden_size, 4 * hidden_size)
        self.W_m = nn.Linear(input_size, hidden_size)
        self.U_m = nn.Linear(hidden_size, hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, input, hx=None):
        if hx is None:
            hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hx

        # Upstream computes m but does not use it below (faithful copy; effectively an LSTM).
        m = self.tanh(self.W_m(input) * self.U_m(hx))  # noqa: F841

        gates = self.W_ih(input) + self.W_hh(hx)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.sigmoid(ingate)
        forgetgate = self.sigmoid(forgetgate)
        cellgate = self.tanh(cellgate)
        outgate = self.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)

        hy = outgate * self.tanh(cy)

        return hy, cy


class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=False):
        super(mLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.cells.append(mLSTMCell(input_size, hidden_size))
            else:
                self.cells.append(mLSTMCell(hidden_size, hidden_size))

    def forward(self, input, hx=None):
        if self.batch_first:
            input = input.transpose(0, 1)

        seq_len, batch_size, _ = input.size()

        if hx is None:
            hx = [None] * self.num_layers
        else:
            hx = list(zip(*hx))

        outputs = []
        for t in range(seq_len):
            x = input[t]
            for layer in range(self.num_layers):
                hx[layer] = self.cells[layer](x, hx[layer])
                x = hx[layer][0]
            outputs.append(x)

        outputs = torch.stack(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)

        hx = list(zip(*hx))
        return outputs, hx


class mLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(mLSTMModel, self).__init__()
        self.mLSTM = mLSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.mLSTM(x)
        out = self.fc(out[:, -1, :])
        return out


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_mlstm(X, y, hidden_dim=64, num_epochs=50, batch_size=32, lr=0.001, seed=42, log=None):
    """Upstream ``workload_predictor.__init__`` training loop on prepared (normalized) data.

    X: (N, seq_len) or (N, seq_len, 1); y: (N,). Returns the trained model (eval mode).
    Upstream builds X/y from one global min-max-scaled series; here the caller prepares them
    (per-sequence or global normalization, sequences never spanning an excluded gap).
    """
    set_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = np.asarray(y, dtype=np.float32)
    train_loader = DataLoader(TimeSeriesDataset(X, y), batch_size=batch_size, shuffle=True)

    model = mLSTMModel(input_dim=1, hidden_dim=hidden_dim, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch).squeeze()
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        if log is not None and epoch % 10 == 0:
            log(f"epoch {epoch + 1}/{num_epochs} loss {train_loss / len(train_loader):.6f}")
    model.eval()
    return model


def forward_batch(model, X, batch=4096):
    """Model outputs for (N, seq_len) normalized inputs, as float64 numpy."""
    X = np.asarray(X, dtype=np.float32)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            outs.append(model(torch.from_numpy(X[i:i + batch])).reshape(-1).double().numpy())
    return np.concatenate(outs) if outs else np.zeros(0)
