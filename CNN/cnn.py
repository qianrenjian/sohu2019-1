import torch
import torch.nn as nn
import torch.nn.functional as f


class CNN(nn.Module):
    def __init__(self, text, emb_dim, conv_num, dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(len(text.vocab), emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.encoder = [nn.Conv1d(in_channels=emb_dim, out_channels=256, kernel_size=5, padding=2)]
        for i in range(conv_num - 1):  # 0 - conv-2
            self.encoder.append(nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, padding=2))
        self.linear = nn.Linear(256, 3)

    def forward(self, text, title):
        x = torch.cat(self.embedding(text), self.embedding(title))
        x = self.dropout(x).transpose(1, 2)

        for i, layer in enumerate(self.encoder):
            if i == len(self.encoder) - 1:
                x = f.relu(layer(x))
            else:
                x = self.dropout(f.relu(layer(x)))

        x = x.trasnpose(1, 2)
        x = self.linear(x)
        return x
