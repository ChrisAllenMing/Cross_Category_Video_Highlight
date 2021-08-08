import torch
import torch.nn as nn

class ScoreFCN(nn.Module):
    def __init__(self, emb_dim = 4096, dropout_ratio = 0.5):
        super(ScoreFCN, self).__init__()
        self.emb_dim = emb_dim
        self.dropout_ratio = dropout_ratio

        self.fc1 = nn.Linear(self.emb_dim, int(self.emb_dim // 4))
        self.fc2 = nn.Linear(int(self.emb_dim // 4), int(self.emb_dim // 16))
        self.fc3 = nn.Linear(int(self.emb_dim // 16), 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = self.dropout_ratio)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        score = self.fc3(x)

        return score

if __name__ == "__main__":
    inputs = torch.rand(10, 4096)
    score_model = ScoreFCN(emb_dim = 4096)

    outputs = score_model(inputs)
    print(outputs.shape)