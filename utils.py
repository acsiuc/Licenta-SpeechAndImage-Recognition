import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def orthogonal_projection_loss(embeddings, labels):
    # (Copy your existing loss function here)
    simMatrix = torch.matmul(embeddings, embeddings.T)
    labels = labels.view(-1, 1)
    maskPos = torch.eq(labels, labels.T).float()
    maskNeg = 1 - maskPos
    posLoss = torch.sum(maskPos * (1 - simMatrix)) / (maskPos.sum() + 1e-6)
    negLoss = torch.sum(maskNeg * torch.abs(simMatrix)) / (maskNeg.sum() + 1e-6)
    return posLoss + negLoss