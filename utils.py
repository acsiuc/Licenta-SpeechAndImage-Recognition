import torch
import torch.nn.functional as F #we will need this for the log_softmax

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
    simMatrix = torch.matmul(embeddings, embeddings.T)
    labels = labels.view(-1, 1)
    maskPos = torch.eq(labels, labels.T).float()
    maskNeg = 1 - maskPos
    posLoss = torch.sum(maskPos * (1 - simMatrix)) / (maskPos.sum() + 1e-6)
    negLoss = torch.sum(maskNeg * torch.abs(simMatrix)) / (maskNeg.sum() + 1e-6)
    return posLoss + negLoss
    
def cross_modal_alignment_loss(face_embeddings, voice_embeddings, labels, temperature = 0.1):
    #this function will align our face and voice embeddings before fusion. pulls embeddings which are from same identity together and pushes diff ones apart

    #cosine similarity
    sim_matrix = torch.matmul(face_embeddings, voice_embeddings.T) / temperature

    #mask of positive pairs based on labels
    labels = labels.view(-1,1)
    mask_pos = torch.eq(labels,labels.T).float()

    #logsoftmax for getting the probabilitoes
    log_prob_face_to_voice = F.log_softmax(sim_matrix, dim = 1)
    log_prob_voice_to_face = F.log_softmax(sim_matrix.T, dim =1)

    #compute mean loss overpositive pairs
    loss_f2v = -(mask_pos * log_prob_face_to_voice).sum(dim=1)/(mask_pos.sum(dim=1)+ 1e-6)
    loss_v2f = -(mask_pos * log_prob_voice_to_face).sum(dim=1)/(mask_pos.sum(dim=1)+ 1e-6)

    return (loss_f2v.mean() + loss_v2f.mean()) / 2.0

def paeff_fusion(face_emb, voice_emb):

    stacked_embs = torch.stack([face_emb, voice_emb], dim=-1)
    
    # alculate Attention Weights (softmax forces the 2 weights to equal 100%)
    #  finds which modality has the stronger/more confident feature
    attention_weights = F.softmax(stacked_embs, dim=-1)
    
    fused_emb = torch.sum(stacked_embs * attention_weights, dim=-1)
    
    return fused_emb