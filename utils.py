import torch
import torch.nn.functional as F # we will need this for the log_softmax

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        # setup the patience monitor so we don't overcook the model
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0 # strike counter
        self.best_loss = None # tracks the high score
        self.early_stop = False # the kill switch

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss # set the baseline score on the first try
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1 # add a strike if the math didn't improve
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True # pull the plug if it's been too long without progress
        else:
            self.best_loss = val_loss # record the new high score
            self.counter = 0 # reset the strikes

def orthogonal_projection_loss(embeddings, labels):
    #  make sure different people are pushed far apart into their own math corners
    simMatrix = torch.matmul(embeddings, embeddings.T) # measure the distance between everyone
    labels = labels.view(-1, 1)
    maskPos = torch.eq(labels, labels.T).float() # find the vectors that belong to the same person
    maskNeg = 1 - maskPos # find the vectors that belong to completely different people
    posLoss = torch.sum(maskPos * (1 - simMatrix)) / (maskPos.sum() + 1e-6) # penalty if the same person's vectors aren't right on top of each other
    negLoss = torch.sum(maskNeg * torch.abs(simMatrix)) / (maskNeg.sum() + 1e-6) # penalty if different people are intruding on each other's space
    return posLoss + negLoss
    
def cross_modal_alignment_loss(face_embeddings, voice_embeddings, labels, temperature = 0.1):
    # this function will align our face and voice embeddings before fusion. pulls embeddings which are from same identity together and pushes diff ones apart

    # cosine similarity
    sim_matrix = torch.matmul(face_embeddings, voice_embeddings.T) / temperature # measure the math distance between every face and every voice

    # mask of positive pairs based on labels
    labels = labels.view(-1,1)
    mask_pos = torch.eq(labels,labels.T).float() # create a cheat sheet of the true face-voice pairs

    # logsoftmax for getting the probabilitoes
    log_prob_face_to_voice = F.log_softmax(sim_matrix, dim = 1) # turn the distances into percentages for face looking at voice
    log_prob_voice_to_face = F.log_softmax(sim_matrix.T, dim =1) # turn the distances into percentages for voice looking at face

    # compute mean loss overpositive pairs
    loss_f2v = -(mask_pos * log_prob_face_to_voice).sum(dim=1)/(mask_pos.sum(dim=1)+ 1e-6) # penalty if the face couldn't find its matching voice
    loss_v2f = -(mask_pos * log_prob_voice_to_face).sum(dim=1)/(mask_pos.sum(dim=1)+ 1e-6) # penalty if the voice couldn't find its matching face

    return (loss_f2v.mean() + loss_v2f.mean()) / 2.0 # average the two penalties together

def paeff_fusion(face_emb, voice_emb):
    # the magic fusion machine

    stacked_embs = torch.stack([face_emb, voice_emb], dim=-1) # put the face and voice side by side
    
    # calculate attention weights (softmax forces the 2 weights to equal 100%)
    # finds which modality has the stronger/more confident feature
    attention_weights = F.softmax(stacked_embs, dim=-1) 
    
    fused_emb = torch.sum(stacked_embs * attention_weights, dim=-1) # smash them together using the confidence weights
    
    return fused_emb