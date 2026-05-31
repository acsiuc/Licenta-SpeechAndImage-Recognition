import torch
import torch.nn as nn
import torch.nn.functional as F


EMBEDDING_DIM = 128 # base size for our initial frozen vectors

class FaceEncoder(nn.Module):
    # ArcFace iResNet50 pretrained on VGGFace2 via InsightFace buffalo_l
    # Purpose-built for face identity recognition — extracts 512D embeddings
    # that capture identity-discriminative facial geometry (jaw width, orbital
    # spacing, nasal structure) which biologically correlates with vocal tract.
    # Reference: Deng et al., ArcFace, CVPR 2019.
    def __init__(self):
        super(FaceEncoder, self).__init__()
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=-1, det_size=(128, 128))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, img_bgr: torch.Tensor) -> torch.Tensor:
        # img_bgr: (B, H, W, 3) numpy uint8 or (B, 3, H, W) tensor
        # Returns: (B, 512) L2-normalized embeddings
        if isinstance(img_bgr, torch.Tensor):
            # Convert from (B, 3, H, W) float tensor to (B, H, W, 3) uint8 numpy
            img_bgr = (img_bgr.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
        
        embeddings = []
        for img in img_bgr:
            faces = self.app.get(img)
            if len(faces) > 0:
                emb = torch.tensor(faces[0].embedding, dtype=torch.float32)
            else:
                emb = torch.zeros(512, dtype=torch.float32)
            embeddings.append(emb)
        
        emb_tensor = torch.stack(embeddings)
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)
        return emb_tensor


class VoiceEncoder(nn.Module):
    # ECAPA-TDNN pretrained on VoxCeleb1+2 — purpose-built for speaker
    # verification using 1D dilated convolutions that understand speech
    # natively. Replaces ImageNet ResNet18 which had no acoustic knowledge.
    # Reference: Desplanques et al., Interspeech 2020.
    def __init__(self, savedir: str = "pretrained_ecapa"):
        super(VoiceEncoder, self).__init__()
        from speechbrain.inference.speaker import EncoderClassifier
        self.encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=savedir,
            run_opts={"device": "cpu"}
        )
        # Freeze all parameters — we use this as a fixed feature extractor
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, waveform):
        # waveform: (B, T) raw audio at 16kHz, mono
        # output:   (B, 192) L2-normalized speaker embedding
        emb = self.encoder.encode_batch(waveform)  # (B, 1, 192)
        emb = emb.squeeze(1)                        # (B, 192)
        return F.normalize(emb, p=2, dim=1)

class JointClassifier(nn.Module):
    # the bouncer that guesses the final name
    def __init__(self, num_classes, embedding_dim: int = 512): # expanded 512d door to match our upgraded translators
        super(JointClassifier, self).__init__()
        # added a hidden layer to make the bouncer think deeper before guessing
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 1024), # inflate into a bigger thinking space
            nn.ReLU(), # activation function so it can learn complex patterns
            nn.Linear(1024, num_classes) # narrow it down to the exact number of buttons (identities)
        )

    def forward(self, x):
        return self.classifier(x) # make the final guess

class ModalityTranslator(nn.Module):
    def __init__(self, input_dim: int = 128, output_dim: int = 512, dropout: float = 0.3):
        super(ModalityTranslator, self).__init__()
        
        # removed the old Tanh/Sigmoid attention here.
        # a deep, stable projector to get the 128d vectors up to 512d.
        self.projector = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        out = self.projector(x)
        return F.normalize(out, p=2, dim=1)

class TransformerCrossAttention(nn.Module):
    # mplement the Q, K, V math from "The Illustrated Transformer"
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, dropout: float = 0.3):
        super(TransformerCrossAttention, self).__init__()
        
        # PyTorch's built-in Multihead Attention 
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, face_emb, voice_emb):
        #  modalities into a sequence of length 2: [Face, Voice]
        # Shape becomes: (Batch_Size, 2_tokens, 512_dimensions)
        seq = torch.stack([face_emb, voice_emb], dim=1) 
        
        # Face and Voice act as Queries, Keys, and Values for each other
        attn_output, attn_weights = self.multihead_attn(query=seq, key=seq, value=seq)
        
        # add & normalize 
        seq_out = self.layer_norm(seq + self.dropout(attn_output))
        
        # average the two attention-boosted vectors together to get the final fused vector
        fused_emb = torch.mean(seq_out, dim=1)
        return F.normalize(fused_emb, p=2, dim=1)