import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import glob
import random as _random
from models import JointClassifier, ModalityTranslator, TransformerCrossAttention
from utils import supervised_contrastive_loss, EarlyStopping, cross_modal_alignment_loss

# ---- Paths ----
MAVCELEB_EMBEDDING_DIR = "/content/drive/MyDrive/Licenta_Colab/embeddings"
RO_EMBEDDING_DIR = "/content/drive/MyDrive/Licenta_Colab/ro_embeddings_train"
OLD_CHECKPOINT_PATH = "/content/drive/MyDrive/Licenta_Colab/checkpoints/model_arcface.pth"
NEW_CHECKPOINT_PATH = "/content/drive/MyDrive/Licenta_Colab/checkpoints/model_arcface_ro_finetuned.pth"

# ---- Config ----
OLD_NUM_CLASSES = 198
NEW_IDENTITIES = 33          # the 33 Romanian training identities
NUM_CLASSES = OLD_NUM_CLASSES + NEW_IDENTITIES  # 231
BATCH_SIZE = 64
FINETUNE_EPOCHS = 10
FINETUNE_LR = 0.0001         # much lower than the original 0.001 — gentle adjustment, not relearning
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiDirEmbeddingDataset(Dataset):
    """Same format as the original EmbeddingDataset, but reads .pt files
    from two directories (existing MAVCeleb embeddings + new RO embeddings)
    combined into a single pool."""
    def __init__(self, directories):
        self.files = []
        for d in directories:
            self.files.extend(glob.glob(os.path.join(d, "*.pt")))
        print(f"Combined dataset: {len(self.files)} total samples from {len(directories)} directories")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=False)
        return data['face_emb'].squeeze(0), data['voice_emb'].squeeze(0), data['label']


if __name__ == "__main__":

    class SubsampledFinetuneDataset(Dataset):
        """Uses ALL Romanian samples (repeated for stronger gradient signal)
        plus a fixed random subset of the old MAVCeleb data, instead of the
        full 98,828 samples every epoch. This makes each epoch much faster
        and prevents the tiny Romanian signal from being drowned out."""
        def __init__(self, mavceleb_dir, ro_dir, mavceleb_subset_size=15000, ro_repeat=8, seed=42):
            mavceleb_files = glob.glob(os.path.join(mavceleb_dir, "*.pt"))
            ro_files = glob.glob(os.path.join(ro_dir, "*.pt"))

            rng = _random.Random(seed)
            mavceleb_sample = rng.sample(mavceleb_files, min(mavceleb_subset_size, len(mavceleb_files)))

            self.files = mavceleb_sample + (ro_files * ro_repeat)
            print(f"Subsampled dataset: {len(mavceleb_sample)} MAVCeleb samples "
                  f"+ {len(ro_files)} Romanian samples x{ro_repeat} repeats "
                  f"= {len(self.files)} total")

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            data = torch.load(self.files[idx], weights_only=False)
            return data['face_emb'].squeeze(0), data['voice_emb'].squeeze(0), data['label']

    full_dataset = SubsampledFinetuneDataset(MAVCELEB_EMBEDDING_DIR, RO_EMBEDDING_DIR)

    train_size = int(0.9 * len(full_dataset))  # smaller val split since we're fine-tuning, not training from scratch
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # ---- Build model with the EXPANDED class count ----
    model               = JointClassifier(num_classes=NUM_CLASSES, embedding_dim=512).to(DEVICE)
    face_translator     = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator    = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)
    transformer_fusion  = TransformerCrossAttention(embed_dim=512).to(DEVICE)

    # ---- Warm-start from the existing checkpoint ----
    print(f"Loading checkpoint from {OLD_CHECKPOINT_PATH}")
    checkpoint = torch.load(OLD_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    # translators and fusion module have unchanged shapes — load directly
    face_translator.load_state_dict(checkpoint['face_translator'])
    voice_translator.load_state_dict(checkpoint['voice_translator'])
    transformer_fusion.load_state_dict(checkpoint['transformer_fusion'])

    # classifier's final layer changed shape (198 -> 231 outputs), so we
    # copy weights manually: keep everything for the first layer, and for
    # the final layer, keep the old 198 classes' weights/bias and leave
    # the new 33 output neurons at their random initialization
    old_classifier_state = checkpoint['classifier']
    new_classifier_state = model.state_dict()

    # first Linear layer (512 -> 1024): unchanged shape, copy directly
    new_classifier_state['classifier.0.weight'] = old_classifier_state['classifier.0.weight']
    new_classifier_state['classifier.0.bias']   = old_classifier_state['classifier.0.bias']

    # final Linear layer (1024 -> num_classes): copy only the first 198 rows
    old_final_weight = old_classifier_state['classifier.2.weight']  # shape (198, 1024)
    old_final_bias   = old_classifier_state['classifier.2.bias']    # shape (198,)
    new_classifier_state['classifier.2.weight'][:OLD_NUM_CLASSES] = old_final_weight
    new_classifier_state['classifier.2.bias'][:OLD_NUM_CLASSES]   = old_final_bias
    # rows [198:231] keep their random init — these are the new Romanian identity slots

    model.load_state_dict(new_classifier_state)

    print(f"Warm-started from checkpoint (epoch {checkpoint.get('epoch', '?')}, "
          f"original val_loss {checkpoint.get('val_loss', '?')})")
    print(f"Classifier expanded from {OLD_NUM_CLASSES} to {NUM_CLASSES} classes")

    # ---- Optimizer with LOW learning rate for gentle fine-tuning ----
    params_to_train = (
        list(model.parameters()) +
        list(face_translator.parameters()) +
        list(voice_translator.parameters()) +
        list(transformer_fusion.parameters())
    )
    optimizer     = optim.Adam(params_to_train, lr=FINETUNE_LR, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    criterion     = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=5)

    best_val_loss   = float('inf')
    best_checkpoint = None

    for epoch in range(FINETUNE_EPOCHS):

        model.train()
        face_translator.train()
        voice_translator.train()
        transformer_fusion.train()
        total_train_loss = 0

        for batch_idx, (face_emb, voice_emb, labels) in enumerate(train_loader):
            if batch_idx % 200 == 0:
                print(f"  batch {batch_idx}/{len(train_loader)}")
            face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE)
            if batch_idx == 0:
                print(f"  [device check] face_emb on: {face_emb.device}, GPU memory: {torch.cuda.memory_allocated()/1e9:.3f} GB")

            face_emb  = face_translator(face_emb)
            voice_emb = voice_translator(voice_emb)

            align_loss = cross_modal_alignment_loss(face_emb, voice_emb, labels)

            fused_emb = transformer_fusion(face_emb, voice_emb)
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1)

            optimizer.zero_grad()
            logits      = model(fused_emb)
            class_loss  = criterion(logits, labels.view(-1))
            supcon_loss = supervised_contrastive_loss(fused_emb, labels)

            loss = class_loss + (0.5 * supcon_loss) + (10.0 * align_loss)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        face_translator.eval()
        voice_translator.eval()
        transformer_fusion.eval()
        total_val_loss = 0

        with torch.no_grad():
            for face_emb, voice_emb, labels in val_loader:
                face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE)

                face_emb  = face_translator(face_emb)
                voice_emb = voice_translator(voice_emb)

                align_loss = cross_modal_alignment_loss(face_emb, voice_emb, labels)

                fused_emb = transformer_fusion(face_emb, voice_emb)
                fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1)

                logits      = model(fused_emb)
                class_loss  = criterion(logits, labels.view(-1))
                supcon_loss = supervised_contrastive_loss(fused_emb, labels)

                loss = class_loss + (0.5 * supcon_loss) + (10.0 * align_loss)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = {
                'classifier':         model.state_dict(),
                'face_translator':    face_translator.state_dict(),
                'voice_translator':   voice_translator.state_dict(),
                'transformer_fusion': transformer_fusion.state_dict(),
                'epoch':      epoch + 1,
                'val_loss':   avg_val_loss,
                'num_classes': NUM_CLASSES,
                'base_checkpoint': OLD_CHECKPOINT_PATH,
            }
            torch.save(best_checkpoint, NEW_CHECKPOINT_PATH)
            print(f"  -> Best fine-tuned model saved (val_loss={avg_val_loss:.4f})")

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

    print(f"\nFine-tuning complete. Best validation loss: {best_val_loss:.4f}")
    print(f"Saved to: {NEW_CHECKPOINT_PATH}")