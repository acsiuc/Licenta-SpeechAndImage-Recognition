import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset  # no longer duplicated here
from models import JointClassifier, ModalityTranslator, TransformerCrossAttention
from utils import orthogonal_projection_loss, EarlyStopping, cross_modal_alignment_loss

EMBEDDING_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings"
BATCH_SIZE = 32
EPOCHS = 80
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    full_dataset = EmbeddingDataset(EMBEDDING_DIR)

    train_size = int(0.8 * len(full_dataset))
    test_size  = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model              = JointClassifier(num_classes=64, embedding_dim=512).to(DEVICE)
    face_translator    = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    voice_translator   = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE)
    transformer_fusion = TransformerCrossAttention(embed_dim=512).to(DEVICE)

    params_to_train = (
        list(model.parameters()) +
        list(face_translator.parameters()) +
        list(voice_translator.parameters()) +
        list(transformer_fusion.parameters())
    )
    optimizer     = optim.Adam(params_to_train, lr=0.001, weight_decay=1e-4)
    scheduler     = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion     = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=15)

    best_val_loss  = float('inf')
    best_checkpoint = None  # will be populated on first improvement

    for epoch in range(EPOCHS):

        # ── Training pass ──────────────────────────────────────────────────────
        model.train()
        face_translator.train()
        voice_translator.train()
        transformer_fusion.train()
        total_train_loss = 0

        for face_emb, voice_emb, labels in train_loader:
            face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE)

            face_emb  = face_translator(face_emb)
            voice_emb = voice_translator(voice_emb)

            align_loss = cross_modal_alignment_loss(face_emb, voice_emb, labels)

            fused_emb = transformer_fusion(face_emb, voice_emb)
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1)

            optimizer.zero_grad()
            logits     = model(fused_emb)
            class_loss = criterion(logits, labels.view(-1))
            ortho_loss = orthogonal_projection_loss(fused_emb, labels)

            if epoch == 0 and total_train_loss == 0:
                print(f"[Epoch 1 first batch] Class: {class_loss.item():.4f} | "
                      f"Ortho: {ortho_loss.item():.4f} | Align: {align_loss.item():.4f}")

            loss = class_loss + (0.1 * ortho_loss) + (10.0 * align_loss)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # ── Validation pass ────────────────────────────────────────────────────
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

                logits     = model(fused_emb)
                class_loss = criterion(logits, labels.view(-1))
                ortho_loss = orthogonal_projection_loss(fused_emb, labels)

                loss = class_loss + (0.1 * ortho_loss) + (10.0 * align_loss)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Scheduler and early stopping both watch VALIDATION loss now
        scheduler.step(avg_val_loss)
        early_stopper(avg_val_loss)

        # Save whenever validation improves — captures the best model, not just the last one
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint = {
                'classifier':         model.state_dict(),
                'face_translator':    face_translator.state_dict(),
                'voice_translator':   voice_translator.state_dict(),
                'transformer_fusion': transformer_fusion.state_dict(),
                'epoch':    epoch + 1,
                'val_loss': avg_val_loss,
            }
            torch.save(best_checkpoint, "model_cu_transformer.pth")
            print(f"  -> Best model saved (val_loss={avg_val_loss:.4f})")

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break

        # Periodic milestone checkpoint — saves the full dict, not just the classifier
        if (epoch + 1) % 20 == 0 and best_checkpoint is not None:
            torch.save(best_checkpoint, f"checkpoint_epoch_{epoch+1}.pth")

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")