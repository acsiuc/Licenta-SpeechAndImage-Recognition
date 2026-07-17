import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset  # no longer duplicated here
from models import JointClassifier, ModalityTranslator, TransformerCrossAttention
from utils import supervised_contrastive_loss, EarlyStopping, cross_modal_alignment_loss

EMBEDDING_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings"
# NOTE: this folder name doesnt match extract_features_all.py's OUTPUT_DIR ("mavceleb_all_embeddings")
# either theres a manual rename/copy step between the two scripts, or one of these paths is stale
# check which folder actually exists on disk before presenting, this is the kind of thing a committee member will screenshot
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
    # this is the sample-level 80/20 split, seed fixed for reproducibility
    # note this splits at the SAMPLE level not the identity level, both train and val see the same 198 identities,
    # just different specific face/voice draws — this is the "sample memorization" check, not a generalization test
    # Urdu/Romanian evals later are the actual generalization tests (unseen language / unseen identity)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
    # shuffle=False on val is correct, order doesnt matter for eval, no reason to pay the shuffle cost

    model              = JointClassifier(num_classes=198, embedding_dim=512).to(DEVICE)
    face_translator = ModalityTranslator(input_dim=512, output_dim=512).to(DEVICE)
    voice_translator   = ModalityTranslator(input_dim=192, output_dim=512).to(DEVICE)
    transformer_fusion = TransformerCrossAttention(embed_dim=512).to(DEVICE)

    params_to_train = (
        list(model.parameters()) +
        list(face_translator.parameters()) +
        list(voice_translator.parameters()) +
        list(transformer_fusion.parameters())
    )
    # this is the fix for the old bug — all FOUR trainable components are in the optimizer here
    # earlier version of this codebase left the translators out of this list entirely,
    # meaning the optimizer was only ever touching frozen-encoder noise
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
            # align_loss is computed HERE, on the pre-fusion translator outputs, each modality still separate

            align_loss = cross_modal_alignment_loss(face_emb, voice_emb, labels)

            fused_emb = transformer_fusion(face_emb, voice_emb)
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1)
            # re-normalize after fusion since averaging two unit vectors doesnt guarantee unit length back out

            optimizer.zero_grad()
            logits     = model(fused_emb)
            class_loss = criterion(logits, labels.view(-1))
            supcon_loss = supervised_contrastive_loss(fused_emb, labels)
            # supcon is computed on the POST-fusion embedding, different stage than align_loss
            # so the three losses arent all looking at the same representation:
            # align_loss = pre-fusion, per-modality / class_loss + supcon_loss = post-fusion, joint

            loss = class_loss + (0.5 * supcon_loss) + (10.0 * align_loss)
            # this is the exact weighting from the thesis: CE x1.0, InfoNCE x10.0, SupCon x0.5
            # InfoNCE dominates on purpose, cross-modal alignment is the harder problem than classification
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
                supcon_loss = supervised_contrastive_loss(fused_emb, labels)

                loss = class_loss + (0.5 * supcon_loss) + (10.0 * align_loss)
                total_val_loss += loss.item()
                # validation pass mirrors training exactly, just no backward/step, and modules are in .eval() mode
                # so batchnorm uses running stats and dropout is off, standard and correct

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
            torch.save(best_checkpoint, "model_vechi.pth")
            # NOTE: this filename is "model_cu_transformer.pth"
            # test_urdu.py loads "model_arcface.pth" — different name
            # either theres a rename step outside this repo or this needs to be reconciled before the live demo,
            # this is the single most likely thing to break your live run if not checked ahead of time
            print(f"  -> Best model saved (val_loss={avg_val_loss:.4f})")

        if early_stopper.early_stop:
            print("Early stopping triggered.")
            break
            # note: EarlyStopping and the val_loss improvement check above are independent (see utils.py comment)
            # its possible to break here on the same epoch a new best was just saved, that's fine and expected,
            # just worth knowing theyre not the same mechanism if asked

        # Periodic milestone checkpoint — saves the full dict, not just the classifier
        if (epoch + 1) % 20 == 0 and best_checkpoint is not None:
            torch.save(best_checkpoint, f"checkpoint_epoch_{epoch+1}.pth")
            # this saves the LAST best_checkpoint again under a milestone name, not necessarily this exact epoch's model
            # if val hasnt improved in the last few epochs, this just re-saves the same weights as before

    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")