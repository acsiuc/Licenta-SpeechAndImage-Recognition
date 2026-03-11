import torch
from torch.utils.data import DataLoader
from dataset import EmbeddingDataset 
from models import JointClassifier, ModalityTranslator
from utils import paeff_fusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # use gpu if we have one
TEST_DIR = r"C:\Users\Axiuc\Downloads\mavceleb_embeddings" # path to the pre-baked .pt vectors
NUM_CLASSES = 1200 # total number of possible identities to guess

def test_model():
    print("Loading test data and models...")
    
    full_dataset = EmbeddingDataset(TEST_DIR) # load the pre-baked vectors
    train_size = int(0.8 * len(full_dataset)) # calculate 80 percent for training pile
    test_size = len(full_dataset) - train_size # the remaining 20 percent for testing pile
    generator = torch.Generator().manual_seed(42) # use a seed to get the exact same split as train.py
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size], generator=generator)

    test_loader = DataLoader(train_dataset, batch_size=32, shuffle=False) # loading the training pile for our open-book test

    model = JointClassifier(num_classes=NUM_CLASSES, embedding_dim=512).to(DEVICE) # spawn the upgraded 512d
    face_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE) # spawn the upgraded 512d 
    voice_translator = ModalityTranslator(input_dim=128, output_dim=512).to(DEVICE) # spawn the upgraded 512d 

    checkpoint = torch.load("final_model.pth", map_location=DEVICE) # open the suitcase with all our saved brains
    model.load_state_dict(checkpoint['classifier']) # upload the classifier memories
    face_translator.load_state_dict(checkpoint['face_translator']) # upload the face translator memories
    voice_translator.load_state_dict(checkpoint['voice_translator']) # upload the voice translator memories

    model.eval() # lock bouncer into exam mode
    face_translator.eval() # lock face detective so it stops learning
    voice_translator.eval() # lock voice detective

    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad(): # turn off math tracking to go fast and save memory
        for face_emb, voice_emb, labels in test_loader:
            face_emb, voice_emb, labels = face_emb.to(DEVICE), voice_emb.to(DEVICE), labels.to(DEVICE) # move data to gpu
            
            face_emb = face_translator(face_emb) # translate face into 512d space
            voice_emb = voice_translator(voice_emb) # translate voice into 512d space
            
            fused_emb = paeff_fusion(face_emb, voice_emb) # smart mix the two vectors together
            fused_emb = torch.nn.functional.normalize(fused_emb, p=2, dim=1) # normalize the fused vector so the math works
            
            logits = model(fused_emb) # makes a guess on who this is
            
            _, predicted_labels = torch.max(logits, 1) # find the button with the highest percentage guess
            
            total_samples += labels.size(0) # tally up the total people tested
            correct_predictions += (predicted_labels == labels.view(-1)).sum().item() # tally up how many we got exactly right

    accuracy = (correct_predictions / total_samples) * 100 # calculate final grae
    print(f" Total People Tested: {total_samples}")
    print(f" Correct Guesses: {correct_predictions}")
    print(f" Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    test_model()