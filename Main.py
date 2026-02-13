import torch
import torch.nn as nn
import Data
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Next 1hr targets:
    # Sort out dimensions of image and text embeddings so tht they can be concatenated
    # need to apply flatten(1) to image embedding to get rid of extra singleton dimensions


class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiating pretrained Sentence BERT style encoder
        self.transformer_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        # Instantiating pretrained ResNet
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        # Removing prediction head from ResNet
        self.resnet = nn.Sequential(*list(model.children())[:-1])
        # Turning off gradients for params to freeze encoders (No Fine tuning for this project)
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        # Instantiating trainable prediction head for combined embeddings
        self.classificationHead = nn.Sequential(nn.Linear(in_features=896, out_features=256), nn.ReLU(), nn.Linear(in_features=256, out_features=2))

    def forward(self, images, captions):
        # We expect an input to be a tuble of the form (img tensor, capt tensor)
            # Where img tensor is the normalised (C,W,D) tensor and capt is a list of each word/punct in a sentence
        img_embeddings = self.resnet(images) #Ouptut dimension: 512
        img_embeddings = img_embeddings.flatten(1) #Removing Singleton Dimensions
        text_embeddings = self.transformer_encoder.encode(
                                captions,
                                convert_to_tensor=True,
                                device=images.device
                            ) #Output dimension: 384
        # Normalising each embedings vectors before concatenation
        img_embed_norm = img_embeddings/torch.norm(img_embeddings)
        text_embed_norm = text_embeddings/torch.norm(text_embeddings)
        # Combining embeddings
        comb_embed = torch.concat((img_embed_norm,text_embed_norm), dim = 1) #Dimension: 896
        # Passing through MLP head
        pred_logits = self.classificationHead(comb_embed)
        # returning logits
        return pred_logits

def training_loop(epochs, dataloader, model, device):
    model.to(device)
    # Instantiating optimiser
    optimiser = torch.optim.Adam(params=model.parameters())
    # Iterating over each epoch
    for epoch in range(epochs):
        # setting model to training mode
        model.train()
        # Setting up loop
        loop = tqdm(dataloader, leave= True)
        # Iterating over each batch
        for batch in loop:
            # Zeroing batch gradients to stop gradient stacking over batches
            optimiser.zero_grad()
            # Passing values to model
            input_img, input_cap, labels = torch.stack(batch[0], dim=0).to(device), batch[1], torch.tensor(batch[2], dtype=torch.float32).to(device)
            outputs = model(input_img, input_cap)
            # Loss calc and back prop
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            # Updating parameters
            optimiser.step()

        #loop.set_description(f"Epoch [{epoch+1}/{epochs}]")

    print("Training Complete.")
    return model

def main():
    train_dataset = Data.Dataset(Data.dataset_train, tokenizer=Data.tokenizer, vocab=Data.vocab)
    test_dataset = Data.Dataset(Data.dataset_test, tokenizer=Data.tokenizer, vocab=Data.vocab)
    valid_dataset = Data.Dataset(Data.dataset_valid, tokenizer=Data.tokenizer, vocab=Data.vocab)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=50, collate_fn = Data.collate_fn)

    training_loop(epochs=1, dataloader=train_dataloader, model = MultiModalModel(), device='mps')

    test_dataloader = DataLoader(dataset=test_dataset, batch_size=25, collate_fn = Data.collate_fn)

if __name__ == '__main__':
    main()