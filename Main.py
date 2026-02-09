import torch
import torch.nn as nn
import Data
from sentence_transformers import SentenceTransformer
import tqdm

# Next 1hr targets:
    # Determine how cross loss entroy works (what inputs and target tensors does it take)
    # Finish training loop

class MultiModalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Instantiating pretrained Sentence BERT style encoder
        self.transformer_encoder = SentenceTransformer("all-MiniLM-L6-v2")
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

    def forward(self, input):
        # We expect an input to be a tuble of the form (img tensor, capt tensor)
            # Where img tensor is the normalised (C,W,D) tensor and capt is a list of each word/punct in a sentence
        img_embeddings = self.resnet(input[0]) #Ouptut dimension: 512
        text_embeddings = self.transformer_encoder(input[1]) #Output dimension: 384
        # Normalising each embedings vectors before concatenation
        img_embed_norm = img_embeddings/torch.norm(img_embeddings)
        text_embed_norm = text_embeddings/torch.norm(text_embeddings)
        # Combining embeddings
        comb_embed = torch.concat((img_embed_norm,text_embed_norm), dim = 0) #Dimension: 896
        # Passing through MLP head
        pred_logits = self.classificationHead(comb_embed)
        # returning logits
        return pred_logits

def training_loop(epochs, dataloader, model):
    for epoch in range(len(epochs)):
        model.train()
        loop = tqdm(model, leave= True)
        for batch in loop:
            inputs, labels = batch[0], batch[1]



def main():
    ...

if __name__ == '__main__':
    main()