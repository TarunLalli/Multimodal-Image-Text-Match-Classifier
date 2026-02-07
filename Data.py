import torch
from datasets import load_dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import random
from torchvision import transforms

test_data = load_dataset("jxie/flickr8k", split='test')
train_data = load_dataset("jxie/flickr8k",split='train')
valid_data = load_dataset("jxie/flickr8k",split='validation')

print(train_data)
print(test_data)
print(valid_data)

dataset_train_temp_caps = train_data['caption_0'][:1000]
dataset_train_temp_imgs = train_data['image'][:1000]

dataset_test_temp_caps = test_data['caption_0'][:500]
dataset_test_temp_imgs = test_data['image'][:500]

dataset_valid_temp_caps = valid_data['caption_0'][:500]
dataset_valid_temp_imgs = valid_data['image'][:500]

dataset_train = []
dataset_test = []
dataset_valid = []

tokenizer = get_tokenizer("basic_english")

#Defining iterator
def train_iter():
    for line in dataset_train_temp_caps:
        yield tokenizer(line)

#Building Vocabulary
vocab = build_vocab_from_iterator(train_iter(), specials=["<unk>"])

#Setting Vocab default
vocab.set_default_index(vocab.__getitem__('<unk>'))

for i in range(len(dataset_train_temp_caps)):
    dataset_train.append((dataset_train_temp_imgs[i],dataset_train_temp_caps[i],1))

    j = random.randint(0,len(dataset_train_temp_caps)-1)
    while j == i:
        j = random.randint(0,len(dataset_train_temp_caps)-1)

    dataset_train.append((dataset_train_temp_imgs[i],dataset_train_temp_caps[j],0))

for i in range(len(dataset_test_temp_caps)):
    dataset_train.append((dataset_test_temp_imgs[i],dataset_test_temp_caps[i],1))

    j = random.randint(0,len(dataset_test_temp_caps)-1)
    while j == i:
        j = random.randint(0,len(dataset_test_temp_caps)-1)

    dataset_test.append((dataset_test_temp_imgs[i],dataset_test_temp_caps[j],0))

for i in range(len(dataset_valid_temp_caps)):
    dataset_valid.append((dataset_valid_temp_imgs[i],dataset_valid_temp_caps[i],1))

    j = random.randint(0,len(dataset_valid_temp_caps)-1)
    while j == i:
        j = random.randint(0,len(dataset_valid_temp_caps)-1)

    dataset_valid.append((dataset_valid_temp_imgs[i],dataset_valid_temp_caps[j],0))

class Dataset(torch.utils.data.Dataset):
    def __init__(self,data,tokenizer,vocab):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return((self.image_transform(self.data[index][0]),self.text_tokeniser(self.data[1]),self.data[2]))

    def image_transform(self, image):
        transform = transforms.Compose([
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
            transforms.ToTensor()
        ])
        return transform(image)
    
    def text_tokeniser(self,text):
        return((self.vocab.__getitem__(token) for token in self.tokenizer(text)))
    
def collate_fn(batch):
    #Expecting input as follows ((img_tensori,cap_tensori,labeli), (img_tensori+1,cap_tensori+1,labeli+1), ...)
    #Expected output as follows ((img_tensori, cap_tensori),(img_tensori+1, padded_cap_tensori+1), ...), (labeli, labeli+1, ...)

    pad_id = vocab["<unk>"]

    inputs = []
    labels = []
    attention_mask = []
    l_max = len(max([
                pair[1] for pair in batch
            ], key=len))
    
    for pair in batch:
        labels.append(pair[2])
        img = pair[0]
        cap = pair[1]
        pair_mask = [1 for _ in range(len(cap))]

        while len(cap) < l_max:
            cap.append(pad_id)
            pair_mask.append(0)

        inputs.append((img,pair))
        attention_mask.append(pair_mask)

    return inputs, labels, attention_mask