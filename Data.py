import torch
from datasets import load_dataset
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import os
import random
from torchvision import transforms
from torchvision.transforms import functional


test_data = load_dataset("jxie/flickr8k", split='test')
train_data = load_dataset("jxie/flickr8k",split='train')
valid_data = load_dataset("jxie/flickr8k",split='validation')

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
        return((self.image_transform(self.data[index][0]),self.data[index][1],self.data[index][2]))

    def image_transform(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.functional.pil_to_tensor,
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            )
        ])
        return transform(image)
    
def collate_fn(batch):
    #Expecting input as follows ((img_tensor_i, caption_string_list_i, label_i),(...),...)
    #Expected Output as follows ((img_tensor_i,...), (caption_string_list_i,...), (label_i,...)

    images = []
    captions = []
    labels = []
    
    for set in batch:
        labels.append(set[2])
        captions.append(set[1])
        images.append(set[0])

    return (images, captions, labels)
