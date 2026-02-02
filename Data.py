import torch
import torchvision

trainset = torchvision.datasets.Flickr8k(
        root='./Data', ann_file='./captions.txt'
    )

print(trainset)

#flickr_dataload = torch.utils.data.DataLoader(batch_size = )