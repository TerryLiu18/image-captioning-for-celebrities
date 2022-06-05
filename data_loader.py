import os
import random
import spacy
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T

from PIL import Image
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from config import DATA_LOCATION, TEST_LOCATION
from face_recognition import mtcnn, get_name


class Vocabulary:
    #tokenizer
    spacy_eng = spacy.load("en_core_web_sm")

    
    def __init__(self, freq_threshold):
        #setting the pre-reserved tokens int to string tokens
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        
        #string to int tokens
        #its reverse dict self.itos
        self.stoi = {v:k for k,v in self.itos.items()}
        
        self.freq_threshold = freq_threshold
        
        
        
    def __len__(self): return len(self.itos)
    
    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in Vocabulary.spacy_eng.tokenizer(text)]
    
    def build_vocab(self, sentence_list):
        frequencies = Counter()
        
        #staring index 4
        idx = 4
        
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1
                
                #add the word to the vocab if it reaches minum frequecy threshold
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        """ For each word in the text corresponding index token for that word form the vocab built as list """
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]    
    
class FlickrDataset(Dataset):
    """
    FlickrDataset
    """
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform
        
        #Get image and caption colum from the dataframe
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        
        #Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocab(self.captions.tolist())
        
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.captions[idx]
        img_name = self.imgs[idx]
        img_location = os.path.join(self.root_dir,img_name)
        img = Image.open(img_location).convert("RGB")
        
        #apply the transfromation to the image
        if self.transform is not None:
            img = self.transform(img)
        
        #numericalize the caption text
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(caption)
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return img, torch.tensor(caption_vec)


class TestDataset(Dataset):
    """
    test dataset no captions
    """
    def __init__(self, root_dir, caption_file=None, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        # self.df = pd.read_csv(caption_file)
        self.transform = transform
        self.len = len(os.listdir(root_dir))
        
    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        # img = Image.open('0.jpg') # image path
        # img = np.array(img)
        # img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
        # name_list = get_name(img_cropped_list, prob_list)

        # img_location = os.path.join(self.root_dir, str(idx) + ".jpg")
        # img = Image.open(img_location)
        # img2 = img.copy().convert("RGB")
        # if self.transform is not None:
        #     img2 = self.transform(img2)
        # img = np.array(img)
        # img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
        # name_list = get_name(img_cropped_list, prob_list)
        # return img2, name_list

        img_location = os.path.join(self.root_dir, str(idx) + ".jpg")
        img = Image.open(img_location).convert("RGB")
        
        if self.transform is not None:
            img = self.transform(img)
        return img

    

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self,batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs,dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs, targets


def get_data_loader(dataset, batch_size, shuffle=False, num_workers=1):
    """
    Returns torch dataloader for the flicker8k dataset
    
    Args:
        dataset: FlickrDataset
            custom torchdataset named FlickrDataset 
        batch_size: int
            number of data to load in a particular batch
        shuffle: boolean,optional;
            should shuffle the datasests (default is False)
        num_workers: int,optional
            numbers of workers to run (default is 1)  
    
    Returns:
        data_loader: torch.utils.data.DataLoader
    """

    pad_idx = dataset.vocab.stoi["<PAD>"]
    collate_fn = CapsCollate(pad_idx=pad_idx, batch_first=True)

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return data_loader


def test_data(root_dir=TEST_LOCATION, transform=None, idx=0):
    img_location = os.path.join(root_dir, str(idx) + ".jpg")
    img = Image.open(img_location)
    img2 = Image.open(img_location).convert("RGB")
    
    if transform is not None:
        img2 = transform(img2)

    img = np.array(img)
    img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    name_list = get_name(img_cropped_list, prob_list)
    print(f'test_data return type: {type(img2)}, {type(name_list)}')
    return img2, name_list


def get_test_data(root_dir=TEST_LOCATION, shuffle=False):
    testset_size = len(os.listdir(root_dir))
    seed = random.randint(0, testset_size - 1)                                         
    image, name_list = test_data(root_dir, transform=None, idx=seed)
    return image, name_list


def get_test_loader(dataset, batch_size, shuffle=False, num_workers=1):
    """get test loader
    returns are comprised of image and name_list
    
    Args:
        dataset: TestDataset
            custom torchdataset named TestDataset
        batch_size: int
            number of data to load in a particular batch
        shuffle: boolean, optional;
            should shuffle the datasests (default is False)
        num_workers: int, optional
            numbers of workers to run (default is 1)  
    
    Returns:
        data_loader: torch.utils.data.DataLoader
    """

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    return data_loader


if __name__ == '__main__':
    """test dataloader"""
    from config import *
    from utils import transforms

    testdataset = TestDataset(
        root_dir=TEST_LOCATION,
        transform=transforms
    )

    test_loader = get_test_loader(
        dataset=testdataset,
        batch_size=2,
        shuffle=True,
        num_workers=1,
    )

    img, name_list = get_test_data(TEST_LOCATION, shuffle=True)
    print(name_list)

    # dataiter = iter(test_loader)
    # images = next(dataiter)

    # img = images[0].detach().clone()
    # img1 = images[0].detach().clone().numpy()
    # print('type(img): {}'.format(type(img1)))

    # img2 = np.array(img1)
    # img_cropped_list, prob_list = mtcnn(img2, return_prob=True) 
    # print(prob_list)
    # name_list = get_name(img_cropped_list, prob_list)

    # img_location = os.path.join(self.root_dir, str(idx) + ".jpg")
    # img = Image.open(img_location)
    # img2 = img.copy().convert("RGB")
    # if self.transform is not None:
    #     img2 = self.transform(img2)
    # img = np.array(img)
    # img_cropped_list, prob_list = mtcnn(img, return_prob=True) 
    # name_list = get_name(img_cropped_list, prob_list)
    # return img2, name_list

    # img = images[0].detach().clone()
    # img1 = images[0].detach().clone()
    
    
    # caps, alphas = get_caps_from(img.unsqueeze(0))
