import math
import torch
from pathlib import Path
from torchvision import datasets
import multiprocessing
import numpy as np
import pandas as pd
from .helpers import compute_mean_and_std, get_data_location, seed, every_s
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as VT
from torch.utils.data import default_collate
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import os
import torchaudio.transforms as AT
import torchaudio
import random


VALID_SIZE=0.2
df = pd.read_csv('data/metadata.csv',dtype={'class':'category'})
df['target']=df['class'].cat.codes.astype(np.int64)
video_classes = df[['video_dir', 'class']].drop_duplicates()

# Perform the stratified split
train_videos, valid_videos = train_test_split(video_classes, stratify=video_classes['class'], test_size=VALID_SIZE, random_state=seed)

# Get the lists of video directories for train and validation sets
train_video_dirs = train_videos['video_dir'].tolist()
valid_video_dirs = valid_videos['video_dir'].tolist()

# Assign the split video directories back to the data points
train_df = df[df['video_dir'].isin(train_video_dirs)]
valid_df = df[df['video_dir'].isin(valid_video_dirs)]

#--------------------------------------AUDIO-------------------------------------------------

input_freq=48000  
resample_freq= 8000    
class  AudRAVDESSDataset(torch.utils.data.Dataset):
    def __init__(self,df,transforms=None):
        self.df=df
        self.transforms = transforms
        self.classes=list(df['class'].cat.categories)




    def __getitem__(self, idx):
        # load images 
        target = self.df.iloc[idx]['target']
        audio, sample_rate = torchaudio.load(self.df.iloc[idx]['video_dir']+'/raw_audio.mp3')

        # make it mono
        if audio.shape[0]!=1:
            audio=torch.mean(audio, dim=0,keepdim=True)

        if self.transforms:
            audio = self.transforms(audio)

        return audio , target

    def __len__(self):
        return len(self.df)




def audio_data_loaders(
    batch_size: int = 32, num_classes: int =1000,valid_size=0.2, num_workers: int = -1, limit: int = -1, is_mel=True, is_image=True
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")


    class RandomApply(torch.nn.Module):
        def __init__(self, transform, p=0.5):
            super().__init__()
            self.transform = transform
            self.p = p

        def forward(self, waveform):
            if random.random() < self.p:
                return self.transform(waveform)
            return waveform

    class RandomChoice(torch.nn.Module):
        def __init__(self, transforms):
            super().__init__()
            self.transforms = transforms

        def forward(self, waveform):
            transform = random.choice(self.transforms)
            return transform(waveform)


    n_mels=256
    n_fft=512
    data_transforms = {
        "train": torch.nn.Sequential(
            AT.Resample(orig_freq=input_freq, new_freq=resample_freq),
            RandomApply(AT.Vol(1.5), p=0.5),  # Randomly increase volume by 1.5x
            RandomApply(AT.Fade(fade_in_len=1000, fade_out_len=1000), p=0.5),  # Randomly fade in and out
            # RandomApply(AT.TimeStretch(fixed_rate=1.1), p=0.5),  # Randomly speed up by 20%
            RandomApply(AT.FrequencyMasking(freq_mask_param=15), p=0.5),  # Randomly mask a frequency band
            RandomApply(AT.TimeMasking(time_mask_param=35), p=0.5),  # Randomly mask a time band
            # AT.MelSpectrogram(resample_freq, n_mels=n_mels,n_fft=n_fft),
            # AT.MFCC(sample_rate=input_freq, n_mfcc=256,
            #         melkwargs={
            # "n_fft": n_fft,
            #     "n_mels": n_mels,})
                ),

        "valid": torch.nn.Sequential(
            AT.Resample(orig_freq=input_freq, new_freq=resample_freq),
            # AT.MelSpectrogram(resample_freq,n_mels=n_mels,n_fft=n_fft),
            #             AT.MFCC(sample_rate=input_freq, n_mfcc=256,
            #         melkwargs={
            # "n_fft": n_fft,
            #     "n_mels": n_mels,})
        ),
    }



    train_data = AudRAVDESSDataset(train_df, transforms=data_transforms['train'])
    valid_data = AudRAVDESSDataset(valid_df, transforms=data_transforms['valid'])
   

    def collate_fn(batch):
        # [(data, target), (data, target), .... (data, target)]

        max_length=max([tensor.shape[-1] for tensor,target in batch])
        tensors = [[torch.nn.functional.pad(tensor,(max_length-tensor.shape[-1],0),value=0.0), target] for tensor,target in batch]

        return default_collate(tensors) 

    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        #sampler=sampler,
     collate_fn=collate_fn,
#        pin_memory=True,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return data_loaders

#--------------------------------------Video-------------------------------------------------       
class  VidRAVDESSDataset(torch.utils.data.Dataset):
    def __init__(self,df, is_image=True, is_mel=True,transforms=None):
        self.df=df
        self.is_image = is_image
        self.is_mel = is_mel
        self.transforms = transforms
        self.classes=list(df['class'].cat.categories)




    def __getitem__(self, idx):
        # load images 
        target = self.df.iloc[idx]['target']
        if self.is_image:
            img_path = self.df.iloc[idx]['image_path']
            img = read_image(img_path)
            
            if self.transforms:
                img = self.transforms(img)
            if not self.is_mel:
                return img, target


        


        if self.is_mel:
            mel_spec_path = self.df.iloc[idx]['mel_spec_path']
            mel_spec = np.load(mel_spec_path)
            if not self.is_image:
                return mel_spec[:int(every_s*audio_sr)][None,], target

        

        data=[img, mel_spec[:int(every_s*audio_sr)][None,]]
        # print(data.shape)

        return data, target

    def __len__(self):
        return len(self.df)



#-------------------------------------IMAGES---------------------------------------------------------   

train_im_size = 200
valid_im_size = 230
def im_data_loaders(
    batch_size: int = 32, num_classes: int =1000,valid_size=0.2, num_workers: int = -1, limit: int = -1, is_mel=True, is_image=True
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")


    cutmix = VT.CutMix(alpha=1.0 ,num_classes=num_classes)
    mixup = VT.MixUp(alpha=0.2 ,num_classes=num_classes)
    cutmix_or_mixup = T.RandomChoice([cutmix, mixup])
    data_transforms = {
        "train": VT.Compose([
            VT.ToImage(),
            # VT.RandomVerticalFlip(p=0.3),
            # VT.RandomRotation(30),

            VT.Resize((train_im_size+2,train_im_size+2), antialias=True),
            #VT.RandomResizedCrop(train_im_size, antialias=True),
            VT.RandomHorizontalFlip(),
            VT.TrivialAugmentWide(),
            #VT.RandAugment(num_ops=3),
            #VT.ColorJitter(),
            #VT.RandomPosterize(bits=1),
            #VT.RandomAffine(degrees=(1, 70), translate=(0.1, 0.3), scale=(0.6, 1)),
            #VT.RandomPerspective(distortion_scale=0.3, p=0.5),
            #VT.RandomAutocontrast(),
            VT.CenterCrop(train_im_size),
            VT.ToDtype(torch.float32, scale=True),
            VT.Normalize(mean=mean, std=std),# the mean and std are already scaled(as helper.py) no need to rescale them
            VT.RandomErasing(p=0.1),
            


        ]),
        "valid": VT.Compose([
            VT.ToImage(),
            VT.Resize((valid_im_size+2,valid_im_size+2),antialias=True),
            VT.CenterCrop(valid_im_size),
            VT.ToDtype(torch.float32, scale=True),
            VT.Normalize(mean=mean, std=std),
            
        ]),
        "test": VT.Compose([
            VT.ToImage(),
            # VT.Resize(230),
            # VT.CenterCrop(224),
            VT.ToDtype(torch.float32, scale=True),
            VT.Normalize(mean=mean, std=std),
        ]),
    }



    #  Create train and validation datasets
    train_data = RAVDESSDataset(train_df,is_image=is_image, is_mel=is_mel, transforms=data_transforms['train'])
    valid_data = RAVDESSDataset(valid_df,is_image=is_image, is_mel=is_mel, transforms=data_transforms['valid'])
   
    # prepare data loaders

    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        #sampler=sampler,
     #collate_fn=collate_fn,
#        pin_memory=True,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """


    dataiter  = iter(data_loaders['train'])
    images, labels = next(dataiter)

    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)


    class_names  = data_loaders['train'].dataset.classes
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        ax.set_title(class_names[labels[idx].item()])
