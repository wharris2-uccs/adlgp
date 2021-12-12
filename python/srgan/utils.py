import cv2 as cv2
import glob as glob
import numpy as np
import os as os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from natsort import natsorted

def alter_image(img, alpha, beta, pair = None):
    if pair is not None:
        alpha = random.uniform(alpha, pair[0])
        beta = int(random.uniform(beta, pair[1]))
        if beta % 2 != 1:
            beta += 1

    # Add noise.
    noise = np.random.normal(loc=0, scale=1, size=img.shape).astype('float32')
    img = cv2.addWeighted(img, alpha, noise, 1 - alpha, 0, dtype=cv2.CV_32F)

    # Gaussian blur.
    img = cv2.GaussianBlur(img, (beta, beta), 0)

    return torch.from_numpy(np.asarray(img).transpose(2, 0, 1))

def clear_dir(path):
    if os.path.exists(path):
        filelist = glob.glob(path + '/*')
        for f in filelist:
            try:
                os.remove(f)
            except:
                if len(os.listdir(f)) > 0:
                    print('rm ' + f)
                    clear_dir(f)
    else:
        os.mkdir(path)

def normalize_images(images):
    return (np.array(images) - np.array(images).min(0)) / np.array(images).ptp(0)

class CustomDataSet(Dataset):
    def __init__(self, main_dir, transform):
        self.main_dir = main_dir
        self.transform = transform
        all_imgs = os.listdir(main_dir)
        all_imgs.sort()
        self.total_imgs = natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image