import os
import pandas as pd
import json
from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from PIL import Image


class ImageDataset(Dataset):    
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        with open(annotations_file, 'r') as openfile:
            self.img_labels_dict = json.load(openfile)
        self.answers = self.img_labels_dict["annotations"]
        self.questions = self.img_labels_dict["questions"]
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels_dict["annotations"])

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.questions[idx]["file_name"])
        # image = read_image(img_path)
        image = Image.open(img_path)
        solution = self.answers[idx]["answer"]
        caption = self.questions[idx]["question"]
        image_id = self.questions[idx]["image_id"]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, solution, caption, image_id
    