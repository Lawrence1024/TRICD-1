import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from Utils import *
from ImageDataset import ImageDataset
from TestModel2 import Net

from TestModel import TestModel
from Evaluator import Evaluator

# current_model = TestModel("TestModel", {})
# current_evaluator = Evaluator("Fist Evaluator")

# current_model.train()
# result_filename = current_model.generate_result()
# current_evaluator.evaulate(result_filename)


training_data = ImageDataset(annotations_file = "../tags.json", img_dir = "/home/lawrence92/TRICD/train_images", transform = transformFunction)
train_dataloader = DataLoader(training_data, batch_size = 64)

net = Net()
model_path = "../models/TestModel2.pth"
net.load_state_dict(torch.load(model_path))
predictionList = []
with torch.no_grad():
    for data in train_dataloader:
        images, solution, caption, image_id = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        print(outputs.data)
        _, predicted = torch.max(outputs.data, 1)
        predictionList.extend(predicted.tolist())


predictionDict = prediction_list_to_dict(predictionList)
path = write_dict_to_json(predictionDict, "../json_results/TestModel2.json")
evaulate(path)

