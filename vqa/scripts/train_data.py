import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from ImageDataset import ImageDataset 
from torchvision.transforms import ToTensor
from torchvision import transforms
from PIL import Image
from TestModel2 import Net
from Utils import *
# torch.manual_seed(17)

training_data = ImageDataset(annotations_file = "../tags.json", img_dir = "/home/lawrence92/TRICD/train_images", transform = transformFunction)
train_dataloader = DataLoader(training_data, batch_size = 10, shuffle = True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# image, solution, caption, image_id = next(iter(train_dataloader))
# print(f"Image shape: {image.size()}")
# print(f"solution: {solution}")
# print(f"caption: {caption}")
# print(f"image_id: {image_id}")

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
running_loss = 0.0
for i, data in enumerate(train_dataloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    image, solution, caption, image_id = data
    captionTensor = processWords(caption).unsqueeze(dim = 1).unsqueeze(dim = 3)
    # zero the parameter gradients
    optimizer.zero_grad()
    # forward + backward + optimize
    print("image size:", image.size())
    print("captionTensor size:", captionTensor.size())
    # inputs = torch.cat((image, captionTensor), dim = 3)
    inputs = image
    outputs = net(inputs)
    loss = criterion(outputs, solution)
    loss.backward()
    optimizer.step()
    # print statistics
    running_loss += loss.item()
    # if i % 2000 == 1999:    # print every 2000 mini-batches
    #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
    #     running_loss = 0.0

print('Finished Training')
PATH = '../models/' + str(net) + ".pth"
torch.save(net.state_dict(), PATH)

