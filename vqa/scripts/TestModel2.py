import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(238144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.name = "TestModel2"

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def __str__(self):
        return f"{self.name}"
    
    def generate_result(self, file_name = None) -> str:
        self.result_dict = {}
        
        if not file_name:
            file_name = f"../json_results/{self.name}_response.json"
            
        self.write_result(file_name)
        return file_name
        
    
    def write_result(self, file_name):
        json_object = json.dumps(self.result_dict, indent=4)
        with open(file_name, "w+") as outfile:
            outfile.write(json_object)


#net = Net()