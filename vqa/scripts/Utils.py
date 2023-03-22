import torch
import json
import os
from torchvision import transforms
def transformFunction(img):
    img = img.resize((500, 500))
    trans = transforms.Compose([transforms.ToTensor()])
    return trans(img)

def processWords(words):
    max_l = 0
    ts_list = []
    for w in words:
        ts_list.append(torch.ByteTensor(list(bytes(w, 'utf8'))))
        max_l = max(ts_list[-1].size()[0], max_l)

    w_t = torch.zeros((len(ts_list), max_l), dtype=torch.uint8)
    for i, ts in enumerate(ts_list):
        w_t[i, 0:ts.size()[0]] = ts
    return w_t

def prediction_list_to_dict(prediction_list):
    result_dict = {}
    with open('../tags.json', 'r') as openfile:
        tags_dict = json.load(openfile)
    
    for i in range(len(tags_dict["questions"])):
        tag = tags_dict["questions"][i]
        prediction = prediction_list[i]
        result_dict[tag["image_id"]] = prediction
    
    return result_dict

def write_dict_to_json(result_dict, file_name):
    json_object = json.dumps(result_dict, indent=4)
    with open(file_name, "w+") as outfile:
        outfile.write(json_object)
    return file_name

def evaulate(result_filename):
    cmd = "python ../evaluation/main.py --results_file " + result_filename
    os.system(cmd)