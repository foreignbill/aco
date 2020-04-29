from torchvision import transforms
import os
import sys
import cv2
import torch
import torchvision

from rpc_dataset import RPCDataset

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 4  # 1 class (traffic light) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

from engine import train_one_epoch, evaluate
import utils

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
num_classes = 4

list_fpath = './RPC-dataset/instances_train2019.json'
dataset = RPCDataset(list_fpath)
dataset_test = RPCDataset(list_fpath)

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-20])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-80:])

# define training and validation data loaders
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=1,
    collate_fn=utils.collate_fn)

data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=1,
    collate_fn=utils.collate_fn)

model.to(device)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

from PIL import Image
import cv2
import torch
import torchvision
import numpy as np
import json

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 4  # 1 class (traffic light) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
TL_MODEL_PATH = 'rpc_detect_model.pth'
model.load_state_dict(torch.load(TL_MODEL_PATH))
model.eval()
trans = transforms.Compose([transforms.ToTensor()])

# new_dict = {}
with open('./RPC-dataset/instances_test2019.json', 'r') as f:
    new_dict = json.loads(f.read())

print(new_dict['images'][0])

test_img_dir = './RPC-dataset/test2019'
for i in range(100):
    filename = new_dict['images'][i]['file_name']
    test_img_path = '{}/{}'.format(test_img_dir, filename)
    test_img = Image.open(test_img_path).convert("RGB")
    cv_img = cv2.cvtColor(np.asarray(test_img),cv2.COLOR_RGB2BGR)
    x = [trans(test_img).to(device)]
    predictions = model(x)
    # print(predictions)
    num_boxes = predictions[0]['scores'].size()[0]
    for j in range(num_boxes):
        xmin = int(predictions[0]['boxes'][j][0].item())
        ymin = int(predictions[0]['boxes'][j][1].item())
        xmax = int(predictions[0]['boxes'][j][2].item())
        ymax = int(predictions[0]['boxes'][j][3].item())
        confidence = float(predictions[0]['scores'][j].item())
        print(xmin, ymin, xmax, ymax)
        print(confidence)
        if confidence < 0.7:
            continue
        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
    cv2.imwrite('./vis_det/{:0>5d}.jpg'.format(i), cv_img)