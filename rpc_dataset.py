import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import json

class RPCDataset(torch.utils.data.Dataset):
    def __init__(self, list_file_path):
        dataset_root_dir = os.path.dirname(list_file_path)
        self.dataset_root_dir = dataset_root_dir
        self.all_labels = []
        with open(list_file_path, 'r') as f:
            new_dict = json.loads(f.read())
            for annotation in new_dict['annotations']:
                # if annotation['category_id'] >= 2:
                #     continue
                bbox = annotation['bbox']
                label_dict = {
                    'supercategory': new_dict['categories'][annotation['category_id'] - 1]['supercategory'],
                    'label': new_dict['categories'][annotation['category_id'] - 1]['name'],
                    'xmin': int(bbox[0]),
                    'ymin': int(bbox[1]),
                    'xmax': int(bbox[0]) + int(bbox[2]),
                    'ymax': int(bbox[1]) + int(bbox[3]),
                }
                json2path_dict = {
                    "./RPC-dataset/instances_train2019.json": "train2019",
                    "./RPC-dataset/instances_test2019.json": "test2019",
                    "./RPC-dataset/instances_val2019.json": "val2019"
                }
                image_label = {
                    'path': os.path.join("RPC-dataset", json2path_dict[list_file_path], new_dict['images'][annotation['image_id'] - 1]['file_name']),
                    'boxes': label_dict,
                }
                self.all_labels.append(image_label)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        img_path = self.all_labels[index]['path']
        img = Image.open(img_path).convert("RGB")

        boxes = []
        box = self.all_labels[index]['boxes']
        xmin = box['xmin']
        ymin = box['ymin']
        xmax = box['xmax']
        ymax = box['ymax']
        boxes.append([xmin, ymin, xmax, ymax])
        num_boxes = len(boxes)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_boxes,), dtype=torch.int64)
        image_id = torch.tensor([index])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_boxes,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        return self.transforms(img), target

    def __len__(self):
        return len(self.all_labels)

if __name__ == "__main__":
    dataset = RPCDataset('RPC-dataset/instances_train2019.json')
    print(dataset.__getitem__(0))