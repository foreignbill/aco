import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import json

# load train dataset
def parse_label_file(label_file_path):
    label_list = []
    with open(label_file_path, 'r') as f:
        new_dict = {}
        new_dict = json.loads(f.read())
        i = 0
        for annotation in new_dict['annotations']:
            i = i + 1
            bbox = annotation['bbox']
            label_dict = {
                'supercategory': new_dict['categories'][annotation['category_id']]['supercategory'],
                'label': new_dict['categories'][annotation['category_id']]['name'],
                'xmin': int(bbox[0]),
                'ymin': int(bbox[1]),
                'xmax': int(bbox[0]) + int(bbox[2]),
                'ymax': int(bbox[1]) + int(bbox[3]),
            }
            label_list.append(label_dict)
            # remove when run
            if i >= 100:
                break
    return label_list

class RPCDataset(torch.utils.data.Dataset):
    def __init__(self, list_file_path):
        dataset_root_dir = os.path.dirname(list_file_path)
        self.dataset_root_dir = dataset_root_dir
        self.all_labels = []
        with open(list_file_path, 'r') as f:
            new_dict = json.loads(f.read())
            for annotation in new_dict['annotations']:
                bbox = annotation['bbox']
                label_dict = {
                    'supercategory': new_dict['categories'][annotation['category_id']]['supercategory'],
                    'label': new_dict['categories'][annotation['category_id']]['name'],
                    'xmin': int(bbox[0]),
                    'ymin': int(bbox[1]),
                    'xmax': int(bbox[0]) + int(bbox[2]),
                    'ymax': int(bbox[1]) + int(bbox[3]),
                }
                image_label = {
                    'path': '',
                    'boxes': label_dict,
                }
                self.all_labels.append(image_label)
        self.transforms = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.all_labels)

if __name__ == '__main__':
    print(parse_label_file('RPC-dataset/instances_train2019.json'))