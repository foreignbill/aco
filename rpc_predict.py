from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
import torchvision
import numpy as np
import json

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

num_classes = 4  # 1 class (traffic light) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
TL_MODEL_PATH = 'rpc_detect_model9.pth'
model.load_state_dict(torch.load(TL_MODEL_PATH))
model.to(device)
model.eval()
trans = transforms.Compose([transforms.ToTensor()])

new_dict = {}
with open('./RPC-dataset/instances_test2019.json', 'r') as f:
    new_dict = json.loads(f.read())

categories = new_dict['categories']
chinese_names = new_dict['__raw_Chinese_name_df']
print(categories[0])

test_img_dir = './RPC-dataset/test2019'
for i in range(100):
    filename = new_dict['images'][i]['file_name']
    test_img_path = '{}/{}'.format(test_img_dir, filename)
    test_img = Image.open(test_img_path).convert("RGB")
    cv_img = cv2.cvtColor(np.asarray(test_img),cv2.COLOR_RGB2BGR)
    x = [trans(test_img).to(device)]
    predictions = model(x)
    print(predictions)
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
        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        category_id = int(predictions[0]['labels'][j]) - 1
        # cv2.putText(cv_img, chinese_names[category_id]['name'], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        cv_img = cv2ImgAddText(cv_img, chinese_names[category_id]['name'], xmin, ymin, (0, 255, 0), 12)

    print('./output/{:0>5d}.jpg'.format(i))
    cv2.imwrite('./output/{:0>5d}.jpg'.format(i), cv_img)