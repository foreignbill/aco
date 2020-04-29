from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TL_MODEL_PATH = 'rpc_detect_model.pth'
model.load_state_dict(torch.load(TL_MODEL_PATH))
model.eval()
trans = transforms.Compose([transforms.ToTensor()])

test_img_dir = './apollo_tl_demo_data/testsets/images/'

for i in range(100):
    test_img_path = '{}/{:0>5d}.jpg'.format(test_img_dir, i)
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