from utils import make_path_list, DataTransform, VOCDataset, PSPNet, PSPLoss, lambda_epoch, train_model
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 이미지, annotation 경로 가져오기
train_img_list, train_anno_list, val_img_list, val_anno_list = make_path_list('./data/VOCdevkit/VOC2012/')

net = PSPNet(n_classes=21)  

state_dict = torch.load('./weights/pspnet50_10.pth', map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)

# print(val_img_list[0])
# 임의의 이미지 하나 가져오기
img = Image.open(val_img_list[0])
img_width, img_height = img.size
# print(img_width, img_height)

# 전처리
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)
transform = DataTransform(input_size=475, color_mean=color_mean, color_std=color_std)

# annotation 이미지를 준비하여 색상 팔레트 정보 추출
anno_class_img = Image.open(val_anno_list[10])
p_palette = anno_class_img.getpalette()
# print(p_palette)
phase = 'val'

img, anno_class_img = transform(phase, img, anno_class_img) # 이미지, annotaion 전처리

# 추론
net.eval()
x = img.unsqueeze(0) # 미니 배치화 시킨다.
outputs = net(x)
y = outputs[0] # output_aux는 무시

# print(y[0].shape)
y = y[0].detach().numpy()  # [1,21,475,475] -> [21,475,475]
y = np.argmax(y, axis=0)  # 21개의 클래스 확률중 가장 큰 클래스만을 구한다.
# print(y.shape)

anno_class_img = Image.fromarray(np.uint8(y), mode='P')
anno_class_img = anno_class_img.resize((img_width, img_height), Image.NEAREST)
anno_class_img.putpalette(p_palette)
plt.imshow(anno_class_img)
plt.savefig('pallete_img.jpg')

# 이미지를 투과시켜 겹치도록 한다.
trans_img = Image.new('RGBA', anno_class_img.size, (0,0,0,0))
anno_class_img = anno_class_img.convert('RGBA')

for x in range(img_width):
    for y in range(img_height):
        pixel = anno_class_img.getpixel((x,y))
        r, g, b, a = pixel

        if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
            continue
        else:
            trans_img.putpixel((x,y),(r,g,b,150))

img = Image.open(val_img_list[0])
result = Image.alpha_composite(img.convert('RGBA'), trans_img)
plt.imshow(result)
plt.savefig('result.jpg')
