from utils import make_path_list, DataTransform, VOCDataset, PSPNet, PSPLoss, lambda_epoch, train_model
import torch

# 이미지, annotation 경로 가져오기
train_img_list, train_anno_list, val_img_list, val_anno_list = make_path_list('./data/VOCdevkit/VOC2012/')

net = PSPNet(n_classes=21)  

state_dict = torch.load('./weights/pspnet50_10.pth', map_location={'cuda:0': 'cpu'})
net.load_state_dict(state_dict)

