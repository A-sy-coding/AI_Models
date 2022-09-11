from utils import make_path_list, DataTransform, VOCDataset, PSPNet, PSPLoss, lambda_epoch
import torch.utils.data as data
import torch
import torch.nn as nn
import torch.optim as optim

# 이미지, annotation 경로 가져오기
train_img_list, train_anno_list, val_img_list, val_anno_list = make_path_list('./data/VOCdevkit/VOC2012/')

# 색의 평균값과 표준편차
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

# 데이터 전처리 하기
train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train', 
                            transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))  
val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val', 
                            transform=DataTransform(input_size=475, color_mean=color_mean, color_std=color_std))

# dataloader 설정
batch_size = 8

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader}  # 데이터로더들을 딕셔너리 형태로 저장

# 처음에는 사전학습된 데이터가 150클래스로 학습되어 있기 때문에 해당 모델을 불러온뒤
# 추후, 새로 학습을 진행할 때는 Decoder와 AuxLoss의 클래스를 21로 변경하여 진행하도록 한다
net = PSPNet(n_classes=150)

state_dict = torch.load('./weights/pspnet50_ADE20K.pth')
net.load_state_dict(state_dict, strict=False)

n_classes = 21 # 다시 21클래스로 변경
net.decode_feature.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)
net.aux.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

# 교체한 합성곱 충을 초기화하도록 한다.(이전 합성곱 층은 150클래스에 적합하게 학습된 모델이기 때문에 학습가중치를 초기화해야 한다.)
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:  # bias가 존재하면 0으로 초기화
            nn.init.constant_(m.bias, 0.0)
net.decode_feature.classification.apply(weights_init)
net.aux.classification.apply(weights_init)

print('네트워크 설정 완료!!')

criterion = PSPLoss(aux_weight=0.4)
optimizer = optim.SGD([
    {'params' : net.feature_conv.parameters(), 'lr' : 1e-3},
    {'params' : net.feature_res_1.parameters(), 'lr': 1e-3},
    {'params' : net.feature_res_2.parameters(), 'lr': 1e-3},
    {'params' : net.feature_dilated_res_1.parameters(), 'lr': 1e-3},
    {'params' : net.feature_dilated_res_2.parameters(), 'lr': 1e-3},
    {'params' : net.pyramid_pooling.parameters(), 'lr': 1e-3},
    {'params' : net.decode_feature.parameters(), 'lr': 1e-2},
    {'params' : net.aux.parameters(), 'lr' : 1e-2},
], momentum=0.9, weight_decay=0.0001)

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_epoch)