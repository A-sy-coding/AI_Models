from utils import make_path_list, DataTransform, VOCDataset
from torch.utils.data as data

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

