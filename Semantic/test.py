from utils import VOCDataset, DataTransform, make_path_list
import torch.utils.data as data

# 이미지, annotation 경로 가져오기
train_img_list, train_anno_list, val_img_list, val_anno_list = make_path_list('./data/VOCdevkit/VOC2012/')

# 색의 평균값과 표준편차
color_mean = (0.485, 0.456, 0.406)
color_std = (0.229, 0.224, 0.225)

# 데이터 가져오기
train_dataset = VOCDataset(train_img_list, train_anno_list, phase='train',
                            transform=DataTransform(
                                input_size=475, color_mean=color_mean, color_std=color_std))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase='val',
                        transform=DataTransform(
                            input_size=475, color_mean=color_mean, color_std=color_std))

print(val_dataset.__getitem__(0)[0].shape) # img 사이즈
print(val_dataset.__getitem__(0)[1].shape) # annotation 사이즈

# 데이터 로더 작성
batch_size = 8

train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(val_dataset, batch_size = batch_size, shuffle=False)

dataloaders_dict = {'train':train_dataloader, 'val':val_dataloader} # 딕셔너리 형태로 저장

# 실행
batch_iterator = iter(dataloaders_dict['val']) # 반복자로 설정
images, anno_class_images = next(batch_iterator) # 하나씩 가져온다.
print(images.size())
print(anno_class_images.size())
# print(images)


print(val_dataset.__getitem__(0)[0])
print(val_dataset.__getitem__(0)[0][:,:,0])
print(val_dataset.__getitem__(0)[0][:,:,1])