from utils import VOCDataset, DataTransform, make_path_list

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