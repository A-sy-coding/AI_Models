from .data_argument import Compose, Scale, RandomRotation, RandomMirror, Resize, Normalize_Tensor
import numpy as np
import torch
import torch.utils.data as data
import os, cv2
import matplotlib.pyplot as plt
from PIL import Image

def make_path_list(rootpath):
    '''
    데이터 값들의 경로를 저장한 리스트를 반환한다.
    Args:
        rootpath(str) : 데이터 경로
    Returns:
        경로를 저장한 리스트
    '''

    # 이미지 파일과 annotation 경로 지정
    img_template = os.path.join(rootpath, 'JPEGImages', '%s.jpg')
    anno_template = os.path.join(rootpath, 'SegmentationClass', '%s.png')

    # 훈련, 검증 데이터의 파일이름 설정(Segmentation)
    train_id_names = os.path.join(rootpath, 'ImageSets/Segmentation/train.txt')
    val_id_names = os.path.join(rootpath, 'ImageSets/Segmentation/val.txt')

    # 훈련 데이터의 이미지 파일과 annotation 파일 경로 저장할 리스트 생성
    train_img_list, train_anno_list = [], []

    for line in open(train_id_names):
        file_id = line.strip()
        img_path = (img_template % file_id)  # img_tempalte를 지정할 때 사용한 %s에 file_id가 들어가게 된다.
        anno_path = (anno_template % file_id)
        train_img_list.append(img_path)
        train_anno_list.append(anno_path)

    # 검증 데이터의 이미지 파일과 annotation 파일 경로 저장할 리스트 생성
    val_img_list, val_anno_list = [], []
    
    for line in open(val_id_names):
        file_id = line.strip()
        img_path = (img_template % file_id)
        anno_path = (anno_template % file_id)
        val_img_list.append(img_path)
        val_anno_list.append(anno_path)

    return train_img_list, train_anno_list, val_img_list, val_anno_list

class DataTransform():
    '''
    Description:
        이미지와 annotation의 전처리 클래스 
        훈련시 이미지 확장 및 전처리 수행
        검정시 이미지 전처리만 수행
    Args:
        input_size(int) : resize 크기
        color_mean(R, G, B) : 각 채널의 평균값
        color_std(R, G, B) : 각 채널의 표준편차
    '''
    def __init__(self, input_size, color_mean, color_std):
        self.data_transform = {
            'train' : Compose([
                Scale(scale=[0.5,1.5]), # 이미지 확대 및 축소
                RandomRotation(angle=[-10, 10]), # 이미지 회전
                RandomMirror(), # 50%확률로 이미지 좌우반전
                Resize(input_size), 
                Normalize_Tensor(color_mean, color_std)
            ]),
            'val' : Compose([ # val은 데이터 증상 수행 x
                Resize(input_size),
                Normalize_Tensor(color_mean, color_std)
            ])
        }
    
    def __call__(self, phase, img, anno_class_img):
        return self.data_transform[phase](img, anno_class_img)

class VOCDataset(data.Dataset):
    '''
    Description:
        dataset을 만드는 클래스
    Args:
        img_list(리스트) : 이미지 경로를 저장한 리스트
        anno_list(리스트) : annotation 경로를 저장한 리스트
        phase('train' or 'val') : 훈련 or 검증 설정
        transform(클래스) : 전처리 클래스 인스턴스
    '''
    def __init__(self, img_list, anno_list, phase, transform):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase
        self.transform = transform
    
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        ''' 전처리한 이미지, annotation 데이터 얻기 '''
        img, anno_class_img = self.pull_item(index)
        return img, anno_class_img

    def pull_item(self, index):
        ''' 실제 이미지, annotation 전처리 수행 '''
        
        # 이미지 읽기
        image_file_path = self.img_list[index]
        img = Image.open(image_file_path)

        # annotation 읽기
        anno_file_path = self.anno_list[index]
        anno_class_img = Image.open(anno_file_path)

        # 전처리 수행
        img, anno_class_img = self.transform(self.phase, img, anno_class_img)

        return img, anno_class_img





