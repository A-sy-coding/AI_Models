# 데이터 증상 처리 함수들 구현

import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageOps, ImageFilter

class Compose(object):
    '''
    Args:
        transforms(list) : 전처리를 수행하는 단계들이 리스트 형태로 들어가 있다.
    '''
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, anno_class_img):
        for t in self.transforms:
            img, anno_class_img = t(img, anno_class_img)
        return img, anno_class_img

class Scale(object):
    ''' 이미지 스케일링 -> 이미지를 확대 및 축소 '''
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img, anno_class_img):
        width = img.size[0]
        height = img.size[1]

        # 확대 비율 설정 (random)
        scale = np.random.uniform(self.scale[0], self.scale[1])

        scaled_w = int(width * scale)
        scaled_h = int(height * scale)

        # 이미지 리사이즈 -> 리사이즈 하고, bicubic으로 interpolation 하기
        img = img.resize((scaled_w, scaled_h), Image.BICUBIC)
        
        # annotation 리사이즈
        anno_class_img = anno_class_img.resize((scaled_w, scaled_h),
                                                Image.NEAREST)
        
        # 리사이즈 후 원래 원본 크기로 이미지 crop
        if scale > 1.0:
            left = scaled_w - width
            left = int(np.random.uniform(0, left))

            top = scaled_h - height
            top = int(np.random.uniform(0, top))

            img = img.crop((left, top, left+width, top+height))
            anno_class_img = anno_class_img.crop((left, top, left+width, top+height))
        else:
            # 리사이즈 후 이미지가 축소되면 패딩으로 원본 크기로 채운다.
            p_palette = anno_class_img.copy().getpalette()

            img_original = img.copy()
            anno_class_img_original = anno_class_img.copy()

            pad_width = width - scaled_w
            pad_width_left = int(np.random.uniform(0, pad_width)) # 시작점을 랜덤으로 정하는것??
            
            pad_height = height - scaled_h
            pad_height_top = int(np.random.uniform(0, pad_height))

            img = Image.new(img.mode, (width, height), (0,0,0)) # 검은색 캔버스 만들기
            img.paste(img_original, (pad_width_left, pad_height_top))

            anno_class_img = Image.new(anno_class_img.mode, (width, height), (0))
            anno_class_img.paste(anno_class_img_original, (pad_width_left, pad_height_top))
            anno_class_img.putpalette(p_palette)

        return img, anno_class_img

class RandomRotation(object):
    ''' 랜덤으로 이미지 회전 '''
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img, anno_class_img):
        rotate_angle = (np.random.uniform(self.angle[0], self.angle[1])) 

        # 회전
        img = img.rotate(rotate_angle, Image.BILINEAR)
        anno_class_img = anno_class_img.rotate(rotate_angle, Image.NEAREST)

        return img, anno_class_img

class RandomMirror(object):
    ''' 50%의 확률로 이미지를 좌우반전 시킨다.'''

    def __call__(self, img, anno_class_img):
        if np.random.randint(2): # 1또는 0만 나온다. -> 50% 확률
            img = ImageOps.mirror(img)
            anno_class_img = ImageOps.mirror(anno_class_img)
        
        return img, anno_class_img

class Resize(object):
    ''' 이미지 input_size 크기 조절 '''
    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, img, anno_class_img):
        img = img.resize((self.input_size, self.input_size),
                         Image.BICUBIC)
        anno_class_img = anno_class_img.resize((self.input_size, self.input_size),
                                                Image.NEAREST)
        return img, anno_class_img

class Normalize_Tensor(object):
    ''' 이미지 데이터를 파이토치 텐서로 변환 후 색상 정보 표준화'''
    def __init__(self, color_mean, color_std):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, img, anno_class_img):
        img = transforms.functional.to_tensor(img) # pytorch tensor로 변환
        
        # 색상 정보 표준화
        img = transforms.functional.normalize(img, self.color_mean, self.color_std)

        # annotation에서 라벨 255(배경)을 라벨 0(배경)으로 변경
        anno_class_img = np.array(anno_class_img)
        index = np.where(anno_class_img == 255)
        anno_class_img[index] = 0
        anno_class_img = torch.from_numpy(anno_class_img) # pytorch tensor로 변환

        return img, anno_class_img


    


