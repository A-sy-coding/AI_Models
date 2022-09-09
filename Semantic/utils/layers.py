import torch
import torch.nn as nn
import torch.functional as F

#-- Feuature Module 서브 네트워크 구성 Layers

#-- FeatureMap_convolution 서브 네트워크
class conv2DBatchNormRelu(nn.Module):
    '''
    3*3 conv, 배치 정규화, relu로 구성되어 있다.
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                                kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs

class FeatureMap_convolution(nn.Module):
    '''
    위의 conv층이 3개가 존재하고 마지막에서 max pooling 층이 존재하게 된다.
    input size : (3, 475, 475)
    middle : (64, 238, 238) -> (64, 238, 238) -> (128, 238, 238)
    output size : (128, 119, 119)
    '''
    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        # conv1 : (3, 64, 3, 2, 1, 1, False) = (in_channels, out_channels, kernel, stride, padding, dilation, bias )
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 3, 64, 3, 2, 1, 1, False
        self.cbnr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # conv2
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 64, 3, 1, 1, 1, False
        self.cbnr_2 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # conv3
        in_channels, out_channels, kernel_size, stride, padding, dilation, bias = 64, 128, 3, 1, 1, 1, False
        self.cbnr_3 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size, stride, padding, dilation, bias)

        # max pooling
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)

        return outputs    

#-- residualBlockPSP 서브 네트워크
class conv2DBatchNorm(nn.Module):
    ''' 합성곱 층과 배치 정규화만 있는 층 구현'''
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)

        return outputs

class bottleNeckPSP(nn.Module):
    ''' conv,batchnorm, relu가 한 계층인 모듈이 2개 존재하고, conv,batchnorm이 한 계층인 모듈이 2개 존재한다. '''
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(bottleNeckPSP, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # skip connection
        self.cb_residual = conv2DBatchNorm(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        
        return self.relu(conv + residual)

class bottleNeckIdentifyPSP(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(bottleNeckIdentifyPSP, self).__init__()

        self.cbr_1 = conv2DBatchNormRelu(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = conv2DBatchNormRelu(mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = conv2DBatchNorm(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # bottleNeckPSP 준비 (한번만 실행)
        self.add_module('block1', bottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))

        # bottleNeckIdentifyPSP 반복 준비
        for i in range(n_blocks-1):
            self.add_module('block'+ str(i+2), bottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))

#-- Pyramid Pooling 모듈
class PyramidPooling(nn.Module):
    '''
    PSPNet의 가장 핵심적인 module
    Feature Module에서 나온 output 값이 해당 모듈의 input값으로 들어가게 된다. (2048 * 60 * 60)
    60*60을 6*6, 3*3, 2*2, 1*1로 축소시키고 conv층을 지나간 뒤 다시 upsample을 통해 다시 60*60으로 복원시킨다.
    '''
    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()

        self.height = height
        self.width = width

        # 합성곱 층의 출력 채널 수 -> 2048 / 4(4개로 분할) = 512
        out_channels = int(in_channels / len(pool_sizes))

        # 합성곱 층 구현 --> pool_sizes : [6, 3, 2, 1]
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = conv2DBatchNormRelu(in_channels, out_channels, kernel_size=1, stride=1, padding=0,
                                        dilation=1, bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])                                        
        self.cbr_2 = conv2DBatchNormRelu(in_channels, out_channels,kernel_size=1, stride=1, padding=0,
                                        dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = conv2DBatchNormRelu(in_channels, out_channels,kernel_size=1, stride=1, padding=0,
                                        dilation=1, bias=False)
        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = conv2DBatchNormRelu(in_channels, out_channels,kernel_size=1, stride=1, padding=0,
                                        dilation=1, bias=False)

    # 순전파
    def forward(self, x):
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = torch.nn.functional.interpolate(out1, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = torch.nn.functional.interpolate(out2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = torch.nn.functional.interpolate(out3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = torch.nn.functional.interpolate(out4, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output = torch.cat([x, out1, out2, out3, out4], dim=1)

        return output

#-- Decoder

class DecodePSPFeature(nn.Module):
    '''
    pyramid pooling을 통과해서 나온 output (4096*60*60)
    최종 output이 21 * 475 * 475 
    Args:
        height -> output height
        width -> output width
        n_classes -> 클래스 분류 개수
    '''
    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(in_channels=4096, out_channels=512, kernel_size=3, stride=1,
                                        padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = torch.nn.functional.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output

#--- AuxLoss module
class AuxiliaryPSPlayers(nn.Module):
    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPlayers, self).__init__()

        self.height = height
        self.width = width

        self.cbr = conv2DBatchNormRelu(in_channels=in_channels, out_channels=256, kernel_size=3, stride=1,padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = torch.nn.functional.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)

        return output



