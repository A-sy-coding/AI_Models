import torch
import torch.nn as nn
import torch.functional as F

class PSPNet(nn.Module):
    '''
    Feature Module은 5개의 서브 네트워크로 구성되어 있다.
    AuxLoss 모듈은 Feature 모듈의 네번째 서브 네트워크 뒤로 출력을 뺀다.
    이후, 피라미드 풀링계층, 디코드 계층을 통과하여 최종 output을 가져오게 된다.
    '''
    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        # 파라미터 설정
        block_config = [3, 4, 6, 3] # ResNet50 모델
        img_size = 475
        img_size_8 = 60  # 원본 이미지에서 Feature 모듈을 통과하면 channel이 60이 되게 한다.
        
        # 전체 모듈 구조 설정
        self.feature_conv = FeatureMap.convolution()
        self.feature_res_1 = ResidualBlockPSP( n_blocks=block_config[0], in_channels=128, mid_channels=64,
                                                out_channels=256, stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP( n_blocks=block_config[1], in_channels=256, mid_channels=128, 
                                                out_channels=512, stride=2, dilation=1 )
        self.feature_dilated_res_1 = ResidualBlockPSP( n_blocks=block_config[2], in_channels=512, mid_channels=256,
                                                        out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP( n_blocks=block_config[3], in_channels=1024, mid_channels=512,
                                                        out_channels=2048, stride=1, dilation=4)

        self.pyramid_poolig = PyramidPooling(in_channels=2048, pool_size=[6, 3, 2, 1],
                                                height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature( height=img_size, width=img_size, n_classes=n_classes)

        self.aux = AuxiliaryPSPlayers(in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    # 순전파
    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_@(x)
        x = self.feature_dilated_res_1(x)

        output_aux = self.aux(x)

        x = self.feature_dilated_res_2(x)
        x = self.pyramid_poolig(x)
        output = self.decode_feature(x)

        return (output, output_aux)