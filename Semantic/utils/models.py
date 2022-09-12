import torch
import torch.nn as nn
import torch.functional as F
from .layers import FeatureMap_convolution, ResidualBlockPSP, PyramidPooling, DecodePSPFeature, AuxiliaryPSPlayers
import time
import pandas as pd

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
        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP( n_blocks=block_config[0], in_channels=128, mid_channels=64,
                                                out_channels=256, stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP( n_blocks=block_config[1], in_channels=256, mid_channels=128, 
                                                out_channels=512, stride=2, dilation=1 )
        self.feature_dilated_res_1 = ResidualBlockPSP( n_blocks=block_config[2], in_channels=512, mid_channels=256,
                                                        out_channels=1024, stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP( n_blocks=block_config[3], in_channels=1024, mid_channels=512,
                                                        out_channels=2048, stride=1, dilation=4)

        self.pyramid_pooling = PyramidPooling(in_channels=2048, pool_sizes=[6, 3, 2, 1],
                                                height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature( height=img_size, width=img_size, n_classes=n_classes)

        self.aux = AuxiliaryPSPlayers(in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    # 순전파
    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        output_aux = self.aux(x)

        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return (output, output_aux)


#-- train model 함수 구현
def train_model(net, dataloaders_dict, criterion, scheduler, optimizer, num_epochs):
    '''
    net : PSPNet 신경망
    dataloaders_dict(dict) : train과 val이 key값으로 들어가고 해당 데이터셋들이 value값으로 들어간 딕셔너리 형태
    criterion(class) : 모델 loss 클래스
    scheduler(def) : 학습이 진행될수록 학습률이 낮아지도록 조절
    optimizer : 최적화 방식 -> SGD사용
    num_epochs(int) : 학습을 진행할 횟수
    '''
    
    # gpu 사용가능 확인
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('사용 장치 : ', device)

    net.to(device) # gpu or cpu로 설정
    
    # 네트워크가 고정되면 고속화
    torch.backends.cudnn.benchmark = True

    num_train_imgs = len(dataloaders_dict['train'].dataset)
    num_val_imgs = len(dataloaders_dict['val'].dataset)
    batch_size = dataloaders_dict['train'].batch_size

    iteration = 1
    logs = []

    # 멀티 gpu 설정
    batch_multiplier = 3

    # 학습 시작
    for epoch in range(num_epochs):
        t_epoch_start = time.time()
        t_iter_start = time.time()
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        print('-----------------')
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-----------------')

        for phase in ['train','val']:
            if phase == 'train':
                net.train() # 학습 시작
                scheduler.step()
                optimizer.zero_grad()
                print(' (train) ')
            else:
                if ((epoch+1)%5 == 0):
                    net.eval() # 평가모드
                    print('--------')
                    print(' (val) ')
                else:
                    continue

            # 미니배치 시작
            count = 0  # multiple 미니배치
            for imgs, anno_class_imgs in dataloaders_dict[phase]:
                if imgs.size()[0] == 1:
                    continue
                
                imgs = imgs.to(device)
                anno_class_imgs = anno_class_imgs.to(device)

                # 미니 배치로 파라미터 갱신
                if (phase == 'train') and (count == 0):
                    optimizer.step()
                    optimizer.zero_grad()
                    count = batch_multiplier
                
                # 순전파 계산
                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(imgs)
                    loss = criterion(outputs, anno_class_imgs.long()) / batch_multiplier 

                    if phase=='train':
                        loss.backward()
                        count -= 1

                        if (iteration % 10 == 0):
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print(f'반복 {iteration} || Loss : {loss.item()/batch_size*batch_multiplier:.4f} || \
                                    10iter : {duration:.4f} sec')
                            t_iter_start = time.time()
                        
                        epoch_train_loss += loss.item() * batch_multiplier
                        iteration += 1
                    else:
                        epoch_val_loss += loss.item() * batch_multiplier
        # epoch의 phase별 loss와 정답률
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss/num_train_imgs, epoch_val_loss/num_val_imgs))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # 로그 저장
        log_epoch = {'epoch': epoch+1, 'train_loss': epoch_train_loss /
                        num_train_imgs, 'val_loss': epoch_val_loss/num_val_imgs}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

    # 최후의 네트워크를 저장
    torch.save(net.state_dict(), 'weights/pspnet50_' + str(epoch+1) + '.pth')

