import torch.nn as nn
import torch.functional as F
import math

#-- 손실함수 정의
class PSPLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super(PSPLoss, self).__init__()
        self.aux_weight = aux_weight
    
    def forward(self, outputs, targets):
        '''
        outputs (tuple) : PSPNet 출력
        targets (list) : 정답 어노테이션 정보
        '''
        loss = F.cross_entropy(outputs[0], targets, reduction='mean')
        loss_aux = F.cross_entropy(outputs[1], targets, reduction='mean')

        return loss + self.aux_weight * loss_aux

#-- 스케쥴러 설정
def lambda_epoch(epoch):
    max_epoch = 30
    return math.pow((1-epoch/max_epoch), 0.9)
