import torch
import torch.nn as nn
import torch.nn.functional as F

class AIMloss(nn.Module):
    def __init__(self, lambda_aim=0.1):
        super().__init__()
        self.lambda_aim = lambda_aim

    def forward(self, features_A, features_B, features_C):
        
        assert features_A.shape == features_B.shape == features_C.shape, "All feature tensors should have the same shape"
         
        # A와 B 도메인 특징의 중간점 계산
        mid_AB = (features_A + features_B) / 2

        # C 도메인 특징과 중간점 사이의 거리 계산
        distance = F.mse_loss(features_C, mid_AB)

        # 거리를 최소화하는 손실 반환
        return self.lambda_aim * distance
    
    
    
class PAMloss(nn.Module):
    def __init__(self, num_stages=4, lambda_pam=0.1):
        super().__init__()
        self.num_stages = num_stages
        self.lambda_pam = lambda_pam

    def forward(self, features_list, domain_labels):
        total_loss = 0
        for stage, features in enumerate(features_list):
            # 도메인 별로 특징 분리
            domains = [features[domain_labels == i] for i in range(3)]
            
            # 모든 도메인 쌍에 대해 거리 계산
            for i in range(3):
                for j in range(i+1, 3):
                    if len(domains[i]) > 0 and len(domains[j]) > 0:
                        total_loss += self.compute_distance(domains[i], domains[j])

        return self.lambda_pam * (total_loss / self.num_stages)


    def compute_distance(self, features1, features2):
        # MMD (Maximum Mean Discrepancy) 거리 계산
        mean1 = torch.mean(features1, dim=0)
        mean2 = torch.mean(features2, dim=0)
        return torch.sum((mean1 - mean2) ** 2)