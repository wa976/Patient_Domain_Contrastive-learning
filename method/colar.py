import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCORALLoss(nn.Module):
    def __init__(self):
        super(DeepCORALLoss, self).__init__()

    def compute_covariance(self, features):
        """
        Compute the covariance matrix for a set of features.
        :param features: Tensor of features (batch_size x num_features)
        :return: Covariance matrix.
        """
        n = features.size(0)
        features_mean = torch.mean(features, dim=0, keepdim=True)
        features_centered = features - features_mean
        covariance = torch.mm(features_centered.t(), features_centered) / n
        return covariance

    def forward(self, source_features, target_features):
        """
        Compute the Deep CORAL loss between source and target features.
        :param source_features: Tensor of features from the source domain.
        :param target_features: Tensor of features from the target domain.
        :return: Deep CORAL loss.
        """
        # Check for empty tensors
        if source_features.nelement() == 0 or target_features.nelement() == 0:
            return torch.tensor(0.0).to(source_features.device)

        # Flatten the last two dimensions
        source_features = source_features.view(source_features.size(0), -1)
        target_features = target_features.view(target_features.size(0), -1)

        # Compute the covariance matrices
        source_cov = self.compute_covariance(source_features)
        target_cov = self.compute_covariance(target_features)

        # Compute the CORAL loss
        coral_loss = torch.mean(torch.pow(source_cov - target_cov, 2))
        return coral_loss / (4 * source_cov.size(0) * source_cov.size(0))