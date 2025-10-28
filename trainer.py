import torch
import torch.nn as nn
import torch.nn.functional as F


class SADLoss(nn.Module):
    """
    Spectral Angle Divergence (SAD) Loss Module.

    Calculates the angle between the predicted and true spectra, providing a
    measure of spectral shape similarity. A smaller angle means higher similarity.
    """

    def __init__(self):
        super(SADLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        Calculates the SAD loss.

        Args:
            y_pred (torch.Tensor): The predicted spectra from the model.
                                   Shape: (batch_size, num_bands)
            y_true (torch.Tensor): The ground truth spectra.
                                   Shape: (batch_size, num_bands)

        Returns:
            torch.Tensor: A scalar tensor representing the mean SAD loss for the batch.
        """
        # Normalize the vectors to unit length along the last dimension (num_bands)
        y_true_normalized = F.normalize(y_true, p=2, dim=-1)
        y_pred_normalized = F.normalize(y_pred, p=2, dim=-1)

        # Calculate the dot product
        # The result is the cosine of the angle between the vectors.
        cosine_similarity = torch.sum(y_true_normalized * y_pred_normalized, dim=-1)

        # Clamp for numerical stability (to handle values slightly out of [-1, 1] range)
        # This is crucial before applying acos.
        cosine_similarity_clamped = torch.clamp(cosine_similarity, -1.0, 1.0)

        # Calculate the angle in radians (the SAD)
        sad = torch.acos(cosine_similarity_clamped)

        # Return the mean SAD over the batch
        return torch.mean(sad)
