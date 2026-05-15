import torch

def weighted_pred_loss(predictions, targets, K=5, eta=0.7):
    """
    Weighted MSE loss for future K steps (Equation 12).
    :param predictions: Predicted CSI blocks (batch, K, 15)
    :param targets: Ground-truth CSI blocks (batch, K, 15)
    :param K: Prediction horizon (default 5)
    :param eta: Discount factor (default 0.7)
    """
    batch_size = predictions.size(0)
    total_loss = 0.0
    weight_sum = 0.0
    
    for i in range(K):
        # Weight \eta^i for the i-th future step
        weight = eta ** i
        
        # Calculate MSE for this specific step
        step_mse = torch.mean(torch.sum((predictions[:, i, :] - targets[:, i, :]) ** 2, dim=-1))
        
        total_loss += weight * step_mse
        weight_sum += weight
        
    # Return normalized weighted loss
    return total_loss / weight_sum
