import torch
from models import CSIPredictor
from losses import weighted_pred_loss

def run_example():
    # 1. Setup environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSIPredictor().to(device)
    K_steps = 5  # Predict 125ms into the future (5 * 25ms)

    # 2. Dummy data for demonstration (Batch size=32)
    # 10 blocks of history (250ms)
    context_csi = torch.randn(32, 10, 15).to(device)
    context_alpha = torch.zeros(32, 10, 1).to(device) # Age is 0 for observations

    # 3. Warm-up hidden state using context
    model.eval()
    with torch.no_grad():
        _, hidden = model(context_csi, context_alpha)

    # 4. Autoregressive Rollout
    current_input_x = context_csi[:, -1:, :] # Start from the last observation
    predictions = []

    for i in range(1, K_steps + 1):
        # Increment age feature for each recursive step
        age_tensor = torch.full((32, 1, 1), float(i)).to(device)
        
        # Predict next step
        pred_x, hidden = model(current_input_x, age_tensor, hidden)
        predictions.append(pred_x)
        
        # Feedback prediction as next input
        current_input_x = pred_x

    # Combine results (batch, 5, 15)
    final_forecast = torch.cat(predictions, dim=1)
    print(f"Prediction result shape: {final_forecast.shape}")
    print(f"Successfully forecasted {K_steps * 25}ms into the future.")

if __name__ == "__main__":
    run_example()
