import torch

# 파일 불러오기 (경로에 맞게 수정)
file_path = "datasets/csi_2_2GHz.pt"
data = torch.load(file_path)

# 데이터 구조 확인
print("=== 데이터셋 정보 ===")
print(f"저장된 키(Keys): {data.keys()}")

# X (입력 데이터) 확인
X_tensor = data['X']
print(f"\n입력(X) 차원: {X_tensor.shape}")
print(f"입력(X) 데이터 예시 (첫 번째 윈도우): \n{X_tensor[0]}")

# Y (정답 데이터) 확인
Y_tensor = data['Y']
print(f"\n정답(Y) 차원: {Y_tensor.shape}")
print(f"정답(Y) 데이터 예시 (첫 번째 타겟): \n{Y_tensor[0]}")