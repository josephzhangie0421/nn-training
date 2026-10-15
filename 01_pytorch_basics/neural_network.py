import wandb
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

wandb.init(
    project="nn-training",
    config={
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 1001,
        "architecture": "MLP",
    },
)

# 1. Training Data
torch.manual_seed(42)

X = torch.linspace(-2, 2, 200).reshape(-1, 1)
y = X ** 2
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Neural Network
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

# 3. Loss Function
loss_fn = nn.MSELoss()

# 4. Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 5. Training Loop
for epoch in range(1001):

    total_loss = 0.0

    for batch_X, batch_y in dataloader:

        prediction = model(batch_X)
        loss = loss_fn(prediction, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_X)

    average_loss = total_loss / len(dataset)

    wandb.log({
    "epoch": epoch,
    "train_loss": average_loss,
    })

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {average_loss:.6f}")

# 6. Inference
with torch.no_grad():
    test_input = torch.tensor([[1.5]])
    test_output = model(test_input)

    print("\nTest Input:", test_input.item())
    print("Model Prediction:", test_output.item())
    print("True Value:", 1.5 ** 2)

    wandb.finish()