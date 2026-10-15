import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb


def main():
    torch.manual_seed(42)

    # 集中记录 Hyperparameters，便于在 W&B 中查看。
    config = {
        "seed": 42,
        "samples_per_cluster": 1000,
        "cluster_centers": [[-2.0, 0.0], [2.0, 0.0]],
        "cluster_std": 0.3,
        "batch_size": 128,
        "T": 100,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "learning_rate": 0.001,
        "epochs": 200,
        "hidden_dims": [64, 64],
        "activation": "SiLU",
        "optimizer": "Adam",
        "loss": "MSE",
    }

    # 两个 Gaussian Clusters；不需要类别标签。
    clusters = [
        torch.randn(config["samples_per_cluster"], 2) * config["cluster_std"]
        + torch.tensor(center)
        for center in config["cluster_centers"]
    ]
    data = torch.cat(clusters, dim=0)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Linear Beta Schedule；t=0 表示第一步加噪后的状态。
    T = config["T"]
    beta = torch.linspace(config["beta_start"], config["beta_end"], T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)

    # 输入为二维 x_t 和一个归一化 Timestep，输出二维预测噪声。
    model = nn.Sequential(
        nn.Linear(3, 64),
        nn.SiLU(),
        nn.Linear(64, 64),
        nn.SiLU(),
        nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()

    # 退出 with 时自动结束 W&B Run。
    with wandb.init(
        project="nn-training", name="ddpm-2d-denoiser", config=config
    ) as run:
        model.train()
        for epoch in range(1, config["epochs"] + 1):
            total_loss = 0.0
            for (x0,) in loader:
                batch_size = x0.shape[0]

                # 每个样本独立采样整数 Timestep 和 Ground Truth Noise。
                t = torch.randint(0, T, (batch_size,))
                epsilon = torch.randn_like(x0)

                # [B, 1] 的系数会广播到 [B, 2]。
                alpha_bar_t = alpha_bar[t].unsqueeze(1)
                x_t = (
                    torch.sqrt(alpha_bar_t) * x0
                    + torch.sqrt(1 - alpha_bar_t) * epsilon
                )

                # Timestep 转成 Float Tensor，再拼接成 [B, 3]。
                t_normalized = t.float().unsqueeze(1) / (T - 1)
                model_input = torch.cat([x_t, t_normalized], dim=1)
                epsilon_hat = model(model_input)
                loss = loss_fn(epsilon_hat, epsilon)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 按样本数加权，正确处理最后一个较小的 Mini-batch。
                total_loss += loss.item() * batch_size

            train_loss = total_loss / len(dataset)
            run.log({"train_loss": train_loss, "epoch": epoch})
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Average Training Loss = {train_loss:.6f}")


if __name__ == "__main__":
    main()
