from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")  # 只保存图片，不打开 GUI。
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb


def main():
    torch.manual_seed(42)
    config = {
        "seed": 42,
        "cluster_centers": [[-2.0, 0.0], [2.0, 0.0]],
        "samples_per_cluster": 1000,
        "cluster_std": 0.3,
        "batch_size": 128,
        "T": 1000,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "schedule": "linear",
        "epochs": 200,
        "learning_rate": 0.001,
        "model_dims": [3, 64, 64, 2],
        "activation": "SiLU",
        "timestep_normalization": "t / (T - 1)",
        "objective": "epsilon_prediction",
        "loss": "MSE",
        "optimizer": "Adam",
        "sampling_variance": "posterior",
        "num_generated": 1000,
    }
    root = Path(__file__).resolve().parent

    # Toy Dataset：两个二维 Gaussian Clusters。
    data = torch.cat([
        torch.randn(config["samples_per_cluster"], 2) * config["cluster_std"]
        + torch.tensor(center)
        for center in config["cluster_centers"]
    ])
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Training 和 Sampling 直接复用同一组 Noise Schedule Tensor。
    T = config["T"]
    beta = torch.linspace(config["beta_start"], config["beta_end"], T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    # 索引 0 的前一个累计系数设为 1，表示完全无噪声的原始数据。
    alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]])
    posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    print(f"Final alpha_bar = {alpha_bar[-1].item():.8f}")

    model = nn.Sequential(
        nn.Linear(3, 64), nn.SiLU(),
        nn.Linear(64, 64), nn.SiLU(),
        nn.Linear(64, 2),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()

    # 新建 Run ID，禁止 Resume，避免覆盖已有 W&B Runs。
    with wandb.init(
        project="nn-training",
        name="ddpm-2d-mlp-t1000",
        id=uuid4().hex,
        resume="never",
        config=config,
    ) as run:
        model.train()
        for epoch in range(1, config["epochs"] + 1):
            total_loss = 0.0
            for (x0,) in loader:
                batch_size = x0.shape[0]
                # 每个样本独立采样 Timestep 和 Ground Truth Noise。
                t = torch.randint(0, T, (batch_size,))
                epsilon = torch.randn_like(x0)
                a_bar = alpha_bar[t].unsqueeze(1)  # [B, 1] 广播到 [B, 2]。
                x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * epsilon
                t_float = t.float().unsqueeze(1) / (T - 1)
                epsilon_hat = model(torch.cat([x_t, t_float], dim=1))
                loss = loss_fn(epsilon_hat, epsilon)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size

            train_loss = total_loss / len(dataset)
            run.log({"epoch": epoch, "train_loss": train_loss})
            if epoch % 20 == 0:
                print(f"Epoch {epoch:3d}: Average Training Loss = {train_loss:.6f}")

        # 保存 Weights、Config 和实际使用的 beta，方便恢复一致的 Schedule。
        checkpoint_path = root / "checkpoints" / "ddpm_2d_t1000.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": config,
            "beta": beta,
        }, checkpoint_path)

        # Reverse Sampling：从标准 Gaussian Noise 开始逐步去噪。
        model.eval()
        with torch.no_grad():
            x = torch.randn(config["num_generated"], 2)
            for t in reversed(range(T)):
                t_float = torch.full((x.shape[0], 1), t / (T - 1))
                epsilon_hat = model(torch.cat([x, t_float], dim=1))
                # DDPM Mean，由预测噪声和当前状态计算。
                mean = (
                    x - beta[t] / (1 - alpha_bar[t]).sqrt() * epsilon_hat
                ) / alpha[t].sqrt()
                if t > 0:
                    x = mean + posterior_variance[t].sqrt() * torch.randn_like(x)
                else:
                    x = mean  # 最后一步回到数据空间，不再加随机噪声。
            generated = x.numpy()

        # 两幅 Scatter Plot 使用相同坐标范围，便于比较分布。
        real = data.numpy()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
        axes[0].scatter(real[:, 0], real[:, 1], s=8, alpha=0.5)
        axes[0].set_title("Real Dataset (2000 points)")
        axes[1].scatter(generated[:, 0], generated[:, 1], s=8, alpha=0.5)
        axes[1].set_title("DDPM Generated Samples (1000 points)")
        for ax in axes:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.2)
        fig.tight_layout()
        output_path = root / "outputs" / "ddpm_2d_comparison.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved checkpoint: {checkpoint_path}")
        print(f"Saved comparison: {output_path}")


if __name__ == "__main__":
    main()
