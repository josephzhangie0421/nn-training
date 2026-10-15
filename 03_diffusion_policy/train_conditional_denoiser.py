from pathlib import Path
from uuid import uuid4

import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from demo_dataset import ExpertDemoDataset


class ConditionalDenoiser(nn.Module):
    """Predict action noise conditioned on observation history and diffusion time."""

    def __init__(self, diffusion_timesteps=100):
        super().__init__()
        self.diffusion_timesteps = diffusion_timesteps
        self.mlp = nn.Sequential(
            nn.Linear(25, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 16),
        )

    def forward(self, noisy_actions, t, observations):
        # [B, 16] + [B, 8] + [B, 1] -> [B, 25].
        t_normalized = t.float().unsqueeze(1) / (self.diffusion_timesteps - 1)
        inputs = torch.cat([
            noisy_actions.flatten(start_dim=1),
            observations.flatten(start_dim=1),
            t_normalized,
        ], dim=1)
        return self.mlp(inputs).reshape(-1, 8, 2)


def main():
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    config = {
        "seed": 42,
        "num_episodes": 100,
        "episode_length": 20,
        "observation_horizon": 2,
        "prediction_horizon": 8,
        "batch_size": 32,
        "diffusion_timesteps": 100,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "beta_schedule": "linear",
        "epochs": 100,
        "learning_rate": 0.001,
        "model_dims": [25, 128, 128, 16],
        "activation": "SiLU",
        "optimizer": "Adam",
        "objective": "noise_prediction_mse",
        "device": str(device),
    }

    # Reuse the existing Episode generation and Sliding Window implementation.
    dataset = ExpertDemoDataset(
        num_episodes=config["num_episodes"],
        episode_length=config["episode_length"],
        observation_horizon=config["observation_horizon"],
        prediction_horizon=config["prediction_horizon"],
        seed=config["seed"],
    )
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    T = config["diffusion_timesteps"]
    beta = torch.linspace(config["beta_start"], config["beta_end"], T)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0).to(device)
    model = ConditionalDenoiser(diffusion_timesteps=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.MSELoss()

    # A fresh Run ID prevents resuming or overwriting an existing experiment.
    with wandb.init(
        project="nn-training",
        name="diffusion-policy-conditional-mlp",
        id=uuid4().hex,
        resume="never",
        config=config,
    ) as run:
        model.train()
        for epoch in range(1, config["epochs"] + 1):
            total_loss = 0.0
            for batch in loader:
                observations = batch["observations"].to(device)  # [B, 2, 4]
                actions = batch["actions"].to(device)  # Clean A0: [B, 8, 2].
                batch_size = actions.shape[0]

                # Each sample gets its own diffusion timestep, not an Episode time.
                t = torch.randint(0, T, (batch_size,), device=device)
                noise = torch.randn_like(actions)
                a_bar = alpha_bar[t].reshape(batch_size, 1, 1)
                # Corrupt Actions only; Observations remain the clean Condition.
                noisy_actions = a_bar.sqrt() * actions + (1 - a_bar).sqrt() * noise
                predicted_noise = model(noisy_actions, t, observations)
                loss = loss_fn(predicted_noise, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_size

            # Weight by sample count to include the smaller final Batch correctly.
            train_loss = total_loss / len(dataset)
            run.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "diffusion_timesteps": T,
                "observation_horizon": config["observation_horizon"],
                "prediction_horizon": config["prediction_horizon"],
            })
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}: Average Training Loss = {train_loss:.6f}")

        # Save portable CPU weights and the configuration in the project root.
        checkpoint_path = (
            Path(__file__).resolve().parent.parent
            / "checkpoints" / "conditional_denoiser.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            },
            "config": config,
            "beta": beta,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
