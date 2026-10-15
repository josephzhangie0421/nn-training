from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from train_conditional_denoiser import ConditionalDenoiser


@torch.no_grad()
def sample_actions(model, observations, beta):
    """Generate an open-loop Action Sequence with a fixed observation Condition."""
    T = len(beta)
    alpha = 1 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    alpha_bar_prev = torch.cat([torch.ones_like(alpha_bar[:1]), alpha_bar[:-1]])
    posterior_variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar)
    actions = torch.randn(1, 8, 2, device=observations.device)
    for step in reversed(range(T)):
        t = torch.full((1,), step, dtype=torch.long, device=observations.device)
        predicted_noise = model(actions, t, observations)
        mean = (
            actions - beta[step] / (1 - alpha_bar[step]).sqrt() * predicted_noise
        ) / alpha[step].sqrt()
        if step > 0:
            actions = mean + posterior_variance[step].sqrt() * torch.randn_like(actions)
        else:
            actions = mean  # Index 0 returns clean Actions without extra noise.
    return actions.squeeze(0).cpu()


def expert_baseline(start, target, steps=8):
    # Match the proportional Expert Controller in demo_dataset.py exactly.
    position = start.clone()
    actions = []
    for _ in range(steps):
        action = 0.25 * (target - position)
        actions.append(action)
        position = position + action
    return torch.stack(actions)


def trajectory(start, actions):
    return torch.cat([start.unsqueeze(0), start + actions.cumsum(dim=0)])


def main():
    torch.manual_seed(42)
    # CPU makes this small demonstration easy to repeat on the same environment.
    device = torch.device("cpu")
    root = Path(__file__).resolve().parent.parent
    checkpoint = torch.load(
        root / "checkpoints" / "conditional_denoiser.pt",
        map_location="cpu", weights_only=True,
    )
    config = checkpoint["config"]
    if (config["observation_horizon"] != 2
            or config["prediction_horizon"] != 8
            or config["model_dims"] != [25, 128, 128, 16]
            or config["activation"] != "SiLU"):
        raise ValueError("Checkpoint architecture does not match this demonstration.")
    T = config["diffusion_timesteps"]
    # Use saved beta directly, rather than constructing a different Schedule.
    beta = checkpoint["beta"].to(device)
    if beta.shape != (T,):
        raise ValueError("Saved beta does not match diffusion_timesteps.")
    model = ConditionalDenoiser(diffusion_timesteps=T).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print(f"Final alpha_bar: {torch.cumprod(1 - beta, dim=0)[-1].item():.6f}")
    print("Note: T=100 does not fully remove the training signal at the final step.")
    print("Repeated static observations also differ from moving Expert histories.")

    start = torch.tensor([0.0, 0.0])
    targets = [torch.tensor(p) for p in [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, target in zip(axes, targets):
        # [1, 2, 4]: two identical observations represent a stationary robot.
        observations = torch.cat([start, target]).repeat(2, 1).unsqueeze(0).to(device)
        actions = sample_actions(model, observations, beta)
        baseline_actions = expert_baseline(start, target)
        predicted_path = trajectory(start, actions)
        expert_path = trajectory(start, baseline_actions)
        distance = torch.linalg.vector_norm(predicted_path[-1] - target).item()
        expert_distance = torch.linalg.vector_norm(expert_path[-1] - target).item()
        print(f"\nTarget: {target.tolist()}")
        print("Generated Action Sequence:\n", actions)
        print("Final Position:", predicted_path[-1].tolist())
        print(f"Distance to Target: {distance:.6f}")
        print("Expert Action Sequence:\n", baseline_actions)
        print("Expert Final Position:", expert_path[-1].tolist())
        print(f"Expert Distance to Target: {expert_distance:.6f}")
        ax.plot(predicted_path[:, 0], predicted_path[:, 1], "o-", label="Predicted")
        ax.plot(expert_path[:, 0], expert_path[:, 1], "x--", label="Expert Baseline")
        ax.scatter(*start.tolist(), c="black", marker="s", s=70, label="Start", zorder=4)
        ax.scatter(*target.tolist(), c="red", marker="*", s=150, label="Target", zorder=4)
        ax.set_title(f"Target {tuple(target.tolist())}\nFinal distance: {distance:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        points = torch.cat([predicted_path, expert_path, target.unsqueeze(0)])
        lower = min(-1.2, points.min().item() - 0.15)
        upper = max(1.2, points.max().item() + 0.15)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output = root / "outputs" / "conditional_policy_trajectories.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves all existing images, including numbered ones.
    suffix = 0
    while True:
        candidate = output if suffix == 0 else output.with_stem(f"{output.stem}_{suffix}")
        try:
            with candidate.open("xb") as image_file:
                fig.savefig(image_file, format="png", dpi=150)
            break
        except FileExistsError:
            suffix += 1
    plt.close(fig)
    print(f"\nSaved plot: {candidate}")


if __name__ == "__main__":
    main()
