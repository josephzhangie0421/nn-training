from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Reuse the exact model class and Reverse Sampling implementation.
from sample_conditional_policy import ConditionalDenoiser, sample_actions


def load_policy(root):
    # Match the CPU loading and saved Noise Schedule used by the sampling demo.
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
    beta = checkpoint["beta"].to("cpu")
    if beta.shape != (T,):
        raise ValueError("Saved beta does not match diffusion_timesteps.")
    model = ConditionalDenoiser(diffusion_timesteps=T).to("cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, beta


@torch.no_grad()
def simulate(model, beta, target, action_horizon=4, max_rounds=5):
    position = torch.zeros(2)
    initial_observation = torch.cat([position, target])
    history = deque([initial_observation.clone(), initial_observation.clone()], maxlen=2)
    positions = [position.clone()]
    distances = [torch.linalg.vector_norm(position - target).item()]
    plans = []

    for round_index in range(max_rounds):
        if distances[-1] < 0.1:
            break
        # After initialization, these are the last two actual consecutive states.
        observations = torch.stack(list(history)).unsqueeze(0)  # [1, 2, 4]
        actions = sample_actions(model, observations, beta)  # [8, 2]
        plan = {
            "observations": observations.clone(),
            "predicted_actions": actions.clone(),
            "distance_before": distances[-1],
            "executed_action_count": 0,
        }
        plans.append(plan)
        print(f"Round {round_index + 1}, predicted Actions:\n{actions}")

        # Execute only the first Action Horizon actions, without clipping or correction.
        for action in actions[:action_horizon]:
            position = position + action
            history.append(torch.cat([position, target]))
            positions.append(position.clone())
            distances.append(torch.linalg.vector_norm(position - target).item())
            plan["executed_action_count"] += 1
            print(f"  Step {len(positions) - 1:2d}: position={position.tolist()}, "
                  f"distance={distances[-1]:.6f}")
            if distances[-1] < 0.1:
                break
        plan["distance_after"] = distances[-1]

    # Keep all states, distances, and full predictions for inspection.
    return {
        "target": target.clone(),
        "positions": torch.stack(positions),
        "distances": distances,
        "plans": plans,
    }


def main():
    torch.manual_seed(42)
    root = Path(__file__).resolve().parent.parent
    model, beta = load_policy(root)
    targets = [torch.tensor(p) for p in [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]]
    results = []
    for target in targets:
        print(f"\nTarget: {target.tolist()}")
        result = simulate(model, beta, target)
        results.append(result)
        print("Final Position:", result["positions"][-1].tolist())
        print(f"Final Distance: {result['distances'][-1]:.6f}")
        print("Planning Rounds:", len(result["plans"]))
        print("Executed Action Count:", len(result["positions"]) - 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, result in zip(axes, results):
        positions = result["positions"]
        target = result["target"]
        ax.plot(positions[:, 0], positions[:, 1], "o-", label="Closed-Loop")
        ax.scatter(0, 0, c="black", marker="s", s=70, label="Start", zorder=4)
        ax.scatter(*target.tolist(), c="red", marker="*", s=150, label="Target", zorder=4)
        points = torch.cat([positions, target.unsqueeze(0)])
        lower = min(-1.2, points.min().item() - 0.15)
        upper = max(1.2, points.max().item() + 0.15)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Target {tuple(target.tolist())}\n"
                     f"Final distance: {result['distances'][-1]:.3f}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    output = root / "outputs" / "closed_loop_trajectories.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve existing figures if the simulation is run again.
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
    return results


if __name__ == "__main__":
    main()
