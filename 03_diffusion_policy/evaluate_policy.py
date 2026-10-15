"""Evaluate the learned policy; run this script explicitly to start evaluation."""
from collections import deque
import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from uuid import uuid4

import torch
import wandb

from closed_loop_policy import load_policy
from sample_conditional_policy import sample_actions


SEED = 2026
NUM_TASKS = 100
ACTION_HORIZON = 4
MAX_ACTIONS = 20
SUCCESS_THRESHOLD = 0.1


def load_or_create_tasks(path):
    # Match training: independent uniform Start and Target in [-1, 1]^2.
    # This local Generator never consumes the diffusion sampling RNG stream.
    generator = torch.Generator().manual_seed(SEED)
    tasks = []
    for episode_id in range(NUM_TASKS):
        tasks.append({
            "episode_id": episode_id,
            "start_position": (torch.rand(2, generator=generator) * 2 - 1).tolist(),
            "target_position": (torch.rand(2, generator=generator) * 2 - 1).tolist(),
        })
    expected = {
        "seed": SEED,
        "distribution": "independent Uniform([-1, 1]^2) start and target",
        "tasks": tasks,
    }
    if path.exists():
        saved = json.loads(path.read_text())
        if saved != expected:
            raise ValueError(f"Existing task set differs from the fixed protocol: {path}")
        return saved["tasks"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(expected, handle, indent=2)
        handle.write("\n")
    return tasks


@torch.no_grad()
def evaluate_task(model, beta, task):
    # Adapt simulate() for arbitrary starts and quiet evaluation. Keep the same
    # history updates, first-four-action execution, and per-action stopping rule.
    position = torch.tensor(task["start_position"], dtype=torch.float32)
    target = torch.tensor(task["target_position"], dtype=torch.float32)
    initial_distance = torch.linalg.vector_norm(position - target).item()
    observation = torch.cat([position, target])
    history = deque([observation.clone(), observation.clone()], maxlen=2)
    distance = initial_distance
    executed_steps = 0
    planning_rounds = 0

    # Isolate each Episode's sampling stream, independent of early stopping.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED + task["episode_id"])
        while executed_steps < MAX_ACTIONS and distance >= SUCCESS_THRESHOLD:
            observations = torch.stack(list(history)).unsqueeze(0)
            actions = sample_actions(model, observations, beta)
            planning_rounds += 1
            remaining = min(ACTION_HORIZON, MAX_ACTIONS - executed_steps)
            for action in actions[:remaining]:
                position = position + action  # No clipping or Expert correction.
                executed_steps += 1
                history.append(torch.cat([position, target]))
                distance = torch.linalg.vector_norm(position - target).item()
                if distance < SUCCESS_THRESHOLD:
                    break

    return {
        "episode_id": task["episode_id"],
        "start_position": task["start_position"],
        "target_position": task["target_position"],
        "initial_distance": initial_distance,
        "final_position": position.tolist(),
        "final_distance": distance,
        "success": distance < SUCCESS_THRESHOLD,
        "executed_action_count": executed_steps,
        "planning_rounds": planning_rounds,
    }


def main():
    torch.manual_seed(SEED)
    root = Path(__file__).resolve().parent.parent
    checkpoint_path = root / "checkpoints" / "conditional_denoiser.pt"
    tasks_path = root / "outputs" / "evaluation_tasks_seed_2026.json"
    csv_path = root / "outputs" / "policy_evaluation_baseline_t100.csv"
    # Preserve previous local evaluation results instead of silently replacing them.
    if csv_path.exists():
        raise FileExistsError(f"Move or rename the previous CSV before rerunning: {csv_path}")
    model, beta = load_policy(root)
    model.eval()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    training_config = checkpoint["config"]
    if len(beta) != 100:
        raise ValueError("This baseline protocol requires the T=100 Checkpoint.")
    tasks = load_or_create_tasks(tasks_path)
    config = {
        "evaluation_seed": SEED,
        "task_seed": SEED,
        "sampling_seed_base": SEED,
        "sampling_seed_rule": "2026 + episode_id, isolated CPU RNG per Episode",
        "num_tasks": NUM_TASKS,
        "task_distribution": "independent Uniform([-1, 1]^2) start and target",
        "task_set_path": str(tasks_path),
        "task_set_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "diffusion_timesteps": len(beta),
        "beta_schedule": training_config["beta_schedule"],
        "beta_start": beta[0].item(),
        "beta_end": beta[-1].item(),
        "noise_schedule_source": "checkpoint beta Tensor",
        "final_alpha_bar": torch.cumprod(1 - beta, dim=0)[-1].item(),
        "sampling_variance": "posterior",
        "timestep_normalization": "t / (T - 1)",
        "observation_horizon": training_config["observation_horizon"],
        "prediction_horizon": training_config["prediction_horizon"],
        "action_horizon": ACTION_HORIZON,
        "action_dimension": 2,
        "max_actions": MAX_ACTIONS,
        "max_planning_rounds": MAX_ACTIONS // ACTION_HORIZON,
        "success_threshold": SUCCESS_THRESHOLD,
        "initial_history": "repeat initial observation twice",
        "stop_check": "before planning and after each action; distance < threshold",
        "device": "cpu",
        "torch_version": str(torch.__version__),
        "training_config": training_config,
    }
    run = wandb.init(
        project="nn-training",
        entity="josephzhangjie-university-of-california-berkeley",
        name="diffusion-policy-eval-baseline-t100",
        job_type="evaluation",
        tags=["diffusion-policy", "evaluation", "baseline", "t100"],
        id=uuid4().hex,
        resume="never",
        config=config,
    )
    exit_code = 1
    try:
        wandb.define_metric("episode_id")
        wandb.define_metric("episode/*", step_metric="episode_id")
        rows = []
        for task in tasks:
            row = evaluate_task(model, beta, task)
            rows.append(row)
            wandb.log({
                "episode_id": row["episode_id"],
                "episode/final_distance": row["final_distance"],
                "episode/success": int(row["success"]),
                "episode/executed_steps": row["executed_action_count"],
                "episode/planning_rounds": row["planning_rounds"],
            })
            if len(rows) % 10 == 0:
                print(f"Evaluation progress: {len(rows)}/{NUM_TASKS}")

        metrics = {
            "success_rate": mean(int(r["success"]) for r in rows),
            "mean_final_distance": mean(r["final_distance"] for r in rows),
            "median_final_distance": median(r["final_distance"] for r in rows),
            "mean_executed_steps": mean(r["executed_action_count"] for r in rows),
            "mean_planning_rounds": mean(r["planning_rounds"] for r in rows),
            "mean_initial_distance": mean(r["initial_distance"] for r in rows),
        }
        columns = list(rows[0])
        with csv_path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                # Store vector fields as JSON arrays for easy round-trip loading.
                writer.writerow({k: json.dumps(v) if isinstance(v, list) else v
                                 for k, v in row.items()})
        table = wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in rows])
        wandb.log({"episode_table": table, **metrics})
        wandb.run.summary.update(metrics)
        task_artifact = wandb.Artifact("reaching-evaluation-tasks-seed-2026", type="dataset")
        task_artifact.add_file(str(tasks_path))
        run.log_artifact(task_artifact)
        results_artifact = wandb.Artifact("policy-evaluation-baseline-t100", type="evaluation")
        results_artifact.add_file(str(csv_path))
        run.log_artifact(results_artifact)
        print("\nAggregate Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
        print(f"CSV: {csv_path}")
        print(f"W&B Run: {run.url}")
        exit_code = 0
    finally:
        wandb.finish(exit_code=exit_code)


if __name__ == "__main__":
    main()
