import torch
from torch.utils.data import DataLoader, Dataset


class ExpertDemoDataset(Dataset):
    """Synthetic reaching episodes exposed as observation/action windows."""

    def __init__(
        self,
        num_episodes=100,
        episode_length=20,
        observation_horizon=2,
        prediction_horizon=8,
        seed=42,
    ):
        if min(num_episodes, observation_horizon, prediction_horizon) < 1:
            raise ValueError("Episode count and horizons must be positive.")
        if episode_length < observation_horizon + prediction_horizon - 1:
            raise ValueError("Episode is too short for a complete window.")

        self.observation_horizon = observation_horizon
        self.prediction_horizon = prediction_horizon
        self.episodes = []
        self.indices = []
        generator = torch.Generator().manual_seed(seed)

        # An Episode contains observations before each action and its displacements.
        for episode_id in range(num_episodes):
            position = torch.rand(2, generator=generator) * 2 - 1
            target = torch.rand(2, generator=generator) * 2 - 1
            observations = []
            actions = []
            for _ in range(episode_length):
                observations.append(torch.cat([position, target]))
                # A proportional Expert Controller moves 25% of the remaining gap.
                action = 0.25 * (target - position)
                actions.append(action)
                position = position + action

            self.episodes.append({
                "observations": torch.stack(observations).float(),
                "actions": torch.stack(actions).float(),
            })

            # Sliding Windows stay inside one Episode; no Padding is needed.
            # k is the last observation index and the first predicted action index.
            for k in range(observation_horizon - 1,
                           episode_length - prediction_horizon + 1):
                self.indices.append((episode_id, k))

    def __len__(self):
        # Dataset length counts valid windows, rather than Episodes.
        return len(self.indices)

    def __getitem__(self, index):
        # Map a flat sample index to an Episode and window endpoint k.
        episode_id, k = self.indices[index]
        episode = self.episodes[episode_id]
        start = k - self.observation_horizon + 1
        return {
            "observations": episode["observations"][start:k + 1],
            "actions": episode["actions"][k:k + self.prediction_horizon],
        }


def main():
    torch.manual_seed(42)
    dataset = ExpertDemoDataset()
    # DataLoader shuffles windows and stacks each dictionary field into batches.
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    first_sample = dataset[0]

    print("Number of Episodes:", len(dataset.episodes))
    print("Number of Training Samples:", len(dataset))
    print("Observation Batch Shape:", batch["observations"].shape)
    print("Action Batch Shape:", batch["actions"].shape)
    print("First Sample Observations:\n", first_sample["observations"])
    print("First Sample Actions:\n", first_sample["actions"])


if __name__ == "__main__":
    main()
