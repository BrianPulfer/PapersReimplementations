from argparse import ArgumentParser

import gymnasium as gym
import torch
import torch.nn as nn
from pytorch_lightning import seed_everything

from src.rl.dqn.dqn import dqn_training

DISCRETE_ACTION_ENVIRONMENTS = [
    "Acrobot-v1",
    "CartPole-v1",
    "LunarLander-v2",
    "MountainCar-v0",
]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=(64, 64)):
        super().__init__()
        self.relu = nn.ReLU()
        self.in_layer = nn.Linear(input_dim, hidden_dims[0])
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dims[i]),
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.ReLU(),
                )
                for i in range(len(hidden_dims) - 1)
            ]
        )
        self.out = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        x = self.in_layer(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


def main(args):
    # Setting seed
    seed_everything(args["seed"])

    # Unpacking args
    gamma = args["gamma"]
    epsilon = args["epsilon"]
    batch_size = args["batch_size"]
    lr = args["lr"]
    episodes = args["train_episodes"]
    checkpoint_path = args["checkpoint_path"]
    buffer_capacity = args["buffer_capacity"]
    optimizer_fn = getattr(torch.optim, args["optimizer"])

    # Environment
    env = gym.make(args["env"])

    # Training
    n_inputs, n_outputs = env.observation_space.shape[0], env.action_space.n
    model = MLP(n_inputs, n_outputs)
    optimizer = optimizer_fn(model.parameters(), lr=lr)
    dqn_training(
        model,
        env,
        optimizer,
        gamma,
        epsilon,
        batch_size,
        episodes,
        checkpoint_path,
        buffer_capacity,
    )

    # Showing episodes
    env = gym.make(args["env"], render_mode="human")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(n_inputs, n_outputs)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model = model.eval().to(device)
    for episode in range(args["test_episodes"]):
        state = env.reset()[0]
        done = False
        while not done:
            action = model(torch.tensor(state).float().to(device)).argmax().item()
            state, _, done, _, _ = env.step(action)
            env.render()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="CartPole-v1",
        choices=DISCRETE_ACTION_ENVIRONMENTS,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train_episodes", type=int, default=100)
    parser.add_argument("--test_episodes", type=int, default=10)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/dqn/dqn.pt")
    parser.add_argument("--buffer_capacity", type=int, default=1024)
    parser.add_argument(
        "--optimizer", type=str, default="Adam", choices=["Adam", "SGD"]
    )
    args = vars(parser.parse_args())
    print(args)
    main(args)
