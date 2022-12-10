"""
Personal reimplementation of
    Proximal Policy Optimization Algorithms
(https://arxiv.org/abs/1707.06347)
"""

from argparse import ArgumentParser

import gym
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.distributions.categorical import Categorical

import pytorch_lightning as pl

# Definitions
MODEL_PATH = "model.pt"


def parse_args():
    """Pareser program arguments"""
    # Parser
    parser = ArgumentParser()

    # Program arguments (default for Atari games)
    parser.add_argument("--max_iterations", type=int, help="Number of iterations of training", default=10_000)
    parser.add_argument("--n_actors", type=int, help="Number of actors for each update", default=8)
    parser.add_argument("--horizon", type=int, help="Number of timestamps for each actor", default=128)
    parser.add_argument("--epsilon", type=float, help="Epsilon parameter", default=0.1)
    parser.add_argument("--n_epochs", type=int, help="Number of training epochs per iteration", default=3)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32*8)
    parser.add_argument("--lr", type=float, help="Learning rate", default=2.5 * 1e-4)
    parser.add_argument("--gamma", type=float, help="Discount factor gamma", default=0.99)
    parser.add_argument("--c1", type=float, help="Weight for the value function in the loss function", default=0.5)  # TODO: set to 1
    parser.add_argument("--c2", type=float, help="Weight for the entropy bonus in the loss function", default=0.01)
    parser.add_argument("--n_test_episodes", type=int, help="Number of episodes to render", default=5)
    parser.add_argument("--seed", type=int, help="Randomizing seed for the experiment", default=0)

    # Dictionary with program arguments
    return vars(parser.parse_args())


def get_device():
    """Gets the device (GPU if any) and logs the type"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Found GPU device: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cpu")
        print("No GPU found: Running on CPU")
    return device


class MyPPO(nn.Module):
    """Implementation of a PPO model. The same backbone is used to get actor and critic values."""
    def __init__(self, in_shape, n_actions):
        # Super constructor
        super(MyPPO, self).__init__()

        # Attributes
        self.in_shape = in_shape
        self.n_actions = n_actions

        # Shared backbone for policy and value functions
        in_dim = np.prod(in_shape)
        self.to_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, in_dim),
            nn.Tanh(),
            nn.Linear(in_dim, in_dim),
            nn.Tanh()
        )

        # State action function
        self.action_fn = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, n_actions),
            nn.Softmax(dim=-1)
        )

        # Value function
        self.value_fn = nn.Sequential(
            nn.Linear(in_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        features = self.to_features(x)
        action = self.action_fn(features)
        value = self.value_fn(features)
        return Categorical(action).sample(), action, value


@torch.no_grad()
def run_timestamps(env, model, timestamps=2048, render=False, device="cpu"):
    """Runs the given policy on the given environment for the given amount of timestamps.
     Returns a buffer with state action transitions and rewards."""
    buffer = []
    state = env.reset()[0]

    # Running timestamps and collecting state, actions, rewards and terminations
    for ts in range(timestamps):
        # Taking a step into the environment
        model_input = torch.from_numpy(state).unsqueeze(0).to(device).float()
        action, action_logits, value = model(model_input)
        new_state, reward, terminated, truncated, info = env.step(action.item())

        # Rendering / storing (s, a, r, t) in the buffer
        if render:
            env.render()
        else:
            buffer.append([model_input, action_logits, reward, terminated or truncated])

        # Updating current state
        state = new_state

        # Resetting environment if episode terminated or truncated
        if terminated or truncated:
            state = env.reset()[0]

    return buffer


def compute_cumulative_rewards(buffer, gamma):
    """Given a buffer with states, policy action logits, rewards and terminations,
    computes the cumulative rewards for each timestamp and substitutes them into the buffer."""
    curr_rew = 0.

    # Traversing the buffer on the reverse direction
    for i in range(len(buffer) - 1, -1, -1):
        r, t = buffer[i][-2], buffer[i][-1]

        if t:
            curr_rew = 0
        else:
            curr_rew = r + gamma * curr_rew

        buffer[i][-2] = curr_rew


def get_losses(model, batch, epsilon, annealing, device="cpu"):
    """Returns the three loss terms for a given model and a given batch and additional parameters"""
    # Getting old data
    n = len(batch)
    states = torch.cat([batch[i][0] for i in range(n)])
    logits = torch.cat([batch[i][1] for i in range(n)])
    cumulative_rewards = torch.tensor([batch[i][-2] for i in range(n)]).view(-1, 1).float().to(device)
    # cumulative_rewards = (cumulative_rewards - torch.mean(cumulative_rewards)) / (torch.std(cumulative_rewards) + 1e-7)

    # Computing predictions with the new model
    new_actions, new_logits, new_values = model(states)
    new_actions = new_actions.view(n, -1)

    # Loss on the state-action-function / actor (L_CLIP)
    advantages = cumulative_rewards - new_values.detach()  # Stopping gradient on the critic function
    margin = epsilon * annealing
    ratios = new_logits.gather(1, new_actions) / logits.gather(1, new_actions)

    l_clip = torch.mean(
        torch.min(
            torch.cat(
                (ratios * advantages,
                 torch.clip(ratios, 1 - margin, 1 + margin) * advantages),
                dim=1),
            dim=1
        ).values
    )

    # Loss on the value-function / critic (L_VF)
    l_vf = torch.mean((cumulative_rewards - new_values) ** 2)

    # Bonus for entropy of the actor
    entropy_bonus = torch.mean(torch.sum(-new_logits * (torch.log(new_logits + 1e-5)), dim=1))

    return l_clip, l_vf, entropy_bonus


def training_loop(env, model, max_iterations, n_actors, horizon, gamma, epsilon, n_epochs, batch_size, lr,
                  c1, c2, device, env_name=""):
    """Train the model on the given environment using multiple actors acting up to n timestamps."""

    """
    # Starting a new Weights & Biases run
    wandb.init(project="Papers Re-implementations",
               entity="peutlefaire",
               name=f"PPO - {env_name}",
               config={
                   "env": str(env),
                   "number of actors": n_actors,
                   "horizon": horizon,
                   "gamma": gamma,
                   "epsilon": epsilon,
                   "epochs": n_epochs,
                   "batch size": batch_size,
                   "learning rate": lr,
                   "c1": c1,
                   "c2": c2
               })
    """

    # Training variables
    max_reward = float("-inf")
    optimizer = Adam(model.parameters(), lr=lr, maximize=True)
    scheduler = LinearLR(optimizer, 1, 0, max_iterations * n_epochs)
    anneals = np.linspace(1, 0, max_iterations)

    # Training loop
    for iteration in range(max_iterations):
        buffer = []
        annealing = anneals[iteration]

        # Collecting timestamps for all actors with the current policy
        for actor in range(1, n_actors + 1):
            buffer.extend(run_timestamps(env, model, horizon, False, device))

        # Computing cumulative rewards and shuffling the buffer
        compute_cumulative_rewards(buffer, gamma)
        np.random.shuffle(buffer)

        # Getting the average reward (for logging and checkpointing)
        avg_rew = np.mean([buffer[i][-2] for i in range(len(buffer))])

        # Running optimization for a few epochs
        for epoch in range(n_epochs):
            for batch_idx in range(len(buffer) // batch_size):
                # Getting batch for this buffer
                start = batch_size * batch_idx
                end = start + batch_size if start + batch_size < len(buffer) else -1
                batch = buffer[start:end]

                # Zero-ing optimizers gradients
                optimizer.zero_grad()

                # Getting the losses
                l_clip, l_vf, entropy_bonus = get_losses(model, batch, epsilon, annealing, device)

                # Computing total loss and back-propagating it
                loss = l_clip - c1 * l_vf + c2 * entropy_bonus
                loss.backward()

                # Optimizing
                optimizer.step()
            scheduler.step()

        # Logging information to stdout
        curr_loss = loss.item()
        log = f"Iteration {iteration + 1} / {max_iterations}: " \
              f"Average Reward: {avg_rew:.2f}\t" \
              f"Loss: {curr_loss:.3f} " \
              f"(L_CLIP: {l_clip.item():.1f} | L_VF: {l_vf.item():.1f} | L_bonus: {entropy_bonus.item():.1f})"
        if avg_rew > max_reward:
            torch.save(model.state_dict(), MODEL_PATH)
            max_reward = avg_rew
            log += " --> Stored model with highest average reward"
        print(log)

        """
        # Logging information to W&B
        wandb.log({
            "loss (total)": curr_loss,
            "loss (clip)": l_clip.item(),
            "loss (vf)": l_vf.item(),
            "loss (entropy bonus)": entropy_bonus.item(),
            "average reward": avg_rew
        })

    # Finishing W&B session
    wandb.finish()
    """

def testing_loop(env, model, n_episodes, device):
    """Runs the learned policy on the environment for n episodes"""
    for _ in range(n_episodes):
        run_timestamps(env, model, timestamps=1_000, render=True, device=device)


def main():
    # Parsing program arguments
    args = parse_args()
    print(args)

    # Setting seed
    pl.seed_everything(args["seed"])

    # Getting device
    device = get_device()

    # Creating environment
    # env_name = "ALE/Breakout-v5"
    env_name = "CartPole-v1"
    env = gym.make(env_name)

    # Creating the model (both actor and critic)
    model = MyPPO(env.observation_space.shape, env.action_space.n).to(device)

    # Training
    training_loop(env, model, args["max_iterations"], args["n_actors"], args["horizon"], args["gamma"], args["epsilon"],
                  args["n_epochs"], args["batch_size"], args["lr"], args["c1"], args["c2"], device, env_name)

    # Loading best model
    model = MyPPO(env.observation_space.shape, env.action_space.n).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    # Testing
    env = gym.make(env_name, render_mode="human")
    testing_loop(env, model, args["n_test_episodes"], device)
    env.close()


if __name__ == '__main__':
    main()
