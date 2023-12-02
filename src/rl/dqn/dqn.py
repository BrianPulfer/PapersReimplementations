"""Reimplementation of 'Playing Atari with Deep Reinforcement Learning' by Mnih et al. (2013)"""
import os
import random

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

import wandb


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(
            *random.sample(self.buffer, batch_size)
        )
        return (
            torch.stack(state),
            torch.tensor(action),
            torch.tensor(reward),
            torch.stack(next_state),
            torch.tensor(done).int(),
        )


def dqn_training(
    model,
    environment,
    optimizer,
    gamma,
    epsilon,
    batch_size,
    episodes,
    checkpoint_path,
    buffer_capacity=10000,
):
    # Initialize run
    wandb.init(project="Papers Re-implementations", name="DQN")
    wandb.watch(model)

    # Create checkpoint directory
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Initialize accelerator
    accelerator = Accelerator()
    model, optimizer = accelerator.prepare(model, optimizer)

    # Initialize replay buffer
    buffer = ReplayBuffer(buffer_capacity)

    def state_process_fn(x):
        return torch.tensor(x).float().to(accelerator.device)

    def optimization_step():
        # Sample replay buffer
        state, action, reward, next_state, done = buffer.sample(batch_size)
        action = action.to(accelerator.device)
        reward = reward.to(accelerator.device).float()
        done = done.to(accelerator.device)

        # Compute the target Q value
        with torch.no_grad():
            target_q = reward + gamma * model(next_state).max(1)[0] * (1 - done)

        # Get current Q estimate
        current_q = model(state).gather(1, action.unsqueeze(1))

        # Compute loss
        loss = torch.nn.functional.mse_loss(current_q, target_q.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        return loss.item()

    # Training loop
    checkpoint_loss = float("inf")
    pbar = tqdm(range(episodes))
    for ep in pbar:
        # Initialize episode
        pbar.set_description(f"Episode {ep+1}/{episodes}")
        state = state_process_fn(environment.reset()[0])
        episode_loss, episode_reward, episode_length = 0, 0, 0

        done, truncated = False, False
        while not done and not truncated:
            # Act in the environment
            if random.random() < epsilon:
                action = environment.action_space.sample()
            else:
                action = model(state).argmax().item()

            # Update environment
            next_state, reward, done, truncated, _ = environment.step(action)
            next_state = state_process_fn(next_state)

            # Register transition in replay buffer
            buffer.push(state, action, reward, next_state, done)

            # Update Q-function
            if len(buffer.buffer) >= batch_size:
                loss = optimization_step()
                episode_reward += reward
                episode_loss += loss

            state = next_state
            episode_length += 1

        if len(buffer.buffer) >= batch_size:
            # Log episode stats
            wandb.log(
                {
                    "loss": episode_loss,
                    "reward": episode_reward,
                    "ep. length": episode_length,
                }
            )

        if len(buffer.buffer) >= batch_size and episode_loss < checkpoint_loss:
            torch.save(model.state_dict(), checkpoint_path)
            checkpoint_loss = episode_loss
            print(
                f"Checkpoint saved with loss {checkpoint_loss:.3f} at episode {ep+1} / {episodes}"
            )

    wandb.finish()
