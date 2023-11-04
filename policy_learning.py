import torch.nn as nn
import torch.optim as optim
import torch
import gymnasium as gym

lr = 1e-2
env = gym.make("LunarLander-v2")
actions = env.action_space.n  # 4
states = env.observation_space.shape[0]  # 8
gamma = 0.99


class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Policy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.model(x)


def train_model(i, episodes, policy, gamma):
    # print('here')
    loss = compute_loss(episodes, policy, gamma)
    global avg_loss
    avg_loss += loss
    print(f'loss: {loss} avg loss: {avg_loss/i}')
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


model = Policy(states, actions)
optimizer = optim.Adam(params=model.parameters(), lr=lr)
state = torch.tensor(env.reset()[0], dtype=torch.float32)
avg_loss = 0


def compute_loss(episodes, policy, gamma):
    total_loss = 0
    for episode in episodes:
        actions, rewards, states = zip(*episode)
        # if len(episode) > 1:
        #   print(actions)

        reward_history = []
        R = 0
        # R + gamma R'
        for reward in reversed(rewards):
            R = reward + R * gamma
            reward_history.insert(0, R)

        actions = torch.stack(actions).unsqueeze(1)  # n by 1
        states = torch.stack(states)  # n by 8
        reward_history = torch.stack(reward_history).unsqueeze(1)  # n by 8
        # print(actions.shape, states.shape)
        prob = torch.log(policy(states).gather(1, actions)) * reward
        loss = torch.sum(prob)
        total_loss -= loss

    return total_loss/len(episodes)


for episode in range(1000):
    state = torch.tensor(env.reset()[0], dtype=torch.float32)
    history = []
    for trajectories in range(10):
        done = False
        trajectory = []
        while not done:
            action_probs = model(state)
            action = torch.multinomial(action_probs, 1).item()  # [n]
            observation, reward, done, truncated, info = env.step(
                action)  # [n]
            next_state = torch.tensor(
                observation, dtype=torch.float32)  # [n,8]

            if reward > 100:
                print("reward ", reward)
            trajectory.append((torch.tensor(action, dtype=torch.int64), torch.tensor(
                reward, dtype=torch.float32), state))
            state = next_state
        history.append(trajectory)
    train_model(episode, history, model, gamma)

env.close()
