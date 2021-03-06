# https://github.com/facebookresearch/torchbeast/blob/master/torchbeast/core/environment.py

import numpy as np
from collections import deque
import gym
from gym import spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import cv2
cv2.ocl.setUseOpenCL(False)
import argparse
from distutils.util import strtobool
import collections
import numpy as np
import gym
from gym.wrappers import TimeLimit, Monitor
from vizdoom import DoomGame, Mode, ScreenFormat, ScreenResolution
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
import itertools
from src.autoencoders import Autoencoder

def initialize_vizdoom(config):
    game = DoomGame()
    game.load_config(config)
    #game.set_window_visible(False)
    game.set_mode(Mode.PLAYER)
    game.set_screen_format(ScreenFormat.CRCGCB)
    game.set_screen_resolution(ScreenResolution.RES_160X120)
    game.init()
    return game

def preprocess(img, resolution):
    img = img.transpose((1, 2, 0))
    return cv2.resize(img, resolution).astype(np.float32).transpose((2,0,1))

class Scale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return x * self.scale

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DQN agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--learning-rate', type=float, default=0.00025,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=1000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--buffer-size', type=int, default=10000,
                        help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target-network-frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch-size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start-e', type=float, default=1.0,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end-e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration-fraction', type=float, default=0.4,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning-starts', type=int, default=800,
                        help="timestep to start learning")
    parser.add_argument('--train-frequency', type=int, default=4,
                        help="the frequency of training")
    parser.add_argument('--scale-reward', type=float, default=0.0005,
                        help='scale reward')
    parser.add_argument('--n-step', type=int, default=3,
                        help="n step")
    parser.add_argument('--ae-path', type=str, default="trained_models/batch_size=8_epoch=400_latent_dim=32_06-11-2022_15:15:40_AE.pt")
    parser.add_argument('--latent-dim', default=32, help='latent dim size')

    args = parser.parse_args()
    #if not args.seed:
    args.seed = int(time.time())

# TRY NOT TO MODIFY: setup the environment
experiment_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
    '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))
if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)
    writer = SummaryWriter(f"/tmp/{experiment_name}")

# TRY NOT TO MODIFY: seeding
device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
frame_repeat = 4

width, height, num_channels = 80, 60, 3
frames = 3
game = initialize_vizdoom("./scenarios/health_gathering.cfg")
n = game.get_available_buttons_size()
actions = [list(a) for a in itertools.product([0, 1], repeat=n)]
actions.remove([True, True, True])
actions.remove([True, True, False])

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

class ReplayBufferNStep(object):
    def __init__(self, size, n_step, gamma):
        self._storage = deque(maxlen=size)
        self._maxsize = size
        self.n_step_buffer = deque(maxlen=n_step)
        self.gamma = gamma
        self.n_step = n_step

    def __len__(self):
        return len(self._storage)

    def get_n_step(self):
        _, _, reward, next_observation, done = self.n_step_buffer[-1]
        for _, _, r, next_obs, do in reversed(list(self.n_step_buffer)[:-1]):
            reward = self.gamma * reward * (1 - do) + r
            mext_observation, done = (next_obs, do) if do else (next_observation, done)
        return reward, next_observation, done

    def append(self, obs, action, reward, next_obs, done):
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_obs, done = self.get_n_step()
        obs, action, _, _, _ = self.n_step_buffer[0]
        self._storage.append([obs, action, reward, next_obs, done])

    def sample(self, batch_size):
        idxes = np.random.choice(len(self._storage), batch_size, replace=True)
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def layer_init2(layer, bias_const=1.0):
    torch.nn.init.zeros_(layer.weight)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, actions):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(args.latent_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, len(actions)))
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope =  (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


#rb = ReplayBuffer(args.buffer_size)
rb = ReplayBufferNStep(args.buffer_size, 4, args.gamma)
ae = Autoencoder((num_channels, height, width), args.latent_dim).to(device).eval()
ae.load_state_dict(torch.load(args.ae_path, map_location=device))

q_network = QNetwork(actions).to(device)
target_network = QNetwork(actions).to(device)
target_network.load_state_dict(q_network.state_dict())
optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
loss_fn = nn.MSELoss()
print(device.__repr__())
print(q_network)

# TRY NOT TO MODIFY: start the game
game.new_episode()
episode_reward = 0
obs = game.get_state().screen_buffer
obs = preprocess(obs, (width, height))
episode_length = 0
episode_reward = 0
for global_step in range(args.total_timesteps):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)
    with torch.no_grad():
        obs = torch.from_numpy(obs).float().to(device)
        recon, obs_latent = ae.forward(obs.unsqueeze(0))
        recon_loss = ae.calc_loss(recon, obs / 255.0)

        cv2.imshow("Image", (obs / 255.0).permute(1, 2, 0).cpu().numpy()[:, :, ::-1])
        cv2.imshow("Reconstruction", recon.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        cv2.waitKey(1)
    if random.random() < epsilon:
        action = torch.tensor(random.randint(0, len(actions) - 1)).long().item()
    else:
        logits = q_network.forward(obs_latent)
        action = torch.argmax(logits, dim=1).tolist()[0]

    # TRY NOT TO MODIFY: execute the game and log data.
    reward = game.make_action(actions[action], frame_repeat)
    done = game.is_episode_finished()
    episode_reward += reward
    #reward *= args.scale_reward
    next_obs = preprocess(np.zeros((3, height, width)), (width, height)) if done else preprocess(game.get_state().screen_buffer, (width, height))

    if done:
        writer.add_scalar("charts/episode_reward", episode_reward, global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)
        writer.add_scalar("charts/episode_length", episode_length, global_step)
        writer.add_scalar("charts/epsilon", epsilon, global_step)

    rb.append(obs_latent.squeeze(0).cpu().numpy(), action, reward, next_obs, done)
    if global_step > args.learning_starts and global_step % args.train_frequency == 0:
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(args.batch_size)
        with torch.no_grad():
            _, s_next_obses_latent = ae.forward(torch.from_numpy(s_next_obses).float().to(device))
            target_max = torch.max(target_network.forward(s_next_obses_latent), dim=1)[0]
            td_target = torch.Tensor(s_rewards).to(device) + args.gamma * target_max * (1 - torch.Tensor(s_dones).to(device))
        old_val = q_network.forward(torch.Tensor(s_obs).to(device)).gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
        loss = loss_fn(td_target, old_val)

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)

        # optimize the midel
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(list(q_network.parameters()), args.max_grad_norm)
        optimizer.step()

        # update the target network
        if global_step % args.target_network_frequency == 0:
            target_network.load_state_dict(q_network.state_dict())
    # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    if done:
        game.new_episode()
        obs, episode_reward = game.get_state().screen_buffer, 0
        obs = preprocess(obs, (width, height))
        episode_length = 0
    else:
        obs = next_obs
        episode_length += 1


writer.close()
