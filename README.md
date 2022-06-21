# Applying representation learning to reinforcement learning 
Using denosing autoencoder to learn compact representaion and using it to train rl agent.

## Collect data while plying Doom
```bash
python play_and_collect.py --image-dir=images
```

## Train an autoencoder using previously collected data 
```bash
python train_ae.py --images-dir=images --model-dir=trained_models
```

## Train PPO agent using compress representation of the state.
```bash
python train_ppo.py --ae-path=trained_models/mymodel.pt
```

## Train DQN agent using compress representation of the state.
```bash
python train_dqn.py --ae-path=trained_models/mymodel.pt
```

