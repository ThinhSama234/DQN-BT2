# explain Hyperparameter
import random
import numpy as np
import torch

class Settings:
    # Hyperparameters
    SEED = 7
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    NUM_EPISODES = 300          # increase to 800+ for stronger results
    BUFFER_SIZE = 50_000
    BATCH_SIZE = 128
    GAMMA = 0.99
    LR = 1e-3
    TARGET_SYNC_EVERY = 250
    LEARN_START = 1_000
    LEARN_EVERY = 4
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY_STEPS = 20_000
    MAX_STEPS_PER_EPISODE = 5_000
    GRAD_CLIP = 10.0