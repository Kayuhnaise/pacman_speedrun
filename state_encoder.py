import numpy as np


def encode_state(env, obs):
    """
    Encode the PacmanEnv observation into a 6-channel grid:
    channel 0 = walls
    channel 1 = pellets
    channel 2 = power pellets
    channel 3 = pacman
    channel 4 = active ghosts
    channel 5 = scared mode indicator (filled with 1 if scared_timer > 0 else 0)

    Returns:
        np.ndarray of shape (6, rows, cols), dtype float32
    """
    rows, cols = env.rows, env.cols
    state = np.zeros((6, rows, cols), dtype=np.float32)

    for r, c in env.wall_cells:
        state[0, r, c] = 1.0

    for r, c in obs["pellets"]:
        state[1, r, c] = 1.0

    for r, c in obs["power_pellets"]:
        state[2, r, c] = 1.0

    pr, pc = obs["pacman"]
    state[3, pr, pc] = 1.0

    for g in obs["ghosts"]:
        if g["active"]:
            gr, gc = g["pos"]
            state[4, gr, gc] = 1.0

    if obs["scared_timer"] > 0:
        state[5, :, :] = 1.0

    return state