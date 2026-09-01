import numpy as np

def get_valid_mask(env):
    mask = np.zeros(env.num_actions, dtype=np.int32)

    px, py = env.pacman_location
    for a, (dx, dy) in env.action.items():
        nx, ny = px + dx, py + dy
        if 0 <= nx < env.width and 0 <= ny < env.height and env.wall.wallmap[ny, nx] == 0:
            mask[a] = 1

    return mask

def obs_to_action_mask(env, obs):
    H, W = env.height, env.width
    N = H * W

    pac_slice = obs[2 * N : 3 * N]
    pac_idx = int(np.argmax(pac_slice))

    py = pac_idx // W
    px = pac_idx % W

    mask = np.zeros(env.num_actions, dtype=np.int32)

    for a, (dx, dy) in env.action.items():
        nx, ny = px + dx, py + dy
        if 0 <= nx < W and 0 <= ny < H and env.wall.wallmap[ny, nx] == 0:
            mask[a] = 1

    return mask