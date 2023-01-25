import pathlib
import sys

directory = pathlib.Path(__file__)
directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import numpy as np

import embodied
from embodied.envs import shadow_hand

# ShadowHand_env = shadow_hand.ShadowHand("cuda:0", 0)
shadow_hand_env = embodied.envs.load_env("shadow_hand", rl_device="cuda:0", sim_device="cuda:0", graphics_device_id=0)

def get_random_action():
    # action = a1_env.act_space['action'].sample()
    reset = shadow_hand_env.act_space['reset'].sample()
    action = 0.0 * np.ones((20,), dtype=np.float32)
    return {'action': np.repeat(np.expand_dims(action, 0), len(shadow_hand_env), axis=0), 'reset': reset}


# print(a1_env.obs_space)

while True:
    # input()
    print('step')
    obs, rewards, resets, info = shadow_hand_env.step(get_random_action())
    for k, v in obs.items():
        print('\t', k, v.shape, v.dtype, v.min(), v.max())
            
    print('is_first', obs['is_first'])
    print('is_last', obs['is_last'])
    print('is_terminal', obs['is_terminal'])
    