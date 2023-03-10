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
from embodied.envs import a1

# a1_env = a1.A1()
a1_env = embodied.envs.load_env('a1_walk', repeat=4, resets=False)

def get_random_action():
    # action = a1_env.act_space['action'].sample()
    reset = a1_env.act_space['reset'].sample()
    action = 0.0 * np.ones((12,), dtype=np.float32)
    return {'action': np.repeat(np.expand_dims(action, 0), len(a1_env), axis=0), 'reset': reset}


print(a1_env.obs_space)

while True:
    # input()
    print('step')
    obs = a1_env.step(get_random_action())
    for k, v in obs.items():
        print('\t', k, v.shape, v.dtype, v.min(), v.max())
    
    print('is_first', obs['is_first'])
    print('is_last', obs['is_last'])
    print('is_terminal', obs['is_terminal'])

    print(obs['projected_gravity'])

