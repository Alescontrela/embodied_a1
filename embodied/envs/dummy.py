import embodied
import numpy as np


class Dummy(embodied.Env):

  def __init__(self, task, size=(64, 64), length=100):
    assert task in ('continuous', 'discrete')
    self._task = task
    self._size = size
    self._length = length
    self._step = 0
    self._done = False

  @property
  def obs_space(self):
    return {
        'image': embodied.Space(np.uint8, self._size + (3,)),
        'step': embodied.Space(np.int32, (), 0),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
    }

  @property
  def act_space(self):
    if self._task == 'continuous':
      space = embodied.Space(np.float32, 6)
    else:
      space = embodied.Space(np.int32, (), 0, 5)
    return {'action': space, 'reset': embodied.Space(bool)}

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      return self._obs(0.0, is_first=True)
    action = action['action']
    if self._task == 'continuous':
      assert (-1 <= action).all() and (action <= 1).all(), action
    else:
      assert action in range(5), action
    self._step += 1
    self._done = (self._step >= self._length)
    return self._obs(1.0, is_last=self._done)

  def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
    return dict(
        image=np.zeros(self._size + (3,), np.uint8),
        step=self._step,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
    )