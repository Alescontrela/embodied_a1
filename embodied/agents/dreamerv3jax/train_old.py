import pathlib
import sys
import warnings

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')
warnings.filterwarnings('ignore', '.*using stateful random seeds*')
warnings.filterwarnings('ignore', '.*is a deprecated alias for.*')
warnings.filterwarnings(
    'ignore', '.*truncated to dtype int32.*',
    module='.*tensorflow_probability.*')

directory = pathlib.Path(__file__)
try:
  import google3  # noqa
except ImportError:
  directory = directory.resolve()
directory = directory.parent
sys.path.append(str(directory.parent))
sys.path.append(str(directory.parent.parent.parent))
__package__ = directory.name

import embodied


def main(argv=None):
  from . import agent as agnt

  parsed, other = embodied.Flags(
      method='unused',
      configs=['defaults'],
      worker=0, workers=1,
      learner_addr='localhost:2222',
  ).parse_known(argv)
  config = embodied.Config(agnt.Agent.configs['defaults'])
  for name in parsed.configs:
    config = config.update(agnt.Agent.configs[name])
  config = embodied.Flags(config).parse(other)

  config = config.update(logdir=str(embodied.Path(config.logdir)))
  args = embodied.Config(logdir=config.logdir, **config.run)
  if config.replay == 'chunks':
    min_fill = 1024 * config.env.amount
  else:
    min_fill = config.batch_length * config.env.amount
  args = args.update(
      expl_until=args.expl_until // config.env.repeat,
      train_fill=max(min_fill, args.train_fill),
      eval_fill=max(min_fill, args.eval_fill),
      batch_steps=config.batch_size * config.batch_length,
  )
  print(config)

  logdir = embodied.Path(config.logdir)
  step = embodied.Counter()

  outdir = logdir
  multiplier = config.env.repeat
  if args.script == 'acting':
    outdir /= f'worker{parsed.worker}'
    multiplier *= parsed.workers
  elif args.script == 'learning':
    outdir /= 'learner'
    multiplier = 1
  logger = embodied.Logger(step, [
      embodied.logger.TerminalOutput(config.filter),
      embodied.logger.JSONLOutput(outdir, 'metrics.jsonl'),
      embodied.logger.JSONLOutput(outdir, 'scores.jsonl', 'episode/score'),
      embodied.logger.TensorBoardOutput(outdir),
  ], multiplier)
  try:
    import google3
    logger.outputs.append(embodied.logger.XDataOutput())
  except ImportError:
    pass

  cleanup = []
  try:
    config = config.update({'env.seed': hash((config.seed, parsed.worker))})
    env = embodied.envs.load_env(
        config.task, mode='train', logdir=logdir, **config.env)
    agent = agnt.Agent(env.obs_space, env.act_space, step, config)
    cleanup.append(env)

    if args.script == 'train':
      replay = make_replay(config, logdir / 'episodes')
      embodied.run.train(agent, env, replay, logger, args)

    elif args.script == 'train_eval':
      eval_env = embodied.envs.load_env(
          config.task, mode='eval', logdir=logdir, **config.env)
      replay = make_replay(config, logdir / 'episodes')
      eval_replay = make_replay(config, logdir / 'eval_episodes', is_eval=True)
      embodied.run.train_eval(
          agent, env, eval_env, replay, eval_replay, logger, args)
      cleanup.append(eval_env)

    elif args.script == 'train_fixed_eval':
      if config.eval_dir:
        assert not config.train.eval_fill
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        assert config.run.eval_fill
        eval_replay = make_replay(
            config, logdir / 'eval_episodes', is_eval=True)
      replay = make_replay(config, logdir / 'episodes')
      embodied.run.train_fixed_eval(
          agent, env, replay, eval_replay, logger, args)

    elif args.script == 'learning':
      env.close()
      port = parsed.learner_addr.split(':')[-1]
      #replay = make_replay(config, logdir / 'episodes', server_port=port)
      replay = make_replay(config, server_port=port)
      if config.eval_dir:
        eval_replay = make_replay(config, config.eval_dir, is_eval=True)
      else:
        eval_replay = replay
      embodied.run.learning(agent, replay, eval_replay, logger, args)

    elif args.script == 'acting':
      replay = make_replay(config, remote_addr=parsed.learner_addr)
      embodied.run.acting(agent, env, replay, logger, outdir, args)

    else:
      raise NotImplementedError(args.script)
  finally:
    for obj in cleanup:
      obj.close()


def make_replay(
    config, directory=None, is_eval=False, remote_addr=None,
    server_port=None, **kwargs):
  assert not remote_addr, 'currently unsupported'
  assert not server_port, 'currently unsupported'
  length = config.batch_length
  size = config.replay_size // 10 if is_eval else config.replay_size
  if config.replay == 'uniform' or is_eval:
    replay = embodied.replay.Uniform(
        length, size, directory, config.replay_online)
  elif config.replay == 'reverb':
    replay = embodied.replay.Reverb(length, size, directory)
  elif config.replay == 'prio':
    replay = embodied.replay.Prioritized(length, size, directory)
  elif config.replay == 'chunks':
    replay = embodied.replay.NaiveChunks(length, size, directory)
  else:
    raise NotImplementedError(config.replay)
  return replay


if __name__ == '__main__':
  main()

