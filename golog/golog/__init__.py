from gymnasium.envs.registration import register

register(id='Golog-v0', entry_point='golog.envs:GologEnv')
register(id='Golog-v1', entry_point='golog.envs:GologEnv_v1')