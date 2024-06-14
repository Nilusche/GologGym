from gymnasium.envs.registration import register

register(id='Golog-v0', entry_point='golog.envs:GologEnv')
register(id='Golog-v1', entry_point='golog.envs:GologEnv_v1')
register(id='Golog-v2', entry_point='golog.envs:GologEnv_v2')
register(id='Golog-v3', entry_point='golog.envs:GologEnv_v3')