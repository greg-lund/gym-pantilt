from gym.envs.registration import register

register(
    id='pantilt-v0',
    entry_point='gym_pantilt.envs:PanTiltEnv',
)
