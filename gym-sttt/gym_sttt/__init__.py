from gym.envs.registration import register

register(
    id='sttt-v0',
    entry_point='gym_sttt.envs:StttEnv',
)