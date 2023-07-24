from gym.envs.registration import register

register(
    id='Othello-v0',
    entry_point='othello.envs.othello:OthelloEnv',
)
