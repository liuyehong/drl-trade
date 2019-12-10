import gym
import pandas as pd
from qtrade_env import QtradeEnv
root_dir = '/Users/liuyehong/Dropbox/CICC/Algorithm_Trading/Platform2/OHLC/data/1Min/'
import pickle


from stable_baselines.common.policies import *
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from stable_baselines.common.policies import FeedForwardPolicy, register_policy, nature_cnn


class CustomLSTMPolicy(LstmPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=64, reuse=False, **_kwargs):
        super().__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
                         net_arch=[8, 'lstm', dict(vf=[5, 10], pi=[10])],
                         layer_norm=True, feature_extraction="mlp", **_kwargs)


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: QtradeEnv()])

model = PPO2(CustomLSTMPolicy, env, verbose=1, nminibatches=1)
model.learn(total_timesteps=50000)
model.save('ppo2_mlplnlstm')

del model
model = PPO2.load('ppo2_mlplnlstm', env=env)

obs = env.reset()
for i in range(20000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
