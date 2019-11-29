import gym
import pandas as pd
from qtrade_env import QtradeEnv
root_dir = '/Users/liuyehong/Dropbox/CICC/Algorithm_Trading/Platform2/OHLC/data/1Min/'
import pickle


from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LstmPolicy, CnnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: QtradeEnv()])

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=50000)
model.save('ppo2')
del model

model = PPO2.load('ppo2')

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
