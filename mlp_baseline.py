import gym
import pandas as pd
from qtrade_env import QtradeEnv
import pickle


from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LstmPolicy, CnnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2


# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: QtradeEnv()])

model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=QtradeEnv().total_steps*20)
model.save('ppo2')
del model

model = PPO2.load('ppo2')


obs = env.reset()
print('---Test begins---')
for i in range(QtradeEnv().total_steps*10):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

