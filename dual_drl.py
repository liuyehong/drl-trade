import gym
import pandas as pd
from dual_env import QtradeEnv
import pickle
import numpy as np

from stable_baselines.common.policies import MlpPolicy, CnnPolicy, LstmPolicy, CnnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2



def fun(n_times):

    # The algorithms require a vectorized environment to run
    env = QtradeEnv()

    # initialization
    vecenv = DummyVecEnv([lambda: env])
    model = PPO2(MlpPolicy, vecenv, verbose=0)

    # dual training
    for k in range(n_times):
        model.learn(total_timesteps=QtradeEnv().total_steps)




if __name__ == '__main__':

    fun(100)






