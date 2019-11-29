import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
from alpha import Alpha
import numpy as np
import os
import datetime


class QtradeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(QtradeEnv, self).__init__()
        self.root_dir = '/Users/liuyehong/Dropbox/CICC/Algorithm_Trading/Platform2/OHLC/data/1Min/'
        self.list_dir = [d for d in os.listdir(self.root_dir) if '.csv' in d]
        self.df_dir = np.random.choice(self.list_dir)
        self.df = pd.read_csv(self.root_dir + self.df_dir)
        self.alpha = Alpha(self.df)
        self.cost = 0.00
        self.interest_rate = 0.1/240/240  # internal interest rate
        self.window = 50
        self.cash = 1
        self.stock = 0
        self.t = self.window + 1
        self.T = len(self.df)
        self.list_asset = np.ones(self.T)

        # alpha
        self.close = self.alpha.close
        self.high = self.alpha.high
        self.low = self.alpha.low
        self.open = self.alpha.open
        self.vol = self.alpha.vol
        self.close_diff = self.alpha.close_diff()
        self.high_diff = self.alpha.high_diff()
        self.low_diff = self.alpha.low_diff()
        self.open_diff = self.alpha.open_diff()

        self.ma = self.alpha.moving_average(window=self.window)
        self.mstd = self.alpha.moving_std(window=self.window)
        self.bollinger_lower_bound = self.alpha.bollinger_lower_bound(window=self.window, width=1)
        self.bollinger_upper_bound = self.alpha.bollinger_upper_bound(window=self.window, width=1)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        # Action space range must be symetric and the order matters.

        self.action_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf]), high=np.array([np.inf, np.inf, np.inf]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.window, 9), dtype=np.float16)

    def _next_observation(self):

        obs = np.array([
            self.close_diff[self.t-self.window+1:self.t+1]/self.close[self.t - self.window + 1],
            self.high_diff[self.t-self.window+1:self.t+1]/self.high[self.t - self.window + 1],
            self.open_diff[self.t-self.window+1:self.t+1]/self.open[self.t - self.window + 1],
            self.low_diff[self.t-self.window+1:self.t+1]/self.low[self.t - self.window + 1],
            self.close[self.t - self.window + 1:self.t + 1]/self.close[self.t - self.window + 1],
            self.high[self.t - self.window + 1:self.t + 1]/self.high[self.t - self.window + 1],
            self.open[self.t - self.window + 1:self.t + 1]/self.open[self.t - self.window + 1],
            self.low[self.t - self.window + 1:self.t + 1]/self.low[self.t - self.window + 1],
            self.list_cash[self.t - self.window + 1:self.t + 1]
               ]).T

        return obs

    def step(self, action):
        # action[buy/sell/hold]
        print(self.t, action, self.cash/self.asset0)
        decision = action[0]
        order_price_b = self.close[self.t] - self.mstd[self.t] * action[1]
        order_price_s = self.close[self.t] + self.mstd[self.t] * action[2]

        if self.cash > 0 and order_price_b > self.alpha.low[self.t+1] and decision > 0:
            self.stock = self.cash/order_price_b*(1-self.cost)
            self.cash = 0
            print('buy')

        elif self.stock > 0 and order_price_s < self.alpha.high[self.t+1] and decision < 0:
            self.cash = self.stock*order_price_s*(1-self.cost)
            self.stock = 0
            print('sell')

        self.list_asset[self.t+1] = self.stock*self.alpha.close[self.t+1] + self.cash
        self.list_cash = [self.cash > 0]*self.T

        if self.cash > 0:
            reward = -self.interest_rate  # penalty for holding cash.
        else:
            reward = (self.list_asset[self.t + 1] - self.list_asset[self.t])/self.list_asset[self.t]

        done = self.t > 2000

        obs = self._next_observation()

        self.t += 1
        return obs, reward, done, {}


    def reset(self):
        print('reset')
        self.t = self.window
        self.list_cash = self.T * [1]

        # random initialization
        if np.random.rand() > 0.5:
            self.cash = 1
            self.stock = 0
            self.asset0 = 1
        else:
            self.cash = 0
            self.stock = 1
            self.asset0 = self.stock*self.close[self.t]


        self.df_dir = np.random.choice(self.list_dir)
        self.df = pd.read_csv(self.root_dir + self.df_dir)
        rnd_int = np.random.randint(0, len(self.df) - 2000)
        self.df = self.df[rnd_int:rnd_int+2000]

        return self._next_observation()

    def render(self, mode='human'):

        print(self.t, self.list_asset[self.t], self.close[self.t]/self.close[self.window])

