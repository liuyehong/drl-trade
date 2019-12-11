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
        self.dir = './data/BTCUSDT.csv'
        self.df = pd.read_csv(self.dir)
        self.alpha = Alpha(self.df)
        self.cost = 0.00
        self.interest_rate = 0.0/240/240  # internal interest rate
        self.window = 50
        self.cash = 1
        self.stock = 0
        self.t = self.window + 1
        self.T = len(self.df)
        self.steps = 0
        self.list_asset = np.ones(self.T)
        self.list_holding = np.ones(self.T)
        self.list_profit = np.zeros(self.T)

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
        self.ema = self.alpha.EMA(window=self.window)
        self.mstd = self.alpha.moving_std(window=self.window)
        self.bollinger_lower_bound = self.alpha.bollinger_lower_bound(window=self.window, width=1)
        self.bollinger_upper_bound = self.alpha.bollinger_upper_bound(window=self.window, width=1)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        # Action space range must be symetric and the order matters.

        self.action_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, self.window, 9), dtype=np.float16)


    def _next_observation(self):

        obs = [np.array([
            self.close_diff[self.t - self.window + 1:self.t + 1] / self.close[self.t - self.window + 1],
            self.high_diff[self.t - self.window + 1:self.t + 1] / self.high[self.t - self.window + 1],
            self.open_diff[self.t - self.window + 1:self.t + 1] / self.open[self.t - self.window + 1],
            self.low_diff[self.t - self.window + 1:self.t + 1] / self.low[self.t - self.window + 1],
            self.close[self.t - self.window + 1:self.t + 1] / self.close[self.t - self.window + 1],
            self.high[self.t - self.window + 1:self.t + 1] / self.high[self.t - self.window + 1],
            self.open[self.t - self.window + 1:self.t + 1] / self.open[self.t - self.window + 1],
            self.low[self.t - self.window + 1:self.t + 1] / self.low[self.t - self.window + 1],
            self.list_holding[self.t - self.window + 1:self.t + 1]

        ]).T]

        return obs



    def step(self, action):

        # action[buy/sell/hold]
        print(self.steps, self.t, self.close[self.t]/self.close0, self.list_asset[self.t]/self.asset0, action,self.list_asset[self.t]/self.asset0 -self.close[self.t]/self.close0)
        decision = action[0]
        order_price_b = self.ma[self.t] + self.mstd[self.t] * action[0]
        order_price_s = self.ma[self.t] + self.mstd[self.t] * action[1]

        if self.cash > 0 and order_price_b > self.alpha.close[self.t + 1]:
            take_price = self.alpha.close[self.t + 1]
            self.stock = self.cash / take_price * (1 - self.cost)
            self.cash = 0
            print('buy')

        elif self.stock > 0 and order_price_s < self.alpha.close[self.t + 1]:
            take_price = self.alpha.close[self.t + 1]
            self.cash = self.stock * take_price * (1 - self.cost)
            self.stock = 0
            print('sell')

        self.list_asset[self.t+1] = self.stock*self.alpha.close[self.t+1] + self.cash
        self.list_cash = [self.cash > 0]*self.T
        self.list_holding[self.t+1] = self.cash>0



        reward = (self.list_asset[self.t + 1] - self.list_asset[self.t])/self.list_asset[self.t] #- (self.close[self.t+1]-self.close[self.t])/self.close[self.t]



        done = self.steps > 2000
        self.steps += 1

        obs = self._next_observation()

        self.t += 1
        return obs, reward, done, {}


    def reset(self):
        print('reset')
        self.t = self.window + np.random.random_integers(0, high=self.T-2000)
        self.steps = 0
        self.list_cash = self.T * [1]
        self.list_holding = self.T*[1]


        self.cash = 1
        self.stock = 0
        self.asset0 = 1
        self.close0 = self.close[self.t+1]

        return self._next_observation()

    def render(self, mode='human'):

        pass

