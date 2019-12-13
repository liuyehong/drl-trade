import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd
from alpha import Alpha
import numpy as np
import os
import datetime

# The positive region is in the upper bollinger band. How should we let the algorithm converges in this global opt?
# The algo always stuck in the loer bollinger band, some pretrain might be important
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
        self.cost = -0.0001
        self.interest_rate = 0./240/240  # internal interest rate
        self.window = 50
        self.cash = 1
        self.stock = 0
        self.t = self.window + 1
        self.i = 0
        self.T = len(self.df)
        self.total_steps = self.T -self.window - 2
        self.list_asset = np.ones(self.T)
        self.list_holding = np.ones(self.T)

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
        self.dema = self.alpha.DEMA(window=self.window)
        self.kama = self.alpha.KAMA(window=self.window)
        self.sma = self.alpha.SMA(window=self.window)
        self.tema = self.alpha.TEMA(window=self.window)
        self.trima = self.alpha.TRIMA(window=self.window)
        self.linearreg_slope = self.alpha.LINEARREG_SLOPE(window=self.window)


        self.mstd = self.alpha.moving_std(window=self.window)
        self.bollinger_lower_bound = self.alpha.bollinger_lower_bound(window=self.window, width=1)
        self.bollinger_upper_bound = self.alpha.bollinger_upper_bound(window=self.window, width=1)
        self.moving_max = self.alpha.moving_max(window=self.window)
        self.moving_min = self.alpha.moving_min(window=self.window)
        self.moving_med = self.alpha.moving_med(window=self.window)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        # Action space range must be symetric and the order matters.

        self.action_space = spaces.Box(
            low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, self.window, 1), dtype=np.float16)


    def _next_observation(self):

        obs = [np.array([
            self.close[self.t - self.window + 1:self.t + 1] / self.ma[self.t ],
            #self.high[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.open[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.low[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.ma[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.ema[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.dema[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.kama[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.sma[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.tema[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.trima[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.bollinger_lower_bound[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #self.bollinger_upper_bound[self.t - self.window + 1:self.t + 1] / self.ma[self.t],
            #np.zeros(self.window), # find optimize window size with constant observation.
            #self.list_holding[self.t - self.window + 1:self.t + 1],
            #self.list_cash[self.t - self.window + 1:self.t + 1],

        ]).T]

        return obs

    def _utility(self, x): #
        if x > 0:
            return 1*x
        else:
            return 1*x

    def step(self, action):

        # action[buy/sell/hold]
        order_price_b = np.floor(100*(self.ma[self.t] + self.mstd[self.t] * action[0]))/100.
        order_price_s = np.ceil(100*self.ma[self.t] + self.mstd[self.t] * action[1])/100.

        if self.cash > 0 and order_price_b > self.close[self.t]:
            take_price = self.close[self.t]
            self.stock = self.cash/take_price*(1-self.cost)
            self.cash = 0
            print('buy: ' + str(take_price))
            print(self.steps, self.t, self.close[self.t] / self.close0, self.list_asset[self.t] / self.asset0, action,
                  self.list_asset[self.t] / self.asset0 - self.close[self.t] / self.close0)


        elif self.stock > 0 and order_price_s < self.close[self.t]:
            take_price = self.close[self.t]
            self.cash = self.stock*take_price*(1-self.cost)
            self.stock = 0
            print('sell: ' + str(take_price))
            print(self.steps, self.t, self.close[self.t] / self.close0, self.list_asset[self.t] / self.asset0, action,
                  self.list_asset[self.t] / self.asset0 - self.close[self.t] / self.close0)


        self.list_asset[self.t+1] = self.stock*self.alpha.close[self.t+1] + self.cash
        self.list_cash = [self.cash > 0]*self.T
        self.list_holding[self.t+1] = self.cash>0


        # it is important to use relative return as a reward
        reward = (self.list_asset[self.t + 1] - self.list_asset[self.t])/self.list_asset[self.t] - \
                 (self.close[self.t + 1] - self.close[self.t]) / self.close[self.t]

        if self.cash>0:
            reward += -self.interest_rate


        done = self.steps > self.total_steps
        self.steps += 1
        obs = self._next_observation()

        self.t += 1
        return obs, reward, done, {}


    def reset(self):
        # To avoid stuck in local opt, it is important to increase the cost step by step
        if self.cost<0.0005:
            self.cost += 0.0001
            print('cost'+str(self.cost))
        self.i += 1

        self.df_dir = np.random.choice(self.list_dir)
        self.df = pd.read_csv(self.root_dir + self.df_dir)

        print('reset: ' + str(self.i))
        self.t = 1 + self.window + np.random.random_integers(0, self.T-self.total_steps-self.window-1)
        self.list_cash = self.T * [1]
        self.list_holding = self.T*[1]
        self.steps = 0


        self.cash = self.close[self.t]
        self.stock = 0
        self.asset0 = self.close[self.t]
        self.close0 = self.close[self.t]


        return self._next_observation()

    def render(self, mode='human'):

        pass

