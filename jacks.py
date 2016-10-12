"""
MDP description: http://web.engr.oregonstate.edu/~tgd/classes/539-old/programs/program1.html

* state: `(n0, n1)`, at the END of the day
* action: transfer `x` cars from loc0 to loc1
* state': `(n0 - x + in0 - out0, n1 + x + in1 - out1)`, need to consider all possibilities of `(in, out)` based on poisson distribution
"""


import itertools
from collections import defaultdict

import numpy as np
import scipy.misc
import scipy.stats

from rl import Environment


class JacksRentalEnv(Environment):
    def __init__(self,  mu_in0, mu_out0, mu_in1, mu_out1,
                 income_per_car, cost_per_car_transfer, max_car, max_transfer):
        """
        :param max_transfer: maximum number of cars that can be transferred per night
        """
        
        self.mu_in0 = mu_in0
        self.mu_out0 = mu_out0
        self.mu_in1 = mu_in1
        self.mu_out1 = mu_out1
        
        self.cost_per_car_transfer = cost_per_car_transfer
        self.income_per_car = income_per_car
        self.max_car = max_car
        self.max_transfer = max_transfer
        
        # as a list of tuple
        self.states = list(itertools.product(range(0, max_car + 1), range(0, max_car + 1)))
        self.actions = range(-max_transfer, max_transfer + 1, 1)
        
        self.num_states = (self.max_car + 1) ** 2
        # e.g. 3 => [-3, -2, -1, 0, 1, 2, 3]
        self.max_num_actions = max_transfer * 2 + 1
        
        self.db = self.compute_db()

    def get_state(self, sid):
        return self.states[sid]
    
    def get_action(self, aid):
        return self.actions[aid]
        
    def get_num_states(self):
        return self.num_states
    
    def get_max_num_actions(self):
        return self.max_num_actions
    
    def get_allowed_action_ids(self, sid):
        """:param sid: index of the state"""
        s0, s1 = self.get_state(sid)
        
        num_1to2 = np.min([ self.max_transfer,  s0])
        num_2to1 = np.max([-self.max_transfer, -s1])

        allowed_actions = range(num_2to1, num_1to2 + 1)

        # shift by max_transfer
        aids = [_ + self.max_transfer for _ in allowed_actions]
        return aids

    def get_next_state_distribution(self, sid, aid):
        """return a object like a list of (spid, prob) tuples"""
        return self.db['probs'][(sid, aid)].items()

    def get_reward(self, sid, aid, spid):
        return self.db['rewards'][(sid, aid)][spid]
    
    def compute_db(self):
        _rpdb0 = self.compute_rewards_and_probs(self.mu_in0, self.mu_out0)
        _rpdb1 = self.compute_rewards_and_probs(self.mu_in1, self.mu_out1)

        db = {
            'rewards': {},
            'probs': {}
        }
        for sid in range(self.get_num_states()):
            state = self.get_state(sid)
            # zero-based index for simplicity. mu is equivalent to lambda in the problem description
            s0, s1 = state
            for aid in self.get_allowed_action_ids(sid):
                action = self.get_action(aid)
                rdb = db['rewards'][(sid, aid)] = defaultdict(float)
                pdb = db['probs'][(sid, aid)] = defaultdict(float)

                # pay attention to abs! It took me hours to debug!
                cost = self.cost_per_car_transfer * abs(action)

                n_beg0 = s0 - action
                n_beg1 = s1 + action

                reward = cost
                for n_end0 in range(self.max_car + 1):
                    for n_end1 in range(self.max_car + 1):
                        sp0, sp1 = n_end0, n_end1
                        next_state = (sp0, sp1)
                        # state_prime_id
                        spid = self.states.index(next_state)
                        
                        key0 = (n_beg0, n_end0)
                        key1 = (n_beg1, n_end1)
                        
                        p0 = _rpdb0['probs'][key0]
                        p1 = _rpdb1['probs'][key1]
                        
                        r0 = _rpdb0['rewards'][key0]
                        r1 = _rpdb1['rewards'][key1]

                        rdb[spid] += p0 * p1 * (cost + r0 + r1)
                        pdb[spid] += p0 * p1
        return db

                    
    def compute_rewards_and_probs(self, mu_in, mu_out):
        """
        pre-compute some data for later use as db:
        1. rewards from the rental part
        2. probility
        """
        max_morning_car = self.max_car + self.max_transfer
        db = {
            'rewards': defaultdict(float),
            'probs': defaultdict(float)
        }
        # n_beg: number of cars at the beginning of the day, some people use morning
        for n_beg in range(max_morning_car + 1):
            # mu + 1: where the probability is very low
            for n_in in range(10 * mu_in + 1):
                for n_out in range(10 * mu_out + 1):
                    prob = (scipy.stats.poisson.pmf(n_in, mu_in) * 
                            scipy.stats.poisson.pmf(n_out, mu_out))
                    
                    # n_in is not added to n_beg here as shown in the commented
                    # line because "Cars can only become available for renting
                    # the day after they are returned" according to the book
                    # real_out = min(n_beg + n_in, n_out)
                    real_out = min(n_beg, n_out)


                    n_end = min(self.max_car, n_beg + n_in - real_out)

                    # this loses a lot of value
                    # st = i + n_in - real_out
                    key = (n_beg, n_end)

                    db['probs'][key] += prob
                    db['rewards'][key] += prob * self.income_per_car * real_out
        return db
