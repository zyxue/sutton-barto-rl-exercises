"""
Example 4.3: Gambler’s Problem
"""


import itertools
from collections import defaultdict

import numpy as np
import scipy.misc
import scipy.stats

from rl import Environment


class GamblerEnv(Environment):
    def __init__(self, max_capital=100, prob_head=0.5):
        """
        :param max_capital: the goal, after reaching which the gambler wins
        """
        self.max_capital = max_capital
        self.prob_head = prob_head

        # the first and last state are dummy states signaling the start and end
        # of games
        self.states = list(range(max_capital + 1))
        # None state is the next_state after exit action, which signals the end
        # of one episod
        self.NONE_STATE = 'NONE_STATE'
        self.states.append(self.NONE_STATE)
        self.state2sid = dict(zip(self.states, range(len(self.states))))

        # all possible actions
        self.actions = list(range(1, int(max_capital / 2) + 1))
        self.EXIT_ACTION = 'exit'
        self.NONE_ACTION = None
        self.actions.append(self.EXIT_ACTION) # exit action
        self.actions.append(self.NONE_ACTION) # action for NONE_STATE
        self.action2aid = dict(zip(self.actions, range(len(self.actions))))

    def get_state(self, sid):
        if sid >= 0:
            return self.states[sid]
    
    def get_action(self, aid):
        if aid >= 0:
            return self.actions[aid]

    def get_state_id(self, state):
        return self.state2sid[state]

    def get_action_id(self, action):
        return self.action2aid[action]

    def get_num_states(self):
        return len(self.states)
    
    # def get_max_num_actions(self):
    #     return len(self.actions)
    
    def get_allowed_action_ids(self, sid):
        """:param sid: index of the state"""
        state = self.get_state(sid)
        
        if (state == 0 or state == self.max_capital):
            return [self.get_action_id(self.EXIT_ACTION)]

        if state == self.NONE_STATE:
            return [self.get_action_id(self.NONE_ACTION)]

        allowed_actions = range(1, min(state, self.max_capital - state) + 1)
        aids = [self.get_action_id(_) for _ in allowed_actions]
        return aids

    def get_allowed_actions(self, sid):
        return [self.get_action(__) for __ in self.get_allowed_action_ids(sid)]

    def get_next_state_distribution(self, sid, aid):
        """return a object like a list of (spid, prob) tuples"""
        state = self.get_state(sid)
        action = self.get_action(aid)

        if action == self.EXIT_ACTION:
            return [(self.get_state_id(self.NONE_STATE), 1)]
        
        if action == self.NONE_ACTION:
            return []

        ns_w = min(state + action, self.max_capital) # win
        ns_f = max(state - action, 0)                # lose
        
        return [(self.get_state_id(ns_w), self.prob_head),
                (self.get_state_id(ns_f), 1 - self.prob_head)]

    def get_reward(self, sid, aid, spid):
        if self.get_action(aid) == self.EXIT_ACTION:
            if self.get_state(sid) == 0: # lose
                return 0
            elif self.get_state(sid) == self.max_capital:
                return 1
        return 0
