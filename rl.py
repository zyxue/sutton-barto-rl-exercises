import itertools
import numpy as np
import scipy.misc
import scipy.stats
from collections import defaultdict


class Environment(object):
    """Abstract class defined with required methods, which are used to interact
    with agent"""

    def get_num_states(self):
        pass

    def get_allowed_actions(self, state_id):
        """
        state_id: an id uniquely identified a state for this environment, the
        exact form of object for representing the state doesn't matter
        """
        pass

    def get_next_state_distribution(state_id, action_id):
        """
        action_id: similar to state_id, uniquely identifies an action, returns a
        object like a list of (spid, prob) tuples"""
        pass

    def get_reward(self, state_id, action_id, next_state_id):
        pass

    
class Agent(object):
    def evaluate_policy(self):
        pass

    def update_policy(self):
        pass


class DPAgent(object):
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma

        self.num_states = self.env.get_num_states()
        self.sids = range(self.num_states)

        # init values for all states
        self.V = np.zeros(self.num_states)

        # init policy randomly at each state

        self.pi = np.array([
            np.random.choice(self.env.get_allowed_action_ids(_)) for _ in self.sids])

    def evaluate_policy(self, counter_cutoff=100):
        delta = 1
        counter = 0
        while delta > 0.1 and counter < counter_cutoff:
            delta = 0
            for sid in self.sids:
                old_v = self.V[sid]
                new_v = 0
                aid = self.pi[sid]
                for (spid, prob) in self.env.get_next_state_distribution(sid, aid):
                    new_v += prob * (self.env.get_reward(sid, aid, spid) + self.gamma * self.V[spid])
                self.V[sid] = new_v
                delta = max(delta, abs(old_v - new_v))
            counter += 1
            
    def update_policy(self, counter_cutoff=100):
        counter = 0
        policy_stable = False
        while not policy_stable and counter < counter_cutoff:
            policy_stable = True
            for sid in self.sids:
                action_vals = []
                for aid in self.env.get_allowed_action_ids(sid):
                    val = 0
                    for (spid, prob) in self.env.get_next_state_distribution(sid, aid):
                        val += prob * (self.env.get_reward(sid, aid, spid) + self.gamma * self.V[spid])
                    action_vals.append((aid, val))
                best_aid = max(action_vals, key=lambda x: x[1])[0]
                if self.pi[sid] != best_aid:
                    self.pi[sid] = best_aid
                    policy_stable = False
            counter += 1

    # def value_iteration(self, counter_cutoff):
    #     delta = 1
    #     counter = 0
    #     while delta > 0.1 and counter < counter_cutoff:
    #         delta = 0
    #         for sid in self.sids:
    #             old_v = self.V[sid]
    #             new_v = 0
    #             aid = self.pi[sid]
    #             for (spid, prob) in self.env.get_next_state_distribution(sid, aid):
    #                 new_v += prob * (self.env.get_reward(sid, aid, spid) + self.gamma * self.V[spid])
    #             self.V[sid] = new_v
    #             delta = max(delta, abs(old_v - new_v))
    #         counter += 1
        
