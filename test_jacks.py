from rl import DPAgent

from jacks import JacksRentalEnv


max_car = 1

env = JacksRentalEnv(
    mu_in0 = 1, # at location 0
    mu_in1 = 1, # at location 1
    mu_out0 = 1,
    mu_out1 = 1,

    income_per_car=10,
    cost_per_car_transfer=-1,
    max_car=max_car,
    max_transfer=1
)

assert env.get_state(0) == (0, 0)
assert env.get_state(1) == (0, 1)
assert env.get_state(2) == (1, 0)
assert env.get_state(3) == (1, 1)

assert [env.get_action(i) for i in env.get_allowed_action_ids(0)] == [0]
assert [env.get_action(i) for i in env.get_allowed_action_ids(1)] == [-1, 0]
assert [env.get_action(i) for i in env.get_allowed_action_ids(2)] == [0, 1]
assert [env.get_action(i) for i in env.get_allowed_action_ids(3)] == [-1, 0, 1]

assert env.get_allowed_action_ids(1) == [0, 1]
assert env.get_action(0) == -1
assert env.get_action(1) == 0
assert env.get_action(2) == 1

# poisson distribution with a mu of 0 is a little too extreme, for test only
_rpdb = env.compute_rewards_and_probs(0, 0)
assert _rpdb['probs'][(0, 0)] == 1
assert _rpdb['rewards'][(0, 0)] == 0

_rpdb = env.compute_rewards_and_probs(1, 1)
assert _rpdb['probs'][(0, 0)] ==   0.65425415122906927
assert _rpdb['rewards'][(0, 0)] == 3.4574582867539809


# # sid:0, aid:1, spid: 0
assert env.db['probs'][(0, 1)][0] == 0.428048494400469837 == 0.65425415122906927 ** 2
# spid: 1
assert env.db['probs'][(0, 1)][1] == 0.22620564368101398
assert abs(env.db['probs'][(0, 1)][1] - 0.65425415122906927 * 0.34574582867539809) < 1e-6
# spid: 2
assert env.db['probs'][(0, 1)][2] == 0.22620564368101398
assert abs(env.db['probs'][(0, 1)][2] - 0.34574582867539809 * 0.65425415122906927) < 1e-6
# spid: 3
assert env.db['probs'][(0, 1)][3] == 0.11954017804643793
assert abs(env.db['probs'][(0, 1)][3] - 0.34574582867539809 ** 2) < 1e-6

for k in env.db['probs']:
    # assert prob sum should all be 1
    assert abs(sum(env.db['probs'][k].values()) - 1)  < 1e-6


# Comparing state 3 (i.e. (1, 1)), if take action 0 (+1) would favor state 2 (i.e.
# (0, 1)). Otherwise, it favors state 1 (i.e. (1, 0))

# (3, 0): defaultdict(float,
#              {0: 0.085364817557765718,
#               1: 0.045111719246700197,
#               2: 0.56888932052371799,
#               3: 0.30063410248075162}),
# ...
#  (3, 2): defaultdict(float,
#              {0: 0.085364817557765718,
#               1: 0.56888932052371799,
#               2: 0.045111719246700197,
#               3: 0.30063410248075162})



# Comparing rewards of (1, 0) & (2, 2), they should be symmetrical
assert env.db['rewards'][(1, 0)][2] == env.db['rewards'][(2, 2)][1]
assert env.db['probs'][(1, 0)][2] == env.db['probs'][(2, 2)][1]

agent= DPAgent(env)
for i in range(5):
    agent.evaluate_policy()
    agent.update_policy()

assert agent.pi.tolist() == [1, 1, 1, 0]
