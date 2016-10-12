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

for k in env.db['probs']:
    # assert prob sum should all be 1
    assert abs(sum(env.db['probs'][k].values()) - 1)  < 1e-6

# Comparing rewards of (1, 0) & (2, 2), they should be symmetrical
assert env.db['rewards'][(1, 0)][2] == env.db['rewards'][(2, 2)][1]
assert env.db['probs'][(1, 0)][2] == env.db['probs'][(2, 2)][1]


agent= DPAgent(env)

# before training
assert (agent.V == 0).all()
assert agent.pi.tolist() == [1, 0, 1, 0]

# after training via policy_iteration
for i in range(5): agent.policy_iteration(num_iter=5)
assert (agent.V == 0).all() == False
assert (agent.pi == 1).all()

# after reset
agent.reset()
assert (agent.V == 0).all()
assert agent.pi.tolist() == [1, 0, 1, 0]

# after training via value_iteration
agent.value_iteration()
assert (agent.V == 0).all() == False
assert (agent.pi == 1).all()
