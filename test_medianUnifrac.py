import MedianUnifrac as MU
import pickle
import numpy as np

env_dict = pickle.load(open("env_dict.p", "rb"))
Tint = pickle.load(open("Tint.p", "rb"))
Lint = pickle.load(open("Lint.p", "rb"))
nodes_in_order = pickle.load(open("nodes_in_order.p", "rb"))
(env_prob_dict, samples) = MU.parse_envs(env_dict, nodes_in_order)


#simple test
P = env_prob_dict['M2Lsft217']
Q = env_prob_dict['M2Midr217']

test = MU.push_up(P, Tint, Lint, nodes_in_order)
test2 = MU.push_up(Q, Tint, Lint, nodes_in_order)
print(np.sum(np.abs(test-test2)))
#there's an issue with this

#test inverse
P_pushed = MU.push_up(P, Tint, Lint, nodes_in_order)
P_inverse_pushed = MU.inverse_push_up(P_pushed, Tint, Lint, nodes_in_order)
print(np.sum(np.abs(P - P_inverse_pushed)))

#randomized test
is_negative = []
num_its = 10
for num_vectors in range(10, len(samples), 10):
    print (num_vectors)
    for i in range(num_its):
        selected_samples = np.random.choice(list(samples), num_vectors, replace = False)
        Ps = []
        for sample in selected_samples:
            Ps.append(env_prob_dict[sample])
        Ps_pushed = []
        for P in Ps:
            Ps_pushed.append(MU.push_up(P, Tint, Lint, nodes_in_order))
        median = np.median(Ps_pushed, axis=0)
        median_inv = MU.inverse_push_up(median, Tint, Lint, nodes_in_order)
        if np.any(median<0):
            is_negative.append(1)
        else:
            is_negative.append(0)

print is_negative