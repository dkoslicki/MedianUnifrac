
from src import MedianUnifrac as MedU
import numpy as np

(Tint, lint, nodes_in_order) = MedU.parse_tree_file('../data/97_otus_unannotated.tree')
env_dict = MedU.create_env('../data/289_seqs_otus.txt')
(env_prob_dict, samples) = MedU.parse_envs(env_dict, nodes_in_order)

#test parse_tree
def test_parse_tree():
    tree_str = '((B:0.1,C:0.2)A:0.3);'
    (Tint1, lint1, nodes_in_order1) = MedU.parse_tree(tree_str)
    assert Tint1 == {0: 2, 1: 2, 2: 3}
    assert lint1 == {(1, 2): 0.1, (2, 3): 0.3, (0, 2): 0.2}
    assert nodes_in_order1 == ['C', 'B', 'A', 'temp0']  # temp0 is the root node

#test push_up and inverse_push_up
def test_inverse():
    #simple tests
    P1 = np.array([0.1, 0.2, 0,  0.3, 0, 0.3, 0.1])
    T1 = {0: 4, 1: 4, 2: 5, 3: 5, 4: 6, 5: 6}
    l1 = {(0, 4): 0.1, (1, 4): 0.1, (2, 5): 0.2, (3, 5): 0, (4, 6): 0.2, (5, 6): 0.2} # 0 edge_length not involving the root
    nodes1 = ['A', 'B', 'C', 'D', 'temp0', 'temp1', 'temp2']
    P_pushed1 = MedU.push_up(P1, T1, l1, nodes1)
    x = MedU.epsilon * 0.3
    answer1 = np.array([0.01, 0.02, 0, x, 0.06, 0.12, 1])
    assert all(np.abs(P_pushed1 - answer1) < 0.00000001) #test push_up
    assert P_pushed1[3] > 10**-18 #P_pushed[3] (edge length 0) is non-zero
    P_inversed1 = MedU.inverse_push_up(P_pushed1, T1, l1, nodes1)
    assert np.sum(abs(P1 - P_inversed1)) < 10**-10 #test inverse_push_up
    #test with real data
    P2 = env_prob_dict['232.M2Lsft217']
    P_pushed2 = MedU.push_up(P2, Tint, lint, nodes_in_order)
    P_inversed2 = MedU.inverse_push_up(P_pushed2, Tint, lint, nodes_in_order)
    assert np.sum(abs(P2 - P_inversed2)) < 10**-10

#test if push_up computes the correct unifrac value
def test_push_up():
    tree_str = '((B:0.1,C:0.2)A:0.3);'  # there is an internal node (temp0) here.
    (T1, l1, nodes1) = MedU.parse_tree(tree_str)
    nodes_samples = {
        'C': {'sample1': 1, 'sample2': 0},
        'B': {'sample1': 1, 'sample2': 1},
        'A': {'sample1': 0, 'sample2': 0},
        'temp0': {'sample1': 0, 'sample2': 1}}  # temp0 is the root node
    (nodes_weighted, samples_temp) = MedU.parse_envs(nodes_samples, nodes1)
    unifrac1 = np.sum(np.abs(MedU.push_up(nodes_weighted['sample1'], T1, l1, nodes1) -
                  MedU.push_up(nodes_weighted['sample2'], T1, l1, nodes1)))
    assert unifrac1 == 0.25
    #test with real data
    P = env_prob_dict['232.M9Okey217']
    Q = env_prob_dict['232.M3Indl217']
    unifrac2 = np.sum(np.abs(MedU.push_up(P, Tint, lint, nodes_in_order) -
                             MedU.push_up(Q, Tint, lint, nodes_in_order)))
    EMDUnifrac = 0.08224035523709478 #calculated using EMDUnifrac_weighted
    assert np.abs(unifrac2 - EMDUnifrac) < 10**-8

def run_tests():
    test_parse_tree()
    test_inverse()
    test_push_up()

run_tests()