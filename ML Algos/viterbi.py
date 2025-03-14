import numpy as np

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
        for y in states:
            (prob, state) = max(
                [(V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states]
            )
            V[t][y] = prob
            newpath[y] = path[state] + [y]
        path = newpath
    (prob, state) = max([(V[-1][y], y) for y in states])
    return (prob, path[state])

states = ('S0', 'S1')
observations = ('A1', 'A2', 'A3')
start_probability = {'S0': 0.6, 'S1': 0.4}
transition_probability = {
   'S0' : {'S0': 0.7, 'S1': 0.3},
   'S1' : {'S0': 0.4, 'S1': 0.6},
   }
emission_probability = {
   'S0' : {'A1': 0.1, 'A2': 0.4, 'A3': 0.5},
   'S1' : {'A1': 0.6, 'A2': 0.3, 'A3': 0.1},
   }

print(viterbi(observations, states, start_probability, transition_probability, emission_probability))
