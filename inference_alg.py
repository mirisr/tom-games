import program_trace as program_trace
from random import randint
import random
import numpy as np
import copy
from scipy.misc import logsumexp

# importance sampling using the score of trace
def importance_sampling(Q, particles):
	traces = []
	scores = np.arange(particles)

	for i in xrange(particles):
		score, trace_vals = Q.run_model()
		traces.append(copy.deepcopy(trace_vals))
		scores[i] = score

	# get weight for score
	weights = np.exp(scores - logsumexp(scores))

	# sample
	chosen_index = np.random.choice([i for i in range(particles)], p=weights)
	return traces[chosen_index]


def metroplis_hastings(Q, particles):
	traces = []
	scores = np.arange(particles)

	current_score, current_trace = Q.run_model()
	for i in xrange(particles):

		# sample from proposal distribution
		proposed_score, proposed_trace = Q.run_model()

		# accept new sample
		if np.log(random.uniform(0,1)) < proposed_score - current_score:
			current_trace = copy.deepcopy(proposed_trace)
			current_score = proposed_score

	return current_trace





