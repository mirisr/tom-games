from program_trace import ProgramTrace
from random import randint
import random
import numpy as np
import copy
from scipy.misc import logsumexp
from multiprocessing import Pool



# importance sampling using the score of trace
def single_run_model(arg, **kwarg):
	return ProgramTrace.run_model(*arg, **kwarg)
	

def single_run_model_sarg( x ):
	if not hasattr( single_run_model_sarg, '__my_init' ):
		single_run_model_sarg.__my_init = True # use function attributes as static variables
		import os
		np.random.seed( os.getpid() )
	return single_run_model()



def importance_sampling_par(Q, particles):
	traces = []
	scores = np.arange(particles)

	# for i in xrange(particles):
		# score, trace_vals = Q.run_model()
		# traces.append(copy.deepcopy(trace_vals))
		# scores[i] = score

#   p = Pool( 12 )
	p = Pool( 5 )
    # we do all of this because Pool.map pickles its arguments, and you can't pickle a lambda...
	params = ((),) * particles
	#print "params:", params
	#results = p.map( single_run_model_sarg, zip([Q]*particles, ((),) * particles) )

	for tmp_retval in results:
		scores.append( tmp_retval[0] )
		traces.append( tmp_retval[1] )

	print "scores:", scores

	
	# get weight for score
	# weights = np.exp(scores - logsumexp(scores))

	# # sample
	# chosen_index = np.random.choice([i for i in range(particles)], p=weights)
	# return traces[chosen_index]
	return 0


from multiprocessing import Pool 
from functools import partial 


def _getattr_proxy_partialable(instance, name, arg): 
    return getattr(instance, name)(arg) 

def getattr_proxy(instance, name): 
    """ 
    A version of getattr that returns a proxy function that can 
    be pickled. Only function calls will work on the proxy. 
    """ 
    return partial(_getattr_proxy_partialable, instance, name) 

import copy_reg 
import multiprocessing 
import new 

def make_instancemethod(inst, methodname): 
    return getattr(inst, methodname) 

def pickle_instancemethod(method): 
    return make_instancemethod, (method.im_self, method.im_func.__name__) 

# copy_reg.pickle( 
#     types.MethodType, 
#     lambda method: (getattr, (method.im_self, method.im_func.__name__)), 
#     getattr 
# ) 

copy_reg.pickle( 
    new.instancemethod, pickle_instancemethod, make_instancemethod) 


# # importance sampling using the score of trace
def importance_sampling(Q, particles):

	#pool = Pool(4) 
	#op = p.map(getattr_proxy(Q, "run_model"), ((),) * particles) 
	#print pool.map(Q.run_model, ((),) * particles)  
	#print(op) 


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





