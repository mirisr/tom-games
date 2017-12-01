import program_trace as program_trace
from methods import load_isovist_map, scale_up, direction, dist, load_segs, point_in_obstacle, get_clear_goal
from my_rrt import *
import copy
from scipy.misc import logsumexp
import cPickle
from multiprocessing import Pool
from tqdm import tqdm
import planner
import time
from random import randint
from program_trace import ProgramTrace
from inference_alg import importance_sampling, metroplis_hastings



class BasicRunner(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		# field of view calculator
		self.isovist = isovist
		# possible start/goal locations
		self.locs = locs
		self.cnt = len(locs)
		# map
		self.seg_map = seg_map
		rx1,ry1,rx2,ry2 = seg_map


	# run the model inside this function
	def run(self, Q):
		self.run_basic(Q)

	def run_basic(self, Q, path_noise=0.003):
		rx1,ry1,rx2,ry2 = self.seg_map
		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )

		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_goal" )
		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		# plan using the latent variables of start and goal
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
		my_loc = np.atleast_2d(my_plan[t])

		# add noise to the plan
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="run_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		Q.keep("runner_plan", my_noisy_plan)


	#post_sample_traces = run_inference(q, post_samples=6, samples=5)
	def run_inference(self, program_trace, post_samples, particles):
		post_traces = []
		for i in  tqdm(xrange(post_samples)):
			post_sample_trace = sampling_importance(trace, samples=samples)
			post_traces.append(post_sample_trace)
		return post_traces



class BasicRunnerPOM(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None]):
		# field of view calculator
		self.isovist = isovist
		# possible start/goal locations
		self.locs = locs
		self.cnt = len(locs)
		# map
		self.seg_map = seg_map
		rx1,ry1,rx2,ry2 = seg_map


	# run the model inside this function
	def run(self, Q):
		self.run_basic_partial(Q)

	def run_basic_partial(self, Q, path_noise=0.003):
		rx1,ry1,rx2,ry2 = self.seg_map
		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )



		#------------- model other agent movements (past and future) -------------- XXX need to change RV names
		o_start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="other_run_start" )
		o_goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="other_run_goal" )
		o_start = np.atleast_2d( self.locs[o_start_i] )
		o_goal = np.atleast_2d( self.locs[o_goal_i] )

		# plan using the latent variables of start and goal
		other_plan = planner.run_rrt_opt( np.atleast_2d(o_start), 
		np.atleast_2d(o_goal), rx1,ry1,rx2,ry2 )

		# add noise to the plan
		other_noisy_plan = [other_plan[0]]
		for i in xrange(1, len(other_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=other_plan[i][0], sigma=path_noise, name="other_run_x_"+str(i) )
			loc_y = Q.randn( mu=other_plan[i][1], sigma=path_noise, name="other_run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			other_noisy_plan.append(loc_t)
		other_noisy_plan.append(other_plan[-1])


		#------------- model agent's own movements (past and future) --------------
		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="run_goal" )
		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		# plan using the latent variables of start and goal
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
		

		# add noise to the plan
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="run_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		my_loc = my_noisy_plan[t]

		#---------------- need to add RV of detection for each time step ----------
		# detection_prob = 0.0
		# for i in xrange(t):
		# 	past_detection = Q.flip( p=detection_prob, name="detected_t_"+str(i) )

		for i in xrange(0, 39):
			cur_loc = scale_up(my_noisy_plan[i])
			next_loc = scale_up(my_noisy_plan[i+1])
			fv = direction(next_loc, cur_loc)

			intersections = None
			# face the runner if within certain radius
			if dist(my_noisy_plan[i], other_noisy_plan[i]) <= .35:
				fv = direction(scale_up(other_noisy_plan[i]), cur_loc)
				intersections = self.isovist.GetIsovistIntersections(cur_loc, fv)
			
				# does the enforcer see me at time 't'
				other_loc = scale_up(other_noisy_plan[i])
				will_other_be_seen = self.isovist.FindIntruderAtPoint( other_loc, intersections )
				detection_prob = 0.999*will_other_be_seen + 0.001*(1-will_other_be_seen) # ~ flip(seen*.999 + (1-seen*.001)
				future_detection = Q.flip( p=detection_prob, name="detected_t_"+str(i) )
			else:
				future_detection = Q.flip( p=.001, name="detected_t_"+str(i) )

			Q.keep("intersections-t-"+str(i), intersections)

		Q.keep("my_plan", my_noisy_plan)
		Q.keep("other_plan", other_noisy_plan)


	#post_sample_traces = run_inference(q, post_samples=6, samples=5)
	def run_inference(self, program_trace, post_samples, particles):
		post_traces = []
		for i in  tqdm(xrange(post_samples)):
			post_sample_trace = sampling_importance(trace, samples=samples)
			post_traces.append(post_sample_trace)
		return post_traces


class TOMCollabRunner(object):
	def __init__(self, isovist=None, locs=None, seg_map=[None,None,None,None], nested_model=None):
		# field of view calculator
		self.isovist = isovist
		# possible start/goal locations
		self.locs = locs
		self.cnt = len(locs)
		# map
		self.seg_map = seg_map
		rx1,ry1,rx2,ry2 = seg_map
		self.nested_model = nested_model


	# run the model inside this function
	def run(self, Q):
		self.tom_run_nested_inference(Q)


	def tom_run_nested_inference(self, Q, path_noise=0.003):
		# represents the other agent
		rx1,ry1,rx2,ry2 = self.seg_map
		t = Q.choice( p=1.0/40*np.ones((1,40)), name="t" )

		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="co_run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="co_run_goal" )
		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		# plan using the latent variables of start and goal
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
		my_loc = np.atleast_2d(my_plan[t])

		# add noise to the plan
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="co_run_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="co_run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		post_sample_traces = self.collaborative_nested_inference(Q)

		# get most probable goal
		partners_goal = self.get_most_probable_goal(post_sample_traces, "run_goal")
		
		same_goal = 0
		if goal_i == partners_goal:
			same_goal = 1
		same_goal_prob = 0.999*same_goal + 0.001*(1-same_goal)

		runners_same_goal = Q.flip( p=same_goal_prob, name="same_goal" ) 

		Q.keep("co_runner_plan", my_noisy_plan)
		Q.keep("partner_goal", partners_goal)
		Q.keep("nested_post_sample_traces", post_sample_traces)


	def collaborative_nested_inference(self, Q):
		t = Q.get_obs("t")
		q = ProgramTrace(self.nested_model)
		q.condition("run_start", Q.get_obs("run_start"))
		q.condition("t", Q.get_obs("t")) 
		# condition on previous time steps
		
		for prev_t in xrange(t):
			q.condition("run_x_"+str(prev_t), Q.get_obs("run_x_"+str(prev_t)))
			q.condition("run_y_"+str(prev_t), Q.get_obs("run_y_"+str(prev_t)))
		
		post_sample_traces = self.run_inference(q, post_samples=10, samples=16)
		return post_sample_traces

	def run_inference(self, trace, post_samples=16, samples=32):
		post_traces = []
		for i in  tqdm(xrange(post_samples)):
			#post_sample_trace = importance_sampling(trace, samples)
			post_sample_trace = importance_sampling(trace, samples)
			post_traces.append(post_sample_trace)
		return post_traces


	def get_most_probable_goal(self, post_sample_traces, goal_rv_name):
		goal_list = []
		# show post sample traces on map
		for trace in post_sample_traces:
			inferred_goal = trace[goal_rv_name]
			goal_list.append(inferred_goal)

		# list with probability for each goal
		goal_probabilities = []
		total_num_inferences = len(goal_list)
		# turn into percents
		for goal in xrange(6):
			goal_cnt = goal_list.count(goal)
			goal_prob = goal_cnt / float(total_num_inferences)
			goal_probabilities.append(goal_prob)

		return goal_probabilities.index(max(goal_probabilities))









		





