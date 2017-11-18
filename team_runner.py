import program_trace as program_trace
from methods import load_isovist_map, scale_up, direction, load_segs, point_in_obstacle, get_clear_goal
from my_rrt import *
import copy
from scipy.misc import logsumexp
import cPickle
from multiprocessing import Pool
from tqdm import tqdm
import planner
import time
from random import randint




class TeamRunner(object):
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
		self.run_collaborative(Q)

	def run_basic(self, Q):
		# need to simulate which place Alice would go
		popular_loc = Q.flip(0.6)

	def run_collaborative(self,Q, path_noise=0.003):
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
		my_loc = np.atleast_2d(my_plan[t])

		# current location of runner (at time 't')
		#curr_loc = [Q.get_obs("run_x_"+str(t)), Q.get_obs("run_y_"+str(t))]

		# write a mini model of the collaborating runner
		partner_detection_prob = self.partner_model(Q, my_noisy_plan)
		future_partner_detection = Q.flip( p=partner_detection_prob, name="partner_detected" )

		Q.keep("runner_plan", my_noisy_plan)

		# write a mini model of the opposing chaser


	def partner_model(self, Q, collaborators_noisy_plan, path_noise=0.003):
		rx1,ry1,rx2,ry2 = self.seg_map
		# the same time
		t = Q.fetch("t")

		# start and goal locations
		start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="partner_run_start" )
		goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="partner_run_goal" )
 		start = np.atleast_2d( self.locs[start_i] )
		goal = np.atleast_2d( self.locs[goal_i] )

		# plan using the latent variables of start and goal
		my_plan = planner.run_rrt_opt( np.atleast_2d(start), 
		np.atleast_2d(goal), rx1,ry1,rx2,ry2 )
		my_loc = np.atleast_2d(my_plan[t])

		# add noise to the plan
		my_noisy_plan = [my_plan[0]]
		for i in xrange(1, len(my_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
			loc_x = Q.randn( mu=my_plan[i][0], sigma=path_noise, name="partner_run_x_"+str(i) )
			loc_y = Q.randn( mu=my_plan[i][1], sigma=path_noise, name="partner_run_y_"+str(i) )
			loc_t = [loc_x, loc_y]
			my_noisy_plan.append(loc_t)
		my_noisy_plan.append(my_plan[-1])

		# make sure the other runner wan't seen in any of the previous time steps
		#detected_prob = 0
		if self.already_seen(Q): 
			detected_prob = .999
		else:
			# set up collaborator's view (forward vector, fv) for the next step
			cur_collaborator_loc = scale_up(collaborators_noisy_plan[t])
			next_collaborator_loc = scale_up(collaborators_noisy_plan[t+1])
			fv = direction(next_collaborator_loc, cur_collaborator_loc)
			intersections = self.isovist.GetIsovistIntersections(next_collaborator_loc, fv)

			# does the enforcer see me at time 't'
			my_next_loc = scale_up(my_noisy_plan[t+1])
			will_i_be_seen = self.isovist.FindIntruderAtPoint( my_next_loc, intersections )
			detected_prob = 0.999*will_i_be_seen + 0.001*(1-will_i_be_seen) # ~ flip(seen*.999 + (1-seen*.001)

		Q.keep("partner_plan", my_noisy_plan)

		return detected_prob

		

	# XXX returns false
	def already_seen(self, Q):
		return False
		t = Q.fetch("t")
		i_already_seen = 0
		for i in xrange(t):
			intersections = Q.cache["enf_intersections_t_"+str(i)]# get enforcer's fv for time t
			i_already_seen = self.isovist.FindIntruderAtPoint(my_noisy_plan[i], intersections)
			if i_already_seen:
				return True
		return False


	#post_sample_traces = run_inference(q, post_samples=6, samples=5)
	def run_inference(self, program_trace, post_samples, particles):
		post_traces = []
		for i in  tqdm(xrange(post_samples)):
			post_sample_trace = sampling_importance(trace, samples=samples)
			post_traces.append(post_sample_trace)
		return post_traces












