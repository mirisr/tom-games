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



		# #------------- model other agent movements (past and future) -------------- XXX need to change RV names
		# o_start_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="other_run_start" )
		# o_goal_i = Q.choice( p=1.0/self.cnt*np.ones((1,self.cnt)), name="other_run_goal" )
		# o_start = np.atleast_2d( self.locs[o_start_i] )
		# o_goal = np.atleast_2d( self.locs[o_goal_i] )

		# # plan using the latent variables of start and goal
		# other_plan = planner.run_rrt_opt( np.atleast_2d(o_start), 
		# np.atleast_2d(o_goal), rx1,ry1,rx2,ry2 )

		# # add noise to the plan
		# other_noisy_plan = [other_plan[0]]
		# for i in xrange(1, len(other_plan)-1):#loc_t = np.random.multivariate_normal(my_plan[i], [[path_noise, 0], [0, path_noise]]) # name 't_i' i.e. t_1, t_2,...t_n
		# 	loc_x = Q.randn( mu=other_plan[i][0], sigma=path_noise, name="other_run_x_"+str(i) )
		# 	loc_y = Q.randn( mu=other_plan[i][1], sigma=path_noise, name="other_run_y_"+str(i) )
		# 	loc_t = [loc_x, loc_y]
		# 	other_noisy_plan.append(loc_t)
		# other_noisy_plan.append(other_plan[-1])
		other_noisy_plan = [[0.42499999999999999, 0.40900000000000003], [0.42637252682699389, 0.45015681250166928], [0.42341590262456913, 0.4990041011765457], [0.4379044695823372, 0.53904127684124392], [0.45788032502031573, 0.55940630406732961], [0.50799614070063526, 0.55464417160785029], [0.54654058973382436, 0.5808784272257409], [0.55952256198212424, 0.61299850534228906], [0.583496600005313, 0.65140274446045765], [0.6005654644648678, 0.69540383781443371], [0.61589998114275823, 0.72963695272460394], [0.63340952989938293, 0.77334914393651799], [0.63415333146793929, 0.81788817596701702], [0.63380864206140508, 0.85068496120846326], [0.6574056904956691, 0.89840736011907418], [0.67365232860101742, 0.92626170086940862], [0.67272997642459853, 0.92476136486233518], [0.670977615078586, 0.9256403675603595], [0.67335036075493437, 0.92761813409888549], [0.67196251574528953, 0.92641384269803029], [0.67098919832850146, 0.92568243002410999], [0.67525871466320719, 0.9229149307167801], [0.67844855732075227, 0.92569329907319242], [0.67321881540381412, 0.93152732857257847], [0.67417822959121554, 0.9200035804784289], [0.67214122482315442, 0.92764945851654945], [0.67164236239673092, 0.92684256939487508], [0.67804513322197213, 0.92666665228204315], [0.6761799161969908, 0.92395809094667547], [0.67542733927721299, 0.92361716626504453], [0.67634231767336883, 0.92658004688912743], [0.67439978040601001, 0.92269389652124378], [0.67429561506189239, 0.92744438186414235], [0.67170106709439659, 0.92538278500943594], [0.67276872621711614, 0.91832099282710455], [0.67247922731692655, 0.92554210689024119], [0.67518132475675507, 0.92729301768019035], [0.6763367594646077, 0.92438316567133738], [0.67522650957951036, 0.92475789453983981], [ 0.675,  0.925]]
	


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

		# just look at the future since we want future detection
		detection_prob = .001
		t_detected = -1
		for i in xrange(5, 20):
			cur_loc = scale_up(my_noisy_plan[i])
			next_loc = scale_up(my_noisy_plan[i+1])
			fv = direction(next_loc, cur_loc)

			intersections = None
			# face the runner if within certain radius
			if dist(my_noisy_plan[i], other_noisy_plan[i]) <= .4: #.35:
				fv = direction(scale_up(other_noisy_plan[i]), cur_loc)
				intersections = self.isovist.GetIsovistIntersections(cur_loc, fv)
			
				# does the enforcer see me at time 't'
				other_loc = scale_up(other_noisy_plan[i])
				will_other_be_seen = self.isovist.FindIntruderAtPoint( other_loc, intersections )
				if will_other_be_seen:
					detection_prob = 0.999
					t_detected = i
					Q.keep("intersections-t-"+str(i), intersections)
					break

			Q.keep("intersections-t-"+str(i), intersections)
		Q.keep("t_detected", t_detected)

		future_detection = Q.flip( p=detection_prob, name="detected" )
			

			

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









		





