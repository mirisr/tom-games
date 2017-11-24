#from enforcer import *
from methods import load_isovist_map, scale_up, direction, dist, detect, load_segs, get_clear_goal,point_in_obstacle
from my_rrt import *
import isovist as i
from random import randint
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cPickle
from team_runner import TeamRunner
from inference_alg import importance_sampling, metroplis_hastings
from program_trace import ProgramTrace
from planner import * 
from tqdm import tqdm
#import seaborn

def plot(poly_map, plot_name=None, locs=None):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	scale = 1

	# plot map
	x1,y1,x2,y2 = poly_map
	#x1,y1,x2,y2 = polygons_to_segments( load_polygons( "./map_2.txt" ) )
	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )


	add_locations = True
	if add_locations:
		for i in xrange(len(locs)):
			ax.scatter( locs[i][0],  locs[i][1] , color="Green", s = 70, marker='+', linestyle='-')
			ax.scatter( locs[i][0],  locs[i][1] , s = 95, facecolors='none', edgecolors='g')

	
	if plot_name is None:
		plot_name = str(int(time.time()))+".eps"

	ax.set_ylim(ymax = 1, ymin = 0)
	ax.set_xlim(xmax = 1, xmin = 0)

	#plt.show()
	fig.savefig(plot_name, bbox_inches='tight')

def setup_plot(poly_map, locs=None):
	fig = plt.figure(1)
	fig.clf()
	ax = fig.add_subplot(1, 1, 1)
	scale = 1

	# plot map
	x1,y1,x2,y2 = poly_map
	for i in xrange(x1.shape[0]):
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black' )

	# show possible start and goal locations
	add_locations = True
	if add_locations:
		for i in xrange(len(locs)):
			ax.scatter( locs[i][0],  locs[i][1] , color="Green", s = 70, marker='+', linestyle='-')
			ax.scatter( locs[i][0],  locs[i][1] , s = 95, facecolors='none', edgecolors='g')
	return fig, ax

def close_plot(fig, ax, plot_name=None):
	if plot_name is None:
		plot_name = str(int(time.time()))+".eps"

	ax.set_ylim(ymax = 1, ymin = 0)
	ax.set_xlim(xmax = 1, xmin = 0)

	#plt.show()
	fig.savefig(plot_name, bbox_inches='tight')

def plot_runner(poly_map, trace, locs=None):
	fig, ax = setup_plot(poly_map, locs)

	# get time
	t = trace["t"]

	# plot runner's plan
	path = trace["runner_plan"]
	for i in range( 0, len(path)-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle=":", linewidth=2)
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='b')

	# plot partner's plan
	path = trace["partner_plan"]
	for i in range( 0, len(path)-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'grey', linestyle="--", linewidth=2)
	# mark the parter on its plan on time t
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')




	close_plot(fig, ax)


def simulate_running_goal_inference(runner_model, poly_map, locs):
	x1,y1,x2,y2 = poly_map
	# plan for the runner
	start = 4
	goal = 0
	path = run_rrt_opt( np.atleast_2d(locs[start]), 
		np.atleast_2d(locs[goal]), x1,y1,x2,y2 )

	fig, ax = setup_plot(poly_map, locs)

	for i in range( 0, len(path)-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'lightgrey', linestyle="-", linewidth=1)

	# ax.scatter( path[0][0],  path[0][1] , s = 120, facecolors='none', edgecolors='g')
	# ax.scatter( path[-1][0],  path[-1][1] , s = 120, facecolors='none', edgecolors='r')

	close_plot(fig, ax, "true_path.eps")
		

	#for t in xrange(15):

	t = 10
	Q = ProgramTrace(runner_model)

	Q.condition("run_start", 4)
	Q.condition("t", t) 
	# condition on previous time steps
	for prev_t in xrange(t):
		Q.condition("run_x_"+str(prev_t), path[prev_t][0])
		Q.condition("run_y_"+str(prev_t), path[prev_t][1])
		ax.scatter( path[prev_t][0],  path[prev_t][1] , s = 70, facecolors='none', edgecolors='b')
	ax.scatter( path[t][0],  path[t][1] , s = 80, facecolors='none', edgecolors='r')

	post_sample_traces = run_inference(Q, post_samples=32, samples=64)

	# show post sample traces on map
	for trace in post_sample_traces:

		path = trace["runner_plan"]
		for i in range( 0, len(path)-1):
			ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.2)

	close_plot(fig, ax, "infered_goal.eps")



def goal_inference_while_moving(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map
	# plan for the runner
	start = 4
	goal = 1
	path = run_rrt_opt( np.atleast_2d(locs[start]), 
		np.atleast_2d(locs[goal]), x1,y1,x2,y2 )

	fig, ax = setup_plot(poly_map, locs)

	for i in range( 0, len(path)-1):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'lightgrey', linestyle="-", linewidth=1)

	ax.scatter( path[0][0],  path[0][1] , s = 120, facecolors='none', edgecolors='b')
	ax.scatter( path[-1][0],  path[-1][1] , s = 120, facecolors='none', edgecolors='r')

	close_plot(fig, ax, "time/" + sim_id + "_true_path.eps")

	#inferrred_goals = { 0:0, 1:0, 2:0, 3:0, 4:0, 5:0 }
	inferrred_goals = []
	for t in xrange(0, min(25, len(path))):
		fig, ax = setup_plot(poly_map, locs)
		Q = ProgramTrace(runner_model)

		Q.condition("run_start", start)
		Q.condition("t", t) 
		# condition on previous time steps
		for prev_t in xrange(t):
			Q.condition("run_x_"+str(prev_t), path[prev_t][0])
			Q.condition("run_y_"+str(prev_t), path[prev_t][1])
			ax.scatter( path[prev_t][0],  path[prev_t][1] , s = 70, facecolors='none', edgecolors='b')
		ax.scatter( path[t][0],  path[t][1] , s = 80, facecolors='none', edgecolors='r')

		post_sample_traces = run_inference(Q, post_samples=10, samples=16)

		goal_list = []
		# show post sample traces on map
		for trace in post_sample_traces:
			inferred_goal = trace["run_goal"]
			goal_list.append(inferred_goal)
			#print goal_list
			inff_path = trace["runner_plan"]
			for i in range( 0, len(inff_path)-1):
				ax.plot( [inff_path[i][0], inff_path[i+1][0] ], [ inff_path[i][1], inff_path[i+1][1]], 
					'red', linestyle="--", linewidth=1, alpha = 0.2)

		inferrred_goals.append(goal_list)
		print "goal list:", goal_list
		close_plot(fig, ax, "time/" + sim_id + "-post-samples-t-"+str(t)+".eps")

	print "inferrred_goals:", inferrred_goals
	return inferrred_goals, sim_id


def plot_inferred_goals_over_time(inferrred_goals):

	#total_time_steps = len(inferrred_goals)
	#plt.plot(range(total_time_steps), )
	plt.ylabel('probability of goal')
	plt.xlabel('time step')
	plt.savefig('inferred_goals_over_time.eps')

def run_inference(trace, post_samples=16, samples=32):
	post_traces = []
	for i in  tqdm(xrange(post_samples)):
		#post_sample_trace = importance_sampling(trace, samples)
		post_sample_trace = importance_sampling(trace, samples)
		post_traces.append(post_sample_trace)
	return post_traces


def line_plotting(inferrred_goals, sim_id):

	#inferrred_goals =  [[0,1,2,3,4,5], [0,0,1,2,3,4], [0,0,0,1,2], [0,0,0,0,1], [0,0,0,0,0]]

	goal_probabilities = [[], [], [], [], [], []]
	for t in xrange(len(inferrred_goals)):
		inf_goals_at_t = inferrred_goals[t]
		total_num_inferences = len(inf_goals_at_t)
		# turn into percents
		for goal in xrange(6):
			goal_cnt = inf_goals_at_t.count(goal)
			goal_prob = goal_cnt / float(total_num_inferences)
			goal_probabilities[goal].append(goal_prob)

	print "goal_probabilities", goal_probabilities



	fig = plt.figure()
	ax = fig.add_subplot(111)
	for i in xrange(len(goal_probabilities)):
		probs = goal_probabilities[i]
		ax.plot(probs, label="Goal " + str(i))
	ax.legend(loc='upper left')
	ax.ylabel('probability of goal')
	ax.xlabel('time step')
	fig.savefig('time/' + sim_id + '_infgoals_IS_16_32.eps')



if __name__ == '__main__':
	#plot("test.eps")
	# setup
	locs_map_2 = [[0.4, 0.062] ,
			[0.964, 0.064] ,
			[0.442, 0.37] ,
			[0.1, 0.95] ,
			[0.946, 0.90] ,
			[0.066, 0.538]]

	locs = locs_map_2
	poly_map  = polygons_to_segments( load_polygons( "./map_2.txt" ) )
	isovist = i.Isovist( load_isovist_map() )

	runner_model = TeamRunner(seg_map=poly_map, locs=locs, isovist=isovist)
	Q = ProgramTrace(runner_model)

	#simulate_running_goal_inference(runner_model, poly_map, locs)
	inferrred_goals, sim_id= goal_inference_while_moving(runner_model, poly_map, locs)
	line_plotting(inferrred_goals, sim_id)

	#practice_line_plotting()




	# Q.condition("run_start", 0)
	# Q.condition("partner_run_start", 4)

	# score, trace = Q.run_model()

	# print score
	# #print trace["partner_plan"]
	# #print trace["runner_plan"]
	# print "detected:", trace["partner_detected"]
	# print "time:", trace["t"]
	# print "start:", trace["run_start"]
	# print "goal:", trace["run_goal"]
	# plot_runner(poly_map, trace, locs)
	

	#plot(poly_map, locs=locs_map_2)

