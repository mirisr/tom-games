#from enforcer import *
from methods import load_isovist_map, scale_up, direction, dist, detect, load_segs, get_clear_goal,point_in_obstacle
from my_rrt import *
import isovist as i
from random import randint
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import cPickle
from team_runner import BasicRunner
from team_runner import TOMCollabRunner
from team_runner import BasicRunnerPOM
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
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black', linewidth=1 )


	add_locations = False
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
		ax.plot( [ x1[i,0] * scale, x2[i,0] * scale ], [ y1[i,0] * scale, y2[i,0] * scale], 'black', linewidth=1  )

	# show possible start and goal locations
	add_locations = True
	if add_locations:
		for i in xrange(len(locs)):
			ax.scatter( locs[i][0],  locs[i][1] , color="Green", s = 50, marker='+', linestyle='-')
			ax.scatter( locs[i][0],  locs[i][1] , s = 75, facecolors='none', edgecolors='g')
	return fig, ax

def close_plot(fig, ax, plot_name=None):
	if plot_name is None:
		plot_name = str(int(time.time()))+".eps"
	print "plot_name:", plot_name

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
			'black', linestyle=":", linewidth=1)
		# if you want to show each time step in blue
	close_plot(fig, ax, plot_name=str(int(time.time()))+"-1.eps")

	for i in range( 0, len(path)):
		ax.scatter( path[i][0],  path[i][1] , s = 35, facecolors='none', edgecolors='blue')

	# mark the runner at time t on its plan
	#ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='b')

	# # plot partner's plan
	# path = trace["partner_plan"]
	# for i in range( 0, len(path)-1):
	# 	ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
	# 		'grey', linestyle="--", linewidth=1)
	# # mark the parter on its plan on time t
	# ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	close_plot(fig, ax, plot_name=str(int(time.time()))+"-2.eps")


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

# This simulates two agents performing goal inference on one another after each move
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


def get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, path, start, t, character):
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

	close_plot(fig, ax, "collab/" + sim_id + character + "-post-samples-t-"+str(t)+".eps")

	# list with probability for each goal
	goal_probabilities = []
	total_num_inferences = len(goal_list)
	# turn into percents
	for goal in xrange(6):
		goal_cnt = goal_list.count(goal)
		goal_prob = goal_cnt / float(total_num_inferences)
		goal_probabilities.append(goal_prob)

	return goal_probabilities.index(max(goal_probabilities))

# This simulates two agents performing nested goal inference using "TOM"
def two_agent_nested_goal_inference_while_moving(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map

	#Alice will start at some location
	alice_start = 4
	alice_path = [locs[alice_start]]

	#Bob will start st some other location
	bob_start = 5
	bob_path = [locs[bob_start]]

	alices_inferrred_goals_for_bob = []
	bobs_inferrred_goals_for_alice = []
	# for each time step
	for t in xrange(0, 25):
		#Alice will conduct goal inference on observations of bob's location
		Q = add_Obs(ProgramTrace(runner_model), alice_start, t, alice_path)
		inferred_bob_goal, bobs_goal_list = nested_most_probable_goal_location(Q, poly_map, locs, sim_id, 
			bob_path, bob_start, t, "B")
		
		#Bob will conduct goal inference on observations of alice's location
		Q = add_Obs(ProgramTrace(runner_model), bob_start, t, bob_path)
		inferred_alice_goal, alices_goal_list = nested_most_probable_goal_location(Q, poly_map, locs, sim_id, 
			alice_path, alice_start, t,"A")

		alices_inferrred_goals_for_bob.append(bobs_goal_list)
		bobs_inferrred_goals_for_alice.append(alices_goal_list)

		#Alice will move toward Bob's goal after planning
		alice_plan = run_rrt_opt( np.atleast_2d(alice_path[-1]), 
		np.atleast_2d(locs[inferred_bob_goal]), x1,y1,x2,y2 )
		alice_path.append(alice_plan[1])

		#Bob will move toward Alice's goal after planning
		bob_plan = run_rrt_opt( np.atleast_2d(bob_path[-1]), 
		np.atleast_2d(locs[inferred_alice_goal]), x1,y1,x2,y2 )
		bob_path.append(bob_plan[1])

		plot_movements(alice_path, bob_path, sim_id, poly_map, locs, t, code="nested")

	line_plotting(alices_inferrred_goals_for_bob, sim_id, code="A-in-B", directory="tom-collab")
	line_plotting(bobs_inferrred_goals_for_alice, sim_id, code="B-in-A", directory="tom-collab")

def nested_most_probable_goal_location(Q, poly_map, locs, sim_id, path, start, t, character):
	fig, ax = setup_plot(poly_map, locs)

	Q.condition("co_run_start", start)
	Q.condition("t", t) 
	# condition on previous time steps
	for prev_t in xrange(t):
		Q.condition("co_run_x_"+str(prev_t), path[prev_t][0])
		Q.condition("co_run_y_"+str(prev_t), path[prev_t][1])
		ax.scatter( path[prev_t][0],  path[prev_t][1] , s = 70, facecolors='none', edgecolors='b')
	ax.scatter( path[t][0],  path[t][1] , s = 80, facecolors='none', edgecolors='r')

	Q.condition("same_goal", True)

	post_sample_traces = run_inference(Q, post_samples=10, samples=16)

	goal_list = []
	# show post sample traces on map
	for trace in post_sample_traces:
		inferred_goal = trace["co_run_goal"]
		goal_list.append(inferred_goal)
		#print goal_list
		inff_path = trace["co_runner_plan"]
		for i in range( 0, len(inff_path)-1):
			ax.plot( [inff_path[i][0], inff_path[i+1][0] ], [ inff_path[i][1], inff_path[i+1][1]], 
				'red', linestyle="--", linewidth=1, alpha = 0.2)

	close_plot(fig, ax, "tom-collab/" + sim_id + character + "-post-samples-t-"+str(t)+".eps")

	# list with probability for each goal
	goal_probabilities = []
	total_num_inferences = len(goal_list)
	# turn into percents
	for goal in xrange(6):
		goal_cnt = goal_list.count(goal)
		goal_prob = goal_cnt / float(total_num_inferences)
		goal_probabilities.append(goal_prob)

	return goal_probabilities.index(max(goal_probabilities)), goal_list



def plot_movements(a_path, b_path, sim_id, poly_map, locs, t, code=""):
	fig, ax = setup_plot(poly_map, locs)
	# PLOTTING MOVEMENTS
	path = a_path
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle=":", linewidth=2, label="Alice")
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	path = b_path
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle="--", linewidth=2, label="Bob")
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')
	close_plot(fig, ax, "collab/" + sim_id + code + "-" +str(t)+".eps")



def follow_the_leader_goal_inference(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map

	#Alice will start at some location
	alice_start = 4
	goal = 3
	alice_path = run_rrt_opt( np.atleast_2d(locs[alice_start]), 
		np.atleast_2d(locs[goal]), x1,y1,x2,y2 )

	#Bob will start st some other location
	bob_start = 5
	#bob_path = [locs[bob_start]]
	bob_path = [[0.2, 0.15]]

	# for each time step
	for t in xrange(0, 25):
		
		#Bob will conduct goal inference on observations of alice's location
		inferred_alice_goal = get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, 
			alice_path, alice_start, t, "FL-A")

		#Bob will move toward Alice's goal after planning
		bob_plan = run_rrt_opt( np.atleast_2d(bob_path[-1]), 
		np.atleast_2d(locs[inferred_alice_goal]), x1,y1,x2,y2 )
		bob_path.append(bob_plan[1])

		plot_movements(alice_path, bob_path, sim_id, poly_map, locs, t, code="FL-t")

def add_Obs(Q, start, t, path):
	Q.set_obs("run_start", start)

	Q.set_obs("t", t)
	# condition on previous time steps
	for prev_t in xrange(t):
		Q.set_obs("run_x_"+str(prev_t), path[prev_t][0])
		Q.set_obs("run_y_"+str(prev_t), path[prev_t][1])
	return Q


def two_agent_goal_inference_while_moving(runner_model, poly_map, locs):
	sim_id = str(int(time.time()))
	x1,y1,x2,y2 = poly_map

	#Alice will start at some location
	alice_start = 4
	alice_path = [locs[alice_start]]

	#Bob will start st some other location
	bob_start = 5
	bob_path = [locs[bob_start]]

	# for each time step
	for t in xrange(0, 25):
		#Alice will conduct goal inference on observations of bob's location
		inferred_bob_goal = get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, 
			bob_path, bob_start, t, "B")
		
		#Bob will conduct goal inference on observations of alice's location
		inferred_alice_goal = get_most_probable_goal_location(runner_model, poly_map, locs, sim_id, 
			alice_path, alice_start, t,"A")

		#Alice will move toward Bob's goal after planning
		alice_plan = run_rrt_opt( np.atleast_2d(alice_path[-1]), 
		np.atleast_2d(locs[inferred_bob_goal]), x1,y1,x2,y2 )
		alice_path.append(alice_plan[1])

		#Bob will move toward Alice's goal after planning
		bob_plan = run_rrt_opt( np.atleast_2d(bob_path[-1]), 
		np.atleast_2d(locs[inferred_alice_goal]), x1,y1,x2,y2 )
		bob_path.append(bob_plan[1])

		plot_movements(alice_path, bob_path, sim_id, poly_map, locs, t, code="double-goal")



def run_inference(trace, post_samples=16, samples=32):
	post_traces = []
	for i in  tqdm(xrange(post_samples)):
		#post_sample_trace = importance_sampling(trace, samples)
		post_sample_trace = importance_sampling(trace, samples)
		post_traces.append(post_sample_trace)
	return post_traces


def line_plotting(inferrred_goals, sim_id, code="", directory="time"):

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
	ax.set_ylabel('probability of goal')
	ax.set_xlabel('time step')
	fig.savefig( directory +'/' + sim_id + code + '_infering_goals.eps')

def run_basic_partial_model(locs, poly_map, isovist):
	runner_model = BasicRunnerPOM(seg_map=poly_map, locs=locs, isovist=isovist)
	Q = ProgramTrace(runner_model)
	Q.condition("run_start", 2)
	Q.condition("run_goal", 6)
	Q.condition("other_run_start", 5)
	Q.condition("other_run_goal", 8)
	Q.condition("t", 10)

	score, trace = Q.run_model()

	fig, ax = setup_plot(poly_map, locs, )

	path = trace["my_plan"]
	t = trace["t"]
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle=":", linewidth=2, label="Agent's Plan")
		


	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='orange')

	path = trace["other_plan"]
	for i in range(0, t):
		ax.plot( [path[i][0], path[i+1][0] ], [ path[i][1], path[i+1][1]], 
			'black', linestyle="--", linewidth=1, label="Other's Plan")
		if trace["detected_t_"+ str(i)] == True:
			ax.scatter( path[i][0],  path[i][1] , s = 50, facecolors='none', edgecolors='red')
		else:
			ax.scatter( path[i][0],  path[i][1] , s = 30, facecolors='none', edgecolors='grey')
	# mark the runner at time t on its plan
	ax.scatter( path[t][0],  path[t][1] , s = 95, facecolors='none', edgecolors='blue')

	close_plot(fig, ax, plot_name="PO_forward_runs/run_and_catch-"+str(int(time.time()))+".eps")

	print "time:", trace["t"]
	print "other_run_start:", trace["other_run_start"]
	print "other_run_goal:", trace["other_run_goal"]

	print "run_start:", trace["run_start"]
	print "run_goal:", trace["run_goal"]


if __name__ == '__main__':
	#plot("test.eps")


	# ------------- setup for map 2 ---------------
	# locs_map_2 = [[0.4, 0.062] ,
	# 		[0.964, 0.064] ,
	# 		[0.442, 0.37] ,
	# 		[0.1, 0.95] ,
	# 		[0.946, 0.90] ,
	# 		[0.066, 0.538]]
	# locs = locs_map_2
	# poly_map  = polygons_to_segments( load_polygons( "./map_2.txt" ) )
	# isovist = i.Isovist( load_isovist_map( fn="./map_2.txt" ) )

	# ------------- setup for map "paths" large bremen map ---------------
	locs = [[ 0.100, 1-0.900 ],[ 0.566, 1-0.854 ],[ 0.761, 1-0.665 ],
		[ 0.523, 1-0.604 ],[ 0.241, 1-0.660 ],[ 0.425, 1-0.591 ],
		[ 0.303, 1-0.429 ],[ 0.815, 1-0.402 ],[ 0.675, 1-0.075 ],
		[ 0.432, 1-0.098 ] ]
	poly_map = polygons_to_segments( load_polygons( "./paths.txt" ) )
	isovist = i.Isovist( load_isovist_map() )

	#plots the map and the locations if said so in the function
	#plot(seg_map, plot_name="large_map_blank.eps", locs=locs)
	


	#---------------- Basic Runner model -------------------------------
	#runner_model = BasicRunner(seg_map=poly_map, locs=locs, isovist=isovist)
	# --------- run goal inference on new observations ----------------
	# inferrred_goals, sim_id= goal_inference_while_moving(runner_model, poly_map, locs)
	# line_plotting(inferrred_goals, sim_id)

	# --------- follow the leader using goal inference tools ----------
	# follow_the_leader_goal_inference(runner_model, poly_map, locs)

	# --------- first experiment of agent "collaboration" -------------
	#two_agent_goal_inference_while_moving(runner_model, poly_map, locs)

	#---------- nested collab experiment ------------------------------
	#tom_runner_model = TOMCollabRunner(seg_map=poly_map, locs=locs, isovist=isovist, nested_model=runner_model)
	#two_agent_nested_goal_inference_while_moving(tom_runner_model, poly_map, locs)

	# ---------- plot generative model of simple runner model
	# Q = ProgramTrace(runner_model)
	# Q.condition("run_start", 4)
	# Q.condition("run_goal", 0)
	# score, trace = Q.run_model()
	# plot_runner(poly_map, trace, locs=locs)
	# print("time:", trace["t"])
	# print("start:", trace["run_start"])
	# print("goal:", trace["run_goal"])

	#-----------run basic partially observable model and plot----------
	#for i in xrange(10):
	run_basic_partial_model(locs, poly_map, isovist)



	# old -----------------------------
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

