import numpy as np
import networkx as nx


# normalization function
def normalize(d, target=1.0):
	raw = sum(d.values())
	factor = target/raw
	return {key: value*factor for key,value in d.items()}

def AgeBasedPR(G, ages, a, b, alpha, max_iter, tol):

	'''
	INPUT:

		G: Directed networkx graph 
		ages: dic with keys are the node ids and the values are the ages of the nodes in months
		a, b: constants
		alpha: damping factor
		max_iter: maximum number of iterations
		tol: tolerance value for the convergence of power iteration

	OUTPUT:
		x: centrality scores for the nodes of the network

	'''

	# number of nodes 
	N = G.number_of_nodes()

	nx.set_node_attributes(G, 'Age', ages)

	# Add the nodeweight to DiGraph
	for u,d in G.nodes(data=True):
		G.node[u]['NodeWeight'] = 1 + a * np.exp(-b * G.node[u]['Age'])

	# initial vector
	x = dict.fromkeys(G, 1.0 / N)

	# personalization vector
	p = dict.fromkeys(G, 1.0 / N)

	# degree
	degree = G.out_degree(weight=None)

	# dangling vector for dangling nodes
	dangling_weights = p
	dangling_nodes = [n for n in G if degree[n] == 0.0]

	# power iteration
	this_iter = 0
	err = 1.0e6
	while (this_iter < max_iter and err > N*tol):
		xlast = x
		x = dict.fromkeys(xlast.keys(), 0)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for n in x:
			for nbr in G[n]:
				x[nbr] += alpha * xlast[n] * 1.0 / degree[n]
			x[n] += danglesum * dangling_weights[n] 
			x[n] *= G.node[n]['NodeWeight']
			x[n] += (1.0 - alpha) * p[n]
		x = normalize(x, target=1.0)
		# check convergence, L1 norm
		err = sum([abs(x[n] - xlast[n]) for n in x])
		this_iter += 1
	print("Converged at iteration {}".format(this_iter))
	return x

# Example
G = nx.DiGraph()
G.add_edges_from([(0,1),(0,2),(2,1),(3,2)])
ages = {0: 1, 1: 1, 2: 1, 3: 1}

x = AgeBasedPR(G, ages, a=0.003, b=0.04, alpha=0.85, max_iter=100, tol=10e-6)
print(x)




