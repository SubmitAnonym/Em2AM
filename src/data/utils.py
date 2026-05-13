import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
import tqdm
from os import listdir
from os.path import isfile, join
from tulip import tlp

# import s_gd2




def nx2tlp(nxG):
    n = nxG.number_of_nodes()
    e = nxG.number_of_edges()
    AM = np.zeros((n, n))
    AM[:n, :n] = nx.to_numpy_array(nxG)
    tlpg, mapping = AM2tlp(AM)
    
    assert tlpg.numberOfNodes() == AM.shape[0]
    assert tlpg.numberOfEdges() == e
    return tlpg, mapping

def AM2tlp(AM):
    g = tlp.newGraph()
    N = AM.shape[0]
    nodes = g.addNodes(N)
    real_i, real_j = 0, 0
    id_mapping = {}
    for i in range(N):
        id_mapping[nodes[i].id] = i
        for j in range(i, N, 1):
            if(AM[i][j] == 1):
                g.addEdge(nodes[i], nodes[j])
        
    return g, id_mapping


def applyTlpLayoutAlgorithm(g, algo, propertyName, params_modifs={}):
    params = tlp.getDefaultPluginParameters(algo, g)
    for k, v in params_modifs.items():
        params[k] = v
    res = g.getLayoutProperty(propertyName)
    success, string = g.applyLayoutAlgorithm(algo, res, params)
    if(not success):
        return "fail layout"
    return res

# def applyS_gd2(self, AM, mask):
#     sparseAM = dense_to_sparse(AM)
#     I,J = sparseAM[0], sparseAM[1]
#     I = I.astype("int32")
#     J = J.astype("int32")
#     pos2d = s_gd2.layout(I, J)
#     return pos2d

def create_random_grid(n,d_min,d_max,connected=True):
    g = None
    while g is None:
        d = np.random.randint(d_min,min(n,d_max+1))
        algo = "Grid Approximation"
        params = tlp.getDefaultPluginParameters(algo)
        params['nodes']=n
        params['degree']=d
        g = tlp.importGraph(algo,params)
        if not tlp.ConnectedTest.isConnected(g):
            g = None
    return g

def make_grids(N_range,n_min,n_max,d_min,d_max,dtype=torch.float32):
    for n in range(n_min,n_max+1):
        for j in N_range:
            g = create_random_grid(n,d_min,d_max)
            AM = np.ones((g.numberOfNodes(),g.numberOfNodes()))
            complete_edges = np.array(np.nonzero(AM))
            distances = []
            gnx = tlp2nx(g)
            pos = tlpPos2np(g)
            path_length = dict(nx.all_pairs_shortest_path_length(gnx))
            for u in gnx:
                for v in gnx:
                    distances.append(path_length[u][v])
            yield Data(x=torch.from_numpy(pos).to(dtype=dtype),
                       edge_index=torch.from_numpy(complete_edges).long(),
                       distance_matrix=torch.from_numpy(np.array(distances)).int())
def tlp2nx(g):
    gnx = nx.Graph()
    for i,n in enumerate(g.nodes()):
        gnx.add_node(i)
    for e in g.edges():
        src = g.nodePos(g.source(e))
        tgt = g.nodePos(g.target(e))
        gnx.add_edge(src,tgt)
    return gnx

def tlpPos2np(g,property="viewLayout"):
    pos = np.zeros([g.numberOfNodes(),2])
    vl = g.getLayoutProperty(property)
    for i,n in enumerate(g.nodes()):
        p = vl[n]
        pos[i,0]=p.x()
        pos[i,1]=p.y()
    return pos

def rescale_avg_edge_length(data):
    edges = data.edge_index[:,data.distance_matrix==1]
    avg = torch.linalg.norm(data.x[edges[0]]-data.x[edges[1]],dim=-1).mean()
    data.x = data.x / avg
    



def load_dataset(path="graphs", device="cpu", maxData=None):
    graphFiles = [f for f in listdir(path) if isfile(join(path, f))]
    if maxData is not None:
        graphFiles = graphFiles[:maxData]
    dataset = []
    for f in tqdm.tqdm(graphFiles):
        data = torch.load(f"{path}/{f}", weights_only=False)
        data.distance_matrix=torch.where(data.edge_index[0]==data.edge_index[1],torch.zeros_like(data.distance_matrix),data.distance_matrix).long()
        for i,j in data:
            if "edge_index" in i:
                data[i]=j.long()
        dataset.append(data)
    return dataset
