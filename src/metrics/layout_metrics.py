from model.neural_network import NormalizeRotationAndScaleShift
import numpy as np
from scipy.special import comb
import torch_geometric
from torch_geometric.data import Batch, Data
from torch_geometric.utils import degree, to_networkx, to_undirected, remove_self_loops,sort_edge_index
import networkx as nx
import torch
import torch_geometric
import tsnet_utils


def min_crossing_angle(graph:Data):
    g:nx.Graph = to_networkx(Data(edge_index=graph.edge_index[:,graph.distance_matrix==1]))

    min_angle = torch.pi
    for n,adj in g.adjacency():
        if(len(adj))> 1:
            pos = graph.x[n].unsqueeze(0)
            pos_tgts = torch.stack([graph.x[tgt] for tgt in adj],dim=0)
            diff = pos_tgts-pos
            norm = torch.linalg.norm(diff,dim=-1,keepdim=True)
            unit = diff/norm
            angles = torch.atan2(unit[:,1],unit[:,0])
            angles = torch.where(angles<0,angles+torch.pi*2,angles)
            angles = torch.sort(angles)[0]
            for i in range(0,len(angles)):
                a  = angles[(i+1)] if i+1<len(angles) else (angles[0]+torch.pi*2)
                min_angle = min(min_angle,a-angles[i])
            
    return min_angle
"""

doi 10.4230/LIPIcs.GD.2025.30

neighbourhood_preservation

"""
def neighbourhood_preservation(graph:Data):
    edge_index = graph.edge_index[:,graph.distance_matrix==1] if hasattr(graph,"distance_matrix") else graph.edge_index
    k = int(np.floor(2*edge_index.shape[1]/graph.num_nodes))
    knn_edge_index = torch_geometric.nn.pool.knn_graph(graph.x,k=k,loop=False,batch=torch.zeros([graph.num_nodes],dtype=torch.long))
    in_graph = torch.stack([
        torch.ones([edge_index.shape[1]]),
        torch.zeros([edge_index.shape[1]]),
    ],dim=-1)
    in_knn = torch.stack([
        torch.zeros([knn_edge_index.shape[1]]),
        torch.ones([knn_edge_index.shape[1]]),
    ],dim=-1)

    ei = torch.cat([edge_index,knn_edge_index],dim=-1)
    ea = torch.cat([in_graph,in_knn],dim=0)
    coalesce_edge_index , in_graph_knn = torch_geometric.utils.coalesce(ei,ea)
    intersection = (in_graph_knn!=0).all(dim=-1).sum()
    union = (in_graph_knn!=0).any(dim=-1).sum()
    return intersection/union
"""
doi 10.4230/LIPIcs.GD.2025.30
node resolution
"""
def node_resolution(graph:Data):
    edge_index = graph.edge_index[:,graph.edge_index[0]!=graph.edge_index[1]]
    distance = torch.linalg.norm(graph.x[edge_index[1]]-graph.x[edge_index[0]],dim=-1)
    return distance.min()/distance.max()
"""
doi 10.4230/LIPIcs.GD.2025.30
node Uniformity
"""
def node_uniformity(graph:Data):
    num_rows = np.floor(np.sqrt(graph.num_nodes))
    num_cols = np.ceil(graph.num_nodes/num_rows)
    x = graph.x[:,0]
    y = graph.x[:,1]
    cols = ((x-x.amin())/(x.amax()-x.amin())*num_cols).floor()
    cols= torch.where(cols>=num_cols,num_cols-1,cols)

    rows = ((y-y.amin())/(y.amax()-y.amin())*num_rows).floor()
    rows= torch.where(rows>=num_rows,num_rows-1,rows)
    cell = (cols+num_cols*rows).long()
    num_cells = int(num_rows*num_cols)
    num_per_cell = degree(cell,num_cells)
    d_max = 2*graph.num_nodes*(num_cells-1)/num_cells
    return 1-(num_per_cell-graph.num_nodes/num_cells).abs().sum()/d_max
"""

doi 10.4230/LIPIcs.GD.2025.30

aspect ratio
"""
def aspect_ratio(graph:Data):
    
    bb = graph.x.amax(dim=0)-graph.x.amin(dim=0)
    return bb.amin()/bb.amax()

def edge_length_uniformity(graph:Data):
    distance = torch.linalg.norm(graph.x[graph.edge_index[1]]-graph.x[graph.edge_index[0]],dim=-1)
    print(distance)
    return (torch.sum((distance-1)**2)/graph.edge_index.shape[1]).sqrt()
"""

doi 10.4230/LIPIcs.GD.2025.30

Edge Length Deviation (ELD) 
assumes graph normalized to average edge length 1 (like in training)
"""
def edge_length_deviation_Purchase(graph:Data):
    edge_index = graph.edge_index[:,graph.distance_matrix==1] if hasattr(graph,"distance_matrix") else graph.edge_index
    distance = torch.linalg.norm(graph.x[edge_index[1]]-graph.x[edge_index[0]],dim=-1)
    return 1/(1+((distance-1).abs()).mean())



"""
https://arxiv.org/pdf/1808.00703

Edge Length Variation (Ml)
assumes graph normalized to average edge length 1 (like in training)
"""
def edge_length_deviation_Haleem(graph:Data):
    edge_index = graph.edge_index[:,graph.distance_matrix==1] if hasattr(graph,"distance_matrix") else graph.edge_index
    distance:torch.Tensor = torch.linalg.norm(graph.x[edge_index[1]]-graph.x[edge_index[0]],dim=-1)
    l_a = ((distance-1).square().sum()/(edge_index.shape[1])).sqrt()
    return l_a/np.sqrt(edge_index.shape[1])


def edges_pair(graph:Data):
    edges = graph.edge_index[:,graph.distance_matrix==1] if hasattr(graph,"distance_matrix") and graph.distance_matrix!=None else graph.edge_index
    edges = to_undirected(edges)
    edges = edges[:,edges[0]<edges[1]]
    first_edges = []
    second_edges = []
    for i in range(edges.shape[1]):
        e1=edges[:,i]
        for j in range(i+1,edges.shape[1]):
            e2=edges[:,j]
            if(torch.cat((e1,e2),dim=0).unique().shape[0]==4):
                first_edges.append(i)
                second_edges.append(j)
    return edges,torch.stack([torch.tensor(first_edges),torch.tensor(second_edges)],dim=0)
def edge_is_crossing(e1_src:torch.Tensor,e1_tgt:torch.Tensor,e2_src:torch.Tensor,e2_tgt:torch.Tensor):
    t, u = t_u_crossing(e1_src, e1_tgt, e2_src, e2_tgt)
    return torch.logical_and(
        torch.logical_and(t>0,t<1),
        torch.logical_and(u>0,u<1)
    )
def t_u_crossing(e1_src, e1_tgt, e2_src, e2_tgt):
    denom = torch.linalg.det(torch.stack([e1_src-e1_tgt,e2_src-e2_tgt],dim=-2))
    t = torch.linalg.det(torch.stack([
            e1_src-e2_src,
            e2_src-e2_tgt
        ],dim=-2))/denom
    u = -torch.linalg.det(torch.stack([
            e1_src-e1_tgt,
            e1_src-e2_src
        ],dim=-2))/denom

    return t,u
"""
Metrics for Graph Drawing Aesthetics
Helen C. Purchase
"""
def edge_crossing(graph:Data):
    g = Data(edge_index=graph.edge_index[:,graph.distance_matrix==1])
    node_degree =degree(g.edge_index[0],g.num_nodes)
    c_max = comb(g.edge_index.shape[1],2)-(comb(node_degree,2).sum())

    edges,pairs = edges_pair(g)
    e1_src =edges[0,pairs[0,:]]
    e1_tgt = edges[1,pairs[0,:]]
    e2_src = edges[0,pairs[1,:]]
    e2_tgt = edges[1,pairs[1,:]]

    e1_src_pos = graph.x[e1_src]
    e1_tgt_pos = graph.x[e1_tgt]
    e2_src_pos = graph.x[e2_src]
    e2_tgt_pos = graph.x[e2_tgt]

    return 1 -(edge_is_crossing(e1_src_pos,e1_tgt_pos,e2_src_pos,e2_tgt_pos).int().sum())/c_max

"""
Universal Quality Metrics for Graph Drawings:
Which Graphs Excite Us Most
"""  
def crossing_angle(graph:Data):
    g = Data(edge_index=graph.edge_index[:,graph.distance_matrix==1])
    edges,pairs = edges_pair(g)
    e1_src =edges[0,pairs[0,:]]
    e1_tgt = edges[1,pairs[0,:]]
    e2_src = edges[0,pairs[1,:]]
    e2_tgt = edges[1,pairs[1,:]]

    e1_src_pos = graph.x[e1_src]
    e1_tgt_pos = graph.x[e1_tgt]
    e2_src_pos = graph.x[e2_src]
    e2_tgt_pos = graph.x[e2_tgt]

    mask = edge_is_crossing(e1_src_pos,e1_tgt_pos,e2_src_pos,e2_tgt_pos)
    e1_src_pos = e1_src_pos[mask]
    e1_tgt_pos = e1_tgt_pos[mask]
    e2_src_pos = e2_src_pos[mask]
    e2_tgt_pos = e2_tgt_pos[mask]
    
    diff1 = e1_tgt_pos-e1_src_pos
    diff1 = torch.where((diff1[:,0]<0).unsqueeze(-1).repeat([1,2]),-diff1,diff1)
    norm1 = torch.linalg.norm(diff1,dim=-1,keepdim=True)
    unit1 = diff1/norm1
    angles1 = torch.atan2(unit1[:,1],unit1[:,0])

    diff2 = e2_tgt_pos-e2_src_pos
    diff2 = torch.where((diff2[:,0]<0).unsqueeze(-1).repeat([1,2]),-diff2,diff2)
    norm2 = torch.linalg.norm(diff2,dim=-1,keepdim=True)
    unit2 = diff2/norm2
    angles2 = torch.atan2(unit2[:,1],unit2[:,0])

    angles = (angles1-angles2).abs()
    angles = torch.where(angles<=torch.pi/2,angles,torch.pi-angles)
    return 1 - ((torch.pi/2-angles)/(torch.pi/2)).abs().mean()


"""
Universal Quality Metrics for Graph Drawings:
Which Graphs Excite Us Most
"""
def angular_resolution(graph:Data):
    g:nx.Graph = to_networkx(Data(edge_index=graph.edge_index[:,graph.distance_matrix==1]))

    rs = []
    for n,adj in g.adjacency():
        if(len(adj))> 1:
            pos = graph.x[n].unsqueeze(0)
            pos_tgts = torch.stack([graph.x[tgt] for tgt in adj],dim=0)
            diff = pos_tgts-pos
            norm = torch.linalg.norm(diff,dim=-1,keepdim=True)
            unit = diff/norm
            angles = torch.atan2(unit[:,1],unit[:,0])
            angles = torch.where(angles<0,angles+torch.pi*2,angles)
            angles = torch.sort(angles)[0]
            min_angle = torch.pi
            for i in range(0,len(angles)):
                a  = angles[(i+1)] if i+1<len(angles) else (angles[0]+torch.pi*2)
                min_angle = min(min_angle,a-angles[i])
            rs.append(((torch.pi*2/len(adj))-min_angle)/(torch.pi*2/len(adj)))
    
    return 1-torch.tensor(rs).mean()
    
def stress(graph):
    edges,dm = remove_self_loops(graph.edge_index,graph.distance_matrix)
    distance = torch.linalg.norm(graph.x[edges[1]]-graph.x[edges[0]],dim=-1)
    return torch.mean((distance-dm)**2/(dm**2))

def get_perplexity(g):
        N = 128
        (Pmin, Pmax) = (2,N/2)
        (Nmin, Nmax) = (2, N)
        P = lambda n: (n-Nmin)*(Pmax-Pmin)/(Nmax-Nmin)+Pmin
        real_n = g.num_nodes
        perplexity = P(real_n)
        return torch.tensor(perplexity)

def kl(g):
  N = g.x.shape[0]
  sigma = tsnet_utils.find_sigma(g,N,get_perplexity(g),50)
  p = tsnet_utils.p_ij_conditional_var(g,sigma)
  sorted_by_dest = sort_edge_index(g.edge_index,p,sort_by_row=False)
  p2 = (p+sorted_by_dest[1])/(2*N)
  edge_index,p = remove_self_loops(g.edge_index,p2)
  Y = g.x
  diff: torch.Tensor = Y[edge_index[0]]-Y[edge_index[1]]
  dist = torch.linalg.norm(diff,dim=-1)
  q = ((1+dist**2)**-1)

  q = (q/q.sum())
  return (p*torch.log(torch.clamp(p,min=1e-20)/torch.clamp(q,min=1e-20))).sum().item()
