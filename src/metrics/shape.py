from torch_geometric.data import Data,Batch
from torch_geometric.utils import to_networkx,from_networkx,scatter,coalesce,to_undirected
from torch_geometric.transforms import Delaunay

import networkx as nx
import torch
from scipy import spatial

"""
P. Eades, S.-H. Hong, K. Klein, and A. Nguyen, “Shape-based quality
metrics for large graph visualization,” in Graph Drawing and Network
Visualization: 23rd International Symposium, GD 2015, Los Angeles, CA,
USA, September 24-26, 2015, Revised Selected Papers. Springer, 2015,
pp. 502–514.
"""


def gabriel(graph,dt_edge_index):
  tree = spatial.KDTree(graph.x.numpy())
  middle = 0.5*(graph.x[dt_edge_index[0]]+graph.x[dt_edge_index[1]])
  lengths = torch.linalg.norm(graph.x[dt_edge_index[1]]-graph.x[dt_edge_index[0]],dim=-1)
  closest = torch.tensor(tree.query(x=middle.numpy(),k=1)[0])
  mask = (closest)>(lengths/2)*(1-1e-5)
  return dt_edge_index[:,mask]

def relative_neighbourhood_graph(graph,dt_edge_index):

  tree = spatial.KDTree(graph.x.numpy())
  srcs_pos = graph.x[dt_edge_index[0]]
  tgts_pos = graph.x[dt_edge_index[1]]

  lengths = torch.linalg.norm(graph.x[dt_edge_index[1]]-graph.x[dt_edge_index[0]],dim=-1)
  closest_src = tree.query_ball_point(x=srcs_pos.numpy(),r=lengths.numpy())
  closest_tgt = tree.query_ball_point(x=tgts_pos.numpy(),r=lengths.numpy())

  mask = torch.ones(lengths.shape,dtype=torch.bool)
  for i in range(len(lengths)):
    for j in closest_src[i]:
      if(j not in dt_edge_index[:,i]):
        if(j in closest_tgt[i]):
          mask[i]=False
          break
  return dt_edge_index[:,mask]
def emst(graph,dt_edge_index):
  lengths = torch.linalg.norm(graph.x[dt_edge_index[0]]-graph.x[dt_edge_index[1]],dim=-1)
  with_edge_attr = Data(edge_index=dt_edge_index,weight=lengths)
  with_edge_attr_nx = to_networkx(with_edge_attr,edge_attrs=["weight"],to_undirected=True)
  T = nx.minimum_spanning_tree(with_edge_attr_nx)
  t = from_networkx(T)
  return t.edge_index

def shape_metric(graph,shape_graph_edge_index):
  edges=graph.edge_index[:,graph.distance_matrix==1]
  num_nodes = graph.x.shape[0]
  return iou_edges(shape_graph_edge_index, edges, num_nodes)

def iou_edges(shape_graph_edge_index, edges, num_nodes):
    real_attr = torch.stack([torch.ones(edges.shape[1],dtype=torch.bool),
    torch.zeros(edges.shape[1],dtype=torch.bool)],dim=-1)
    shape_attr = torch.stack([torch.zeros(shape_graph_edge_index.shape[1],dtype=torch.bool),
    torch.ones(shape_graph_edge_index.shape[1],dtype=torch.bool)],dim=-1)
    coalesced = coalesce(torch.cat([edges,shape_graph_edge_index],dim=1),torch.cat([real_attr,shape_attr]),reduce='max',num_nodes=num_nodes)
    intersection = coalesced[1].all(dim=-1)
    union = coalesced[1].any(dim=-1)
    m2 = (scatter(intersection.int(),coalesced[0][1],reduce="sum")/scatter(union.int(),coalesced[0][1],reduce="sum")).sum()
    m1 = (scatter(intersection.int(),coalesced[0][0],reduce="sum")/scatter(union.int(),coalesced[0][0],reduce="sum")).sum()
    assert (m1==m2).all(),(m1,m2)
    return coalesced,m1
