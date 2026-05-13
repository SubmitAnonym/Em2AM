import argparse
import os
import subprocess
import tempfile
import numpy as np
import torch
import tqdm
from data.utils import applyTlpLayoutAlgorithm, nx2tlp, tlpPos2np
from torch_geometric.utils import to_networkx,from_networkx,to_undirected,remove_self_loops
import torch_geometric.data
import networkx as nx
import resource
def do_normalize(pos:torch.Tensor,normalize,normalize_size):
    match normalize:
        case "constant":
            p = pos - pos.amin(dim=0,keepdim=True)
            size = p.amax(dim=0,keepdim=True)
            p = (p/size.amax())
            p*= float(normalize_size)
            return p
        case _:
            raise ValueError(f"unexpected normalisation {normalize}")
def produce_layout(graph,algo):
    undirected = torch_geometric.data.Data(edge_index=to_undirected(graph.edge_index),num_nodes=graph.num_nodes)
    nxg = to_networkx(undirected,to_undirected=True)
    tlpG,mapping = nx2tlp(nxg)
    assert all([i==j for i,j in mapping.items()])
    match algo:
        case 'PivotMDS':
            applyTlpLayoutAlgorithm(tlpG,'Random layout','viewLayout')
            applyTlpLayoutAlgorithm(tlpG,'Pivot MDS (OGDF)','viewLayout')
            return torch.from_numpy(tlpPos2np(tlpG)).float()
        case 'LinLog':
            applyTlpLayoutAlgorithm(tlpG,'Random layout','viewLayout')
            applyTlpLayoutAlgorithm(tlpG,'LinLog','viewLayout')
            return torch.from_numpy(tlpPos2np(tlpG)).float()
        case 'FM**3':
            applyTlpLayoutAlgorithm(tlpG,'Random layout','viewLayout')
            applyTlpLayoutAlgorithm(tlpG,'FM^3 (OGDF)','viewLayout')
            return torch.from_numpy(tlpPos2np(tlpG)).float()
        case 'GEM':
            applyTlpLayoutAlgorithm(tlpG,'Random layout','viewLayout')
            applyTlpLayoutAlgorithm(tlpG,'GEM Frick (OGDF)','viewLayout')
            return torch.from_numpy(tlpPos2np((tlpG))).float()
        case 'KK':
            applyTlpLayoutAlgorithm(tlpG,'Random layout','viewLayout')
            applyTlpLayoutAlgorithm(tlpG,'Kamada Kawai (OGDF)','viewLayout')
            return torch.from_numpy(tlpPos2np(tlpG)).float()
        case 'Random':
            applyTlpLayoutAlgorithm(tlpG,'Random layout','viewLayout')
            return torch.from_numpy(tlpPos2np(tlpG)).float()
        case 'sgd2':
            sgd2_python = "/data/shared/_envs/sgd2_numpy1/bin/python"
            tempfilein = tempfile.NamedTemporaryFile("w",suffix=".npy")
            tempfileout = tempfile.NamedTemporaryFile("r",suffix=".npy")
            np.save(tempfilein.name,graph.edge_index.int().numpy(),allow_pickle=False)
            process = subprocess.run([sgd2_python,os.path.join(os.path.dirname(os.path.realpath(__file__)),"produce_sgd2_pos.py"),"--input",tempfilein.name,"--output", tempfileout.name],capture_output=True,encoding="utf-8")
            if(process.returncode !=0):
                raise Exception(f"{process.stderr}\n{process.stdout}")
            else:
                pos = np.load(tempfileout.name)
                return torch.from_numpy(pos).float()
        case 'spring':
            nxg:nx.Graph = nxg
            pos = nx.spring_layout(nxg)
            pos = [pos[n] for n in nxg.nodes]
            pos = np.array(pos)
            return torch.from_numpy(pos).float()
        case _:
            raise ValueError(f"unexpected algorithm {algo}")
def layout_graph(g,normalize,algos,normalize_size):
    if not hasattr(g,'layouts') or g.layouts is None:
        g.layouts = ['init']
        p = do_normalize(g.x,normalize,normalize_size)
        g.x = p 
    not_complete_edge_index = g.edge_index[:,g.distance_matrix==1]
    graph = torch_geometric.data.Data(edge_index=not_complete_edge_index,num_nodes=g.num_nodes)
    for a in algos:
        pos = produce_layout(graph,a)
        pos = do_normalize(pos,normalize,normalize_size)
        g.x = torch.concatenate([g.x,pos],dim=-1)
        g.layouts.append(a)
    return g
def select_best_stress(g):
    pos = g.x.view(g.x.shape[0],-1,2)
    edges = g.edge_index[:,g.distance_matrix==1]
    avg_edge_length =  torch.linalg.norm(pos[edges[0]]-pos[edges[1]],dim=-1,keepdims=True).mean(dim=0,keepdims=True)
    pos = pos/avg_edge_length
    no_self_loop,dist = remove_self_loops(g.edge_index,g.distance_matrix)
    eucl_dist = torch.linalg.norm(pos[no_self_loop[0]]-pos[no_self_loop[1]],dim=-1)
    stress = torch.mean((eucl_dist-dist.unsqueeze(1))**2/(dist.unsqueeze(1)**2),dim=0)
    best_stress = torch.argmin(stress)
    g.best_stress = best_stress
    g.stress = stress
    g.layout_results = g.x
    g.x = g.x.reshape(g.x.shape[0],-1,2)[:,best_stress]
    print(f"selected {g.layouts[best_stress]} {stress} {g.x.shape} {g.layouts} {g.layout_results.shape}")

def layout_grids(input_dir,output_dir,normalize,algos,normalize_size):
    tsnet_dir = "/data/shared/graphvis-pytorch-geometric_data/grids_layouts_tsnet"
    dnn2_dir = "/data/shared/graphvis-pytorch-geometric_data/DNN2_gridApprox_layouts"
    for directory in os.listdir(input_dir):
        if not os.path.isdir(os.path.join(input_dir,directory)):
            continue
        output_subdir = os.path.join(output_dir,directory)
        os.makedirs(output_subdir,exist_ok=True)
        with os.scandir(os.path.join(input_dir,directory)) as it:
            for graph_file in it:
                layout_path = os.path.join(output_subdir,graph_file.name)
                if not os.path.exists(layout_path):
                    g = torch.load(graph_file.path,weights_only=False)
                    g = layout_graph(g,normalize,algos,normalize_size)
                    tsnet_path = os.path.join(tsnet_dir,directory,graph_file.name)
                    dnn2_path = os.path.join(dnn2_dir,directory,graph_file.name.replace(".pt",".npy"))
                    pos_tsnet = torch.load(tsnet_path,weights_only=False).tsnet
                    pos_tsnet = do_normalize(pos_tsnet,normalize,normalize_size)
                    pos_dnn2 = torch.tensor(np.load(dnn2_path),dtype=g.x.dtype)
                    pos_dnn2 = do_normalize(pos_dnn2,normalize,normalize_size)
                    g.layouts+=['DNN2','tsNET']
                    g.x = torch.concatenate((g.x,pos_dnn2,pos_tsnet),dim=-1)
                    select_best_stress(g)
                    torch.save(g,layout_path)


if __name__ == "__main__":
    max_mem = 10*(1024**3)
    resource.setrlimit(resource.RLIMIT_RSS,(max_mem,max_mem))
    parser  = argparse.ArgumentParser("layout_grids",description="generate grids layouts")
    parser.add_argument("--input-dir",required=True,help="target directory to load graph data from")
    parser.add_argument("--output-dir",required=True,help="target directory to store layout data to")
    parser.add_argument("--normalize",required=True,default='constant',help="normalisation to use on layouts",choices=['constant'])
    parser.add_argument("--normalize-size",required=True,default=1,help="constant size of normalisation used ",)
    parser.add_argument("--algos",required=True,nargs='+',default=['PivotMDS','LinLog','sgd2','spring','FM**3','GEM','KK','Random'],help="layouts algorithms ",choices=['PivotMDS','LinLog','sgd2','spring','FM**3','GEM','KK','Random'])
    args = parser.parse_args()
    layout_grids(args.input_dir,args.output_dir,args.normalize,args.algos,args.normalize_size)

