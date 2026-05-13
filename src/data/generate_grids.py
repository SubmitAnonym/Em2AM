import os
import torch
from utils import make_grids
import argparse
import sys
from torch_geometric.utils import scatter
import numpy as np
import tqdm
def save_grids(train_dir,val_dir,test_dir,N_train,N_val,N_test,n_min,n_max,d_min,d_max,stats_dir,dtype=torch.float32):
    num_nodes = []
    num_edges = []
    distances = torch.zeros((n_max,),dtype=torch.int32)
    densities = []
    diameters = []
    for (dir,N_range) in zip([train_dir,val_dir,test_dir],[range(0,N_train),range(N_train,N_train+N_val),range(N_train+N_val,N_train+N_val+N_test)]):
        os.makedirs(dir)
        for i,data in enumerate(tqdm.tqdm(make_grids(N_range,n_min,n_max,d_min,d_max),dir,total=len(N_range)*(n_max-n_min+1))):
            num_nodes.append(data.num_nodes)
            data_distances = scatter(torch.ones(data.distance_matrix.shape,dtype=distances.dtype),data.distance_matrix.long(),dim_size=distances.shape[0])
            num_edges.append(data_distances[1].item())
            diameters.append(data.distance_matrix.amax().cpu().item())
            distances += data_distances
            densities.append(np.sqrt(data_distances[1].item()/(data.num_nodes*(data.num_nodes-1))))
            torch.save(data,os.path.join(dir,f"graph_{data.num_nodes}_{data_distances[1].item()}_{i}.pt"))

    num_nodes = torch.from_numpy(np.array(num_nodes)).long()
    num_edges = torch.from_numpy(np.array(num_edges)).long()
    diameters = torch.from_numpy(np.array(diameters)).long()
    np.save(os.path.join(stats_dir,"num_nodes_agg.npy"),scatter(torch.ones(num_nodes.shape,dtype=torch.int),num_nodes).numpy())
    np.save(os.path.join(stats_dir,"num_edges_agg.npy"),scatter(torch.ones(num_edges.shape,dtype=torch.int),num_edges).numpy())
    np.save(os.path.join(stats_dir,"diameters_agg.npy"),scatter(torch.ones(diameters.shape,dtype=torch.int),diameters).numpy())
    np.save(os.path.join(stats_dir,"distances.npy"),distances.numpy())
    np.save(os.path.join(stats_dir,"densities.npy"),np.array(densities))
    np.save(os.path.join(stats_dir,"num_nodes.npy"),num_nodes)
    np.save(os.path.join(stats_dir,"num_edges.npy"),num_edges)
    np.save(os.path.join(stats_dir,"diameters.npy"),diameters)

if __name__ == "__main__":
    parser  = argparse.ArgumentParser("make_grids",description="generate grids using tulip Grid Approximation and save to pytorch format")
    parser.add_argument("--train-dir",required=True,help="target directory to store train data")
    parser.add_argument("--val-dir",required=True,help="target directory to store val data")
    parser.add_argument("--test-dir",required=True,help="target directory to store test data")
    parser.add_argument("--n-min",required=True,default=20,type=int,help="minimum number of nodes")
    parser.add_argument("--n-max",required=True,default=200,type=int,help="maximum number of nodes")
    parser.add_argument("--N-train",required=True,type=int,default=80,help="number of training graphs per number of nodes")
    parser.add_argument("--N-val",required=True,type=int,default=10,help="number of val graphs per number of nodes")
    parser.add_argument("--N-test",required=True,type=int,default=10,help="number of test graphs per number of nodes")
    parser.add_argument("--d-min",required=True,type=int,default=2,help="minimum degree")
    parser.add_argument("--d-max",required=True,type=int,default=5,help="maximum degree")
    parser.add_argument("--stats-dir",required=True,help="target directory to store stats")
    args = parser.parse_args()
    save_grids(args.train_dir,args.val_dir,args.test_dir,args.N_train,args.N_val,args.N_test,args.n_min,args.n_max,args.d_min,args.d_max,args.stats_dir)

