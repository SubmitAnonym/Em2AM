import torch
import torch_geometric as tg
import model.neural_network as mynn
import model.neighbourhood_context as nneig

import data.utils as ut
import numpy as np
from matplotlib import pyplot as plt
import json
import pandas as pd
import sklearn.metrics as skm
import seaborn as sns
import networkx as nx
import warnings
from torch.utils.flop_counter import FlopCounterMode
import os
from scipy.stats import kruskal
from scipy.stats.mstats import kruskalwallis
from scikit_posthocs import posthoc_nemenyi
from metrics.losses import BatchedEdgeList_CrossEntropy2
from torch.nn import L1Loss,MSELoss
from metrics.shape import iou_edges
def all_to(items, device):
    for i in items:
        i.to(device)

device = "cuda"
np.random.seed(0)
torch.manual_seed(0)
def select_model(train_dir):
    history = {}
    with open(f"{train_dir}/history.json", "r") as fd:
        history = json.load(fd)
        val_loss = history["val_loss"]
        best_epoch = np.argmin(val_loss)
    weights_path = f"{train_dir}/epoch_{best_epoch}.pth"
    print("Best epoch:", best_epoch, "with val loss:", val_loss[best_epoch])
    return weights_path
def loadable(train_dir,n_epochs,patience=20):
    if(os.path.exists(os.path.join(train_dir,"history.json"))):
        with open(os.path.join(train_dir,"history.json"),"rt") as f:
            history = json.load(f)
            epoch = np.argmin(history['val_loss'])
            if(len(history['val_loss'])==n_epochs):
                return epoch
            if epoch+patience<len(history['val_loss']):
                return epoch
            return None
    else:
        return None
def get_model(base, num_layers, width_factor):
    model = nneig.IncrementallyGrowing(layers=[base.enc_layers[0].out_features*(2**j)*width_factor for j in range(num_layers)])
    return model

def TPFPFNTN(predMatrix,gtMatrix):
    
    tp = (predMatrix * gtMatrix).sum()
    fp = (predMatrix * (1 - gtMatrix)).sum()
    tn = ((1 - predMatrix) * (1 - gtMatrix)).sum() 
    fn = ((1 - predMatrix) * gtMatrix).sum()
    
    return tp, fp, fn, tn

def F_score(tp, fp, fn, tn):
    return (2*tp) / (2*tp + fp + fn)

def accuracy(tp, fp, fn, tn):
    return (tp+tn) / (tp + fp + fn + tn)

def precision(tp, fp, fn, tn):
    return tp / (tp + fp)

def recall(tp, fp, fn, tn):
    return tp / (tp + fn)

def mcc(tp, fp, fn, tn):
    return (tp*tn - fp*fn) / np.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn))

def IoU(tp,fp,fn,tn):
    return tp/(tp+fp + fn)

for run in ["","_run_2"]:
# for run in ["_run_2"]:
    shape_graph = ("rng_edge_index","rng_layout_id")
    shape_graph = ("gabriel_edge_index","gabriel_layout_id")
    shape_graph = ("dt_edge_index","dt_layout_id")
    for layout_algo in ['GEM','spring','sgd2','LinLog','init','PivotMDS','FM**3', 'KK', 'Random', 'DNN2','tsNET']:
        batch_size = 16
        # graphs_path = "/data/shared/graphvis-pytorch-geometric_data/grids_all/"
        graphs_path = "/data/shared/graphvis-pytorch-geometric_data/grids_with_shape_graphs/"
        val_dataset = ut.load_dataset(path=f"{graphs_path}/val", device=device)
        test_dataset = ut.load_dataset(path=f"{graphs_path}/test", device=device)
        val_AMs = []
        test_AMs = []
        for d in val_dataset:
            d.N = d.x.shape[0]
            d.x=d.layout_results.view(-1,len(d.layouts),2)[:,d.layouts.index(layout_algo)]
            shape_edge_index = getattr(d,shape_graph[0])
            shape_edge_index_layout_id = getattr(d,shape_graph[1])
            edge_index = shape_edge_index[:,shape_edge_index_layout_id==d.layouts.index(layout_algo)]
            d["neighbourhood_index"] = edge_index
            AM = (d.distance_matrix.reshape((d.N, d.N)) == 1).numpy().astype(int)
            val_AMs.append(AM)
        for d in test_dataset:
            d.N = d.x.shape[0]
            d.x=d.layout_results.view(-1,len(d.layouts),2)[:,d.layouts.index(layout_algo)]
            shape_edge_index = getattr(d,shape_graph[0])
            shape_edge_index_layout_id = getattr(d,shape_graph[1])
            edge_index = shape_edge_index[:,shape_edge_index_layout_id==d.layouts.index(layout_algo)]
            d["neighbourhood_index"] = edge_index
            AM = (d.distance_matrix.reshape((d.N, d.N)) == 1).numpy().astype(int)
            test_AMs.append(AM)
        val_loader = tg.loader.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = tg.loader.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        models = []
        names= []
        base = nneig.IncrementallyGrowing()
        n_epochs = 400


        widths = (2**np.arange(5)).tolist()
        widths = [4]
        for num_layers in range(1,7):
            for width_factor in widths:
                model = get_model(base, num_layers, width_factor)
                train_dir = f"./train_dir/AM_predictor_neighbourhood_{layout_algo}_growing_bis{run}_{num_layers}_{width_factor}_{shape_graph[0]}"
                epoch = loadable(train_dir,n_epochs)
                if(epoch is not None):
                    model.load_state_dict(torch.load(os.path.join(train_dir,f"epoch_{epoch}.pth")))
                    model.eval()
                    model.to(device)
                    model.compile()
                    models.append(model)
                    names.append(f"{layout_algo}_depth_{num_layers}_width_{width_factor}{run}_{shape_graph[0]}")


        model_preds = {l:[] for l in names}
        model_preds_test = {l:[] for l in names}

        device = "cuda"

        # all_to([dnn], device) 
        model_preds_metrics_val = {l:[] for l in names}
        model_preds_metrics_test = {l:[] for l in names}
        mse = MSELoss()
        mae = L1Loss()
        bce = BatchedEdgeList_CrossEntropy2(reduction='none')
        with torch.no_grad():
            for i_batch, batch in enumerate(val_loader):
                for i_layout, name in enumerate(names):
                    layout_batch = batch.to(device)
                    dnn = models[i_layout]
                    # layout_batch = batch.clone()
                    # layout_batch.x = layout_batch.x[:, i_layout*2:(i_layout+1)*2]
                    # layout_batch.to(device)
                    pred = dnn(layout_batch)
                    bce_loss = bce(pred,batch)

                    end_of_last_graph = 0
                    for b in range(layout_batch.num_graphs):
                        N = layout_batch.N[b]
                        Nsquare = N**2
                        mae_loss = mae(pred[batch.batch[batch.edge_index[0]]==b],(batch.distance_matrix[batch.batch[batch.edge_index[0]]==b]==1).float())
                        mse_loss = mse(pred[batch.batch[batch.edge_index[0]]==b],(batch.distance_matrix[batch.batch[batch.edge_index[0]]==b]==1).float())
                        model_preds_metrics_val[name].append({"bce":bce_loss[b].item(),"mae":mae_loss.item(),"mse":mse_loss.item()})
                        pred_mat = pred[end_of_last_graph:end_of_last_graph+Nsquare].cpu().numpy().reshape((N, N))
                        pred_mat = pred_mat * (1 - np.eye(N))
                        model_preds[name].append(pred_mat)
                        end_of_last_graph += Nsquare
            for i_batch, batch in enumerate(test_loader):
                for i_layout, name in enumerate(names):
                    layout_batch = batch.to(device)
                    dnn = models[i_layout]
                    # layout_batch = batch.clone()
                    # layout_batch.x = layout_batch.x[:, i_layout*2:(i_layout+1)*2]
                    # layout_batch.to(device)
                    pred = dnn(layout_batch)
                    bce_loss = bce(pred,batch)
                    end_of_last_graph = 0
                    for b in range(layout_batch.num_graphs):
                        N = layout_batch.N[b]
                        Nsquare = N**2
                        mae_loss = mae(pred[batch.batch[batch.edge_index[0]]==b],(batch.distance_matrix[batch.batch[batch.edge_index[0]]==b]==1).float())
                        mse_loss = mse(pred[batch.batch[batch.edge_index[0]]==b],(batch.distance_matrix[batch.batch[batch.edge_index[0]]==b]==1).float())
                        model_preds_metrics_test[name].append({"bce":bce_loss[b].item(),"mae":mae_loss.item(),"mse":mse_loss.item()})
                        pred_mat = pred[end_of_last_graph:end_of_last_graph+Nsquare].cpu().numpy().reshape((N, N))
                        pred_mat = pred_mat * (1 - np.eye(N))
                        model_preds_test[name].append(pred_mat)
                        end_of_last_graph += Nsquare
        np.save(f"pred_val_{layout_algo}_run_{run}.npy",model_preds)
        np.save(f"pred_test_{layout_algo}_run_{run}.npy",model_preds_test)
        df = pd.DataFrame(columns=['graph', 'model_name', 'threshold', 'accuracy', 'f1_score', 'precision', 'recall', "mcc", "dataset","mse","mae","bce","iou","iou_with_dt"])
        thresholds = np.arange(0.5, 0.96, 0.05)
        cpt = 0
        print(model_preds.keys())
        for i, (name, preds) in enumerate(model_preds.items()):
            print(name, "... ", end="")
            for thr in thresholds:
                thr = np.round(thr, 2)
                print(thr, " .. ", end=" ")
                for j, pred in enumerate(preds):
                    gt = val_AMs[j].flatten()
                    p = (pred.flatten() > thr).astype(int)
                    tp, fp, fn, tn = TPFPFNTN(p, gt)
                    acc = accuracy(tp, fp, fn, tn)
                    f1 = F_score(tp, fp, fn, tn)
                    rec = recall(tp, fp, fn, tn)
                    prec = precision(tp, fp, fn, tn)
                    mcc_val = mcc(tp, fp, fn, tn)
                    iou_val = IoU(tp, fp, fn, tn)
                    shape_metric_value = iou_edges(val_dataset[j].dt_edge_index,val_dataset[j].edge_index[:,p>0],val_dataset[j].x.shape[0])[1].item()
                    df.loc[cpt] = [j, name, thr, acc, f1, prec, rec, mcc_val,"val",model_preds_metrics_val[name][j]["mse"],model_preds_metrics_val[name][j]["mae"],model_preds_metrics_val[name][j]["bce"],iou_val,shape_metric_value]
                    cpt+= 1
        for i, (name, preds) in enumerate(model_preds_test.items()):
            for thr in thresholds:
                thr = np.round(thr, 2)
                print(thr, " .. ", end=" ")
                for j, pred in enumerate(preds):
                    gt = test_AMs[j].flatten()
                    p = (pred.flatten() > thr).astype(int)
                    tp, fp, fn, tn = TPFPFNTN(p, gt)
                    acc = accuracy(tp, fp, fn, tn)
                    f1 = F_score(tp, fp, fn, tn)
                    rec = recall(tp, fp, fn, tn)
                    prec = precision(tp, fp, fn, tn)
                    mcc_val = mcc(tp, fp, fn, tn)
                    iou_val = IoU(tp, fp, fn, tn)
                    shape_metric_value = iou_edges(test_dataset[j].dt_edge_index,test_dataset[j].edge_index[:,p>0],test_dataset[j].x.shape[0])[1].item()
                    df.loc[cpt] = [j, name, thr, acc, f1, prec, rec, mcc_val,"test",model_preds_metrics_test[name][j]["mse"],model_preds_metrics_test[name][j]["mae"],model_preds_metrics_test[name][j]["bce"],iou_val,shape_metric_value]
                    cpt+= 1
            print()
        df.to_csv(f"eval_nogit_bis_{layout_algo}{run}_{shape_graph[0]}_metrics.csv", index=False, sep=";")
