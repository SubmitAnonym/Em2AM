import sys

import torch
import torch_geometric as tg
import model.neural_network as mynn
import model.neighbourhood_context as nneig

import data.utils as ut
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import json
import pandas as pd
import sklearn.metrics as skm
import seaborn as sns
import networkx as nx
import warnings

import os
import glob
from scipy.stats import kruskal
from scikit_posthocs import posthoc_nemenyi
from matplotlib.patches import Arc
from matplotlib import transforms
import re
from sklearn.metrics import auc
graphs_path = "/data/shared/graphvis-pytorch-geometric_data/grids_all/"
val_dataset = ut.load_dataset(path=f"{graphs_path}/val", device="cpu")
test_dataset = ut.load_dataset(path="/data/shared/graphvis-pytorch-geometric_data/grids_with_shape_graphs/test", device="cpu")
# print(test_dataset[0])
def add_significance_arcs(ax, pairwise_signif):
    N = pairwise_signif.shape[0]
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(N):
        for j in range(i+1, N, 1):
            if(pairwise_signif[i][j]):
                patch = Arc(((i + j)/2, -0.1), j - i, 0.15*np.sqrt(abs(j - i)), theta1=180.0, theta2=360.0, transform=trans, clip_on=False, edgecolor="black", linewidth=1)
                ax.add_patch(patch)
layouts = set()
color_palette = None
neigh_edge_indices = [
    ("_dt_edge_index","dt"),
    # ("_rng_edge_index","rng"),
    # ("_gabriel_edge_index","gabriel"),
    ]
eval_metric_names = [
    'f1_score','iou',
                     'bce']
for neigh_edge_index in neigh_edge_indices:
    for eval_metric_name in eval_metric_names:
        def func(x):
            y = x.sort_values("depth")
            return pd.DataFrame([{"auc":auc(y["depth"],y[eval_metric_name])}])
        def func2(x):
            y = x.sort_values("depth")
            diff = y[eval_metric_name].to_numpy()[1:]-y[eval_metric_name].to_numpy()[:-1]
            if(len(diff)>=5):
                print(x)
            return pd.concat([
                pd.DataFrame([{
                "x":"2-1",
                "diff":diff[0]}]),
                pd.DataFrame([{
                "x":"3-2",
                "diff":diff[1]}]),
                pd.DataFrame([{
                "x":"4-3",
                "diff":diff[2]}]),pd.DataFrame([{
                "x":"5-4",
                "diff":diff[3]}]),
                pd.DataFrame([{
                "x":"6-5",
                "diff":diff[4]}]),
                ])

        files = sorted(glob.glob(f"eval_nogit_bis*{neigh_edge_index[0]}_metrics.csv"))
        runs= {}
        # print(f"eval_nogit_bis_([^_.]+)(_run_2)?{neigh_edge_index[0]}_metrics.csv")
        for f in files:
            df = pd.read_csv(f,sep=";")
            m = re.match(f"eval_nogit_bis_([^_.]+)(_run_2)?{neigh_edge_index[0]}_metrics.csv",f)
            if m is None:
                # print(f)
                continue
            if m.group(2) is not None :
                layout = m.group(1)
                # print(f,layout,2)
                run = 2
                # layouts_run_2[layout]=df
            else:
                layout = m.group(1)
                # print(f,layout,1)
                run = 1
                # layouts[layout]=df
            if run not in runs:
                runs[run]={"layouts":{},"thrs":{}}
            layouts.add(layout)
            runs[run]['layouts'][layout]=df
        if color_palette is None:
            p = sns.color_palette(n_colors=len(layouts))
            color_palette = {i:p[j] for j,i in enumerate(layouts)}

        for run in runs:
            for layout,df in runs[run]['layouts'].items():
                runs[run]['thrs'][layout]={}
                for model_name in df["model_name"].unique():
                    depth = int(model_name.split("_")[2])
                    thresholds = df["threshold"].unique()
                    metric_values = [df.query(f"model_name == '{model_name}' and threshold == {thr} and dataset == 'val'")[eval_metric_name].mean() for thr in thresholds]
                    best = np.argmax(metric_values)
                    runs[run]['thrs'][layout][depth]={"thr":thresholds[best],f"{eval_metric_name}":metric_values[best]}

            color_scale = mpl.color_sequences["Set3"]
            for i,(layout,depth) in enumerate(runs[run]['thrs'].items()):
                plt.gca().plot(depth.keys(),[i[f"{eval_metric_name}"] for i in depth.values()], label=layout,color=color_palette[layout])
            plt.gca().set_xlabel("depth")
            plt.gca().set_ylabel(f"mean {eval_metric_name}")
            plt.gca().set_title(f"{eval_metric_name} run {run} {neigh_edge_index[1]}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{neigh_edge_index[1]}_{eval_metric_name}_run_{run}_model_perf.png")
            plt.clf()
            # plt.show()
        with open(f"thresholds_{eval_metric_name}.json","wt") as d:
            json.dump({run:runs[run]['thrs'] for run in runs},d)
        aucs = {}
        for run in runs:
            dfs = []
            for layout,df in runs[run]['layouts'].items():
                for model_name in df["model_name"].unique():
                    depth = int(model_name.split("_")[2])   
                    
                    thr = runs[run]['thrs'][layout][depth]["thr"]
                    model_df = df.query(f"model_name == '{model_name}' and threshold == {thr} and dataset == 'test'") 
                    model_df.insert(2,"depth",depth)
                    model_df.insert(3,"layout",layout)
                    dfs.append(model_df[[eval_metric_name,"graph","depth","layout"]])
            print("unique ",run,pd.concat(dfs)["depth"].unique())
            tmp = pd.concat(dfs).groupby(["graph","layout"]).apply(func2,include_groups=False).reset_index()[["graph","layout","diff","x"]]
            tmp2 = tmp.groupby(["graph","layout"]).apply(lambda x : pd.DataFrame([{"monotonous":np.all((x["diff"].to_numpy()<=0.0)==(eval_metric_name=="bce"))}]),include_groups=False).reset_index()[["graph","layout","monotonous"]]
            ax = sns.countplot(tmp2,x="layout",hue="monotonous")
            ax.set_title(f"run {run} {eval_metric_name} {neigh_edge_index[1]}")
            ax.tick_params("x",rotation=90)
            plt.tight_layout()
            plt.savefig(f"{neigh_edge_index[1]}_{eval_metric_name}_run_{run}_monotonous_count.png")
            plt.clf()
            # plt.show()
            plt.clf()
            ax = sns.boxplot(tmp,x="layout",hue="x",y="diff")
            ax.tick_params("x",rotation=90)
            ax.set_title(f"run {run} {eval_metric_name} {neigh_edge_index[1]}")
            plt.tight_layout()
            plt.savefig(f"{neigh_edge_index[1]}_{eval_metric_name}_run_{run}_monotonous_diff.png")
            # plt.show()
            plt.clf()



            aucs_run = (pd.concat(dfs).groupby(["graph","layout"]).apply(func,include_groups=False).reset_index()[["graph","layout","auc"]])
            aucs[run]=aucs_run
        # for run_1 in aucs:
        #     for run_2 in aucs:
        #         if run_1!=run_2:
        #             for i in aucs[run_1]["layout"].unique():
        #                 run_1_res = aucs[run_1].query(f"layout == '{i}'")
        #                 run_2_res = aucs[run_2].query(f"layout == '{i}'")
                        # print(i,run_1,run_2,kruskal(run_1_res["auc"],run_2_res["auc"]))
        values = iter(aucs.values())
        merge = next(values)
        w=1
        for other in values:

            merge = pd.merge(merge,other,on=["graph","layout"])
            merge["auc"]=(merge["auc_x"]*w+merge["auc_y"])/(w+1)
            w = w +1
            merge = merge[["graph","layout","auc"]]
        aucs = merge
        aucs.to_csv(f"aucs_{neigh_edge_index[1]}_{eval_metric_name}.csv",index=False, sep=";")
        metrics = pd.DataFrame([{"graph":i,"stress":graph.stress[graph.layouts.index(j)].item(),
                                "kl":graph.kls[graph.layouts.index(j)],
                                # "min_crossing_angle":graph.min_crossing_angles[graph.layouts.index(j)].item(),
                                # "edge_length_uniformity":graph.edge_length_uniformities[graph.layouts.index(j)].item(),
                                "edge_crossings":graph.edge_crossings[graph.layouts.index(j)].item(),
                                "angular_resolution":graph.angular_resolutions[graph.layouts.index(j)].item(),
                                "neighbourhood_preservations":graph.neighbourhood_preservations[graph.layouts.index(j)].item(),
                                "node_resolutions":graph.node_resolutions[graph.layouts.index(j)].item(),
                                "node_uniformities":graph.node_uniformities[graph.layouts.index(j)].item(),
                                "aspect_ratios":graph.aspect_ratios[graph.layouts.index(j)].item(),
                                "edge_length_deviations_Purchase":graph.edge_length_deviations_Purchase[graph.layouts.index(j)].item(),
                                "edge_length_deviations_Haleem":graph.edge_length_deviations_Haleem[graph.layouts.index(j)].item(),
                                "crossing_angles":graph.crossing_angles[graph.layouts.index(j)].item(),
                                "shape_gabriel":graph.shape_gabriels[graph.layouts.index(j)].item()/graph.x.shape[0],
                                "shape_dt":graph.shape_dts[graph.layouts.index(j)].item()/graph.x.shape[0],
                                "shape_emst":graph.shape_emsts[graph.layouts.index(j)].item()/graph.x.shape[0],
                                "shape_rngs":graph.shape_rngs[graph.layouts.index(j)].item()/graph.x.shape[0],
                                "layout":j} for i,graph in enumerate(test_dataset) for j in layouts])




        bylayout = aucs.groupby("graph").apply(lambda x:x.pivot_table(columns=["layout"],values='auc'),include_groups=False)
        index_auc = list(bylayout.median().sort_values(ascending=True).index)
        metrics_to_plot = [
            {'df':aucs,'metric':"auc",'log_scale':False,'hib':eval_metric_name!="bce"},
            {'df':metrics,'metric':"stress",'log_scale':True,'hib':False},
            {'df':metrics,'metric':"kl",'log_scale':False,'hib':False},
            # {'df':metrics,'metric':"min_crossing_angle",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"angular_resolution",'log_scale':False,'hib':True},
            # {'df':metrics,'metric':"edge_length_uniformity",'log_scale':True,'hib':False},
            {'df':metrics,'metric':"edge_crossings",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"neighbourhood_preservations",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"node_resolutions",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"node_uniformities",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"aspect_ratios",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"edge_length_deviations_Purchase",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"edge_length_deviations_Haleem",'log_scale':True,'hib':False},
            {'df':metrics,'metric':"crossing_angles",'log_scale':True,'hib':True},
            {'df':metrics,'metric':"shape_gabriel",'log_scale':False,'hib':True},
            {'df':metrics,'metric':"shape_dt",'log_scale':False,'hib':True},
            {'df':metrics,'metric':"shape_emst",'log_scale':False,'hib':True},
            {'df':metrics,'metric':"shape_rngs",'log_scale':False,'hib':True}
                                       ]
        # for (df,metric,log_scale,hib) in [(aucs,"auc",False,True),(metrics,"stress",True,False),(metrics,"kl",False,False),(metrics,"min_crossing_angle",True,True),(metrics,"angular_resolution",False,True),(metrics,"edge_length_uniformity",True,False),(metrics,"edge_crossings",True,False),(metrics,"shape_gabriel",False,True),(metrics,"shape_dt",False,True),(metrics,"shape_emst",False,True),(metrics,"shape_rngs",False,True)]:
        for metric_to_plot in metrics_to_plot:
            if(metric_to_plot['metric']=="auc" or (eval_metric_name==eval_metric_names[0] and neigh_edge_index[0]==neigh_edge_indices[0][0])):
                df = metric_to_plot['df']
                metric = metric_to_plot['metric']
                log_scale = metric_to_plot['log_scale']
                hib = metric_to_plot['hib']

                bylayout = df.groupby("graph").apply(lambda x:x.pivot_table(columns=["layout"],values=metric),include_groups=False)
                index = bylayout.median().sort_values(ascending=hib).index
                bylayout = bylayout[index]
                # color_palette =  np.array(sns.color_palette(n_colors=len(index)))[np.array([index_auc.index(i) for i in index])] if color_palette is None else color_palette
                if(metric=="auc"):
                    image_name = f"{neigh_edge_index[1]}_{eval_metric_name}_{metric}_1.png"
                else:
                    image_name = f"{metric}_1.png"
                print(bylayout,color_palette,image_name)
                ax = sns.boxplot(bylayout,palette = color_palette)
                if(log_scale):
                    ax.set_yscale("log")
                ax.tick_params("x",rotation=90)
                ax.set_ylabel(f"{metric} {eval_metric_name}" if metric=="auc"  else metric)
                # kruskal()
                p = kruskal(*bylayout.to_numpy().T)
                # print(p)
                if(p.pvalue<1e-4):
                    stat = posthoc_nemenyi(df,metric,"layout")
                    # print(stat)
                    add_significance_arcs(ax,(stat.loc[list(index)][index]<1e-4).to_numpy())
                ax.set_title(f"{eval_metric_name} {neigh_edge_index[1]}" if metric_to_plot['metric']=="auc" else f"{metric}")
                plt.tight_layout()
                plt.savefig(image_name)
                # plt.show()
                plt.clf()
                ax = sns.boxplot(bylayout,palette = color_palette)
                ax.tick_params("x",rotation=90)
                ax.set_ylabel(f"{metric} {eval_metric_name}" if metric=="auc"  else metric)
                if(log_scale):
                    ax.set_yscale("log")
                if(p.pvalue<1e-4):
                    add_significance_arcs(ax,(stat.loc[list(index)][index]>=1e-4).to_numpy())
                ax.set_title(f"{eval_metric_name} {neigh_edge_index[1]}" if metric_to_plot['metric']=="auc" else f"{metric}")
                plt.tight_layout()
                if(metric=="auc"):
                    plt.savefig(f"{neigh_edge_index[1]}_{eval_metric_name}_{metric}_2.png")
                else:
                    plt.savefig(f"{metric}_2.png")
                # plt.show()         
                plt.clf()
        columns = [i['metric'] for i in metrics_to_plot ]

        for m in ["pearson","spearman"]:
            fig,ax = plt.subplots(1,1,figsize=(10,10))
            ax = sns.boxplot(aucs.merge(metrics).groupby("graph").apply(lambda x:x.sort_values("layout")[columns].corr(m)["auc"].loc[columns],include_groups=False),ax=ax)
            ax.tick_params("x",rotation=90)
            ax.set_ylabel(f"{m} correlation with model {eval_metric_name} auc")
            ax.set_xlabel("metric")
            ax.set_title(f"{eval_metric_name} {neigh_edge_index[1]}")
            fig.tight_layout()
            fig.savefig(f"{neigh_edge_index[1]}_{eval_metric_name}_{m}.png")
            fig.clear()
            plt.clf()




