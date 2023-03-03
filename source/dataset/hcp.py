import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_hcp_data(cfg: DictConfig):

    roi_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    
    pearson_data = roi_data[:,1:].reshape(-1,180,180)
    subj_ids = roi_data[:,0].astype(int)

    task = cfg.dataset.task

    if task == 'sex':
        task_col_name = 'sex'
    elif task == 'age':
        task_col_name = 'age'
    elif task == 'int_total':
        task_col_name = 'CogTotalComp_AgeAdj'    

    if task == 'int_total':
        meta_data = pd.read_csv(cfg.dataset.label_int)[['subject',task_col_name]].dropna()
    else:
        meta_data = pd.read_csv(cfg.dataset.label)[['subject',task_col_name]].dropna()
    subj_ids_with_label = [subj_id in meta_data['subject'].values for subj_id in subj_ids]
    #subj_ids[subj_ids_with_label]
    final_subj_ids = subj_ids[subj_ids_with_label]
    pearson_data = np.nan_to_num(pearson_data.astype(float))
    final_pearson = pearson_data[subj_ids_with_label,:,:]
    
    if task == 'age':
        target_mean = 28.8
        target_std = 3.7
    elif task == 'int_total':
        target_mean = 112.198
        target_std = 20.906

    labels = []
    for subj_id in subj_ids[subj_ids_with_label]:
        if task == 'sex':
            sex = meta_data[meta_data['subject']==subj_id][task_col_name].values[0]
            sex = 1 if sex == 'M' else 0
            labels.append(sex)
        else:
            target = meta_data[meta_data['subject']==subj_id][task_col_name].values[0]
            labels.append((target - target_mean)/target_std)
    labels = np.array(labels)

    # dummy timeseries data, bnt does not use timeseries data   
    final_timeseries = np.zeros((final_pearson.shape[0],300))

    # final_pearson = torch.from_numpy(final_pearson.astype(float)).double()
    # labels = torch.from_numpy(labels)
    # final_timeseries = torch.zeros(final_pearson.shape[0],300)

    
    final_timeseries, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseries, final_pearson, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseries.shape[1]
    return final_subj_ids, final_timeseries, final_pearson, labels, task
