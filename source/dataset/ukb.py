import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_ukb_data(cfg: DictConfig):
    task = cfg.dataset.task
    roi_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    pearson_data = roi_data[:,1:].reshape(-1,180,180)
    subj_ids = roi_data[:,0]
    
    if task == 'sex':
        task_col_name = 'sex'
    elif task == 'age':
        task_col_name = 'age'
    elif task == 'int_fluid':
        task_col_name = 'fluid'   

    meta_data = pd.read_csv(cfg.dataset.label)[['eid',task_col_name]].dropna()
    subj_ids_with_label = [int(subj_id) in meta_data['eid'].values for subj_id in subj_ids]
    # print(subj_ids_with_label)
    #subj_ids[subj_ids_with_label]
    final_subj_ids = subj_ids[subj_ids_with_label]
    # print(final_subj_ids)
    #convert final_sujb_ids to numpy array with int type
    final_subj_ids = np.array(final_subj_ids).astype(int)
    pearson_data = np.nan_to_num(pearson_data.astype(float))
    final_pearson = pearson_data[subj_ids_with_label,:,:]

    
    if task == 'age':
        target_mean = 54.971
        target_std = 7.53
    elif task == 'int_fluid':
        target_mean = 6.655
        target_std = 2.01

    labels = []
    for subj_id in subj_ids[subj_ids_with_label]:
        if task == 'sex':
            labels.append(meta_data[meta_data['eid']==subj_id]['sex'].values[0])
        elif task == 'age' or task == 'int_fluid':
            target = meta_data[meta_data['eid']==subj_id][task_col_name].values[0]
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
