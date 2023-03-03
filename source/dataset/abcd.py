import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abcd_data(cfg: DictConfig):
    task = cfg.dataset.task
    roi_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    pearson_data = roi_data[:,2:].reshape(-1,180,180)
    subj_ids = roi_data[:,1]
    
    if task == 'sex':
        task_col_name = 'sex'
    elif task == 'age':
        task_col_name = 'age'
    elif task == 'int_total':
        task_col_name = 'nihtbx_totalcomp_uncorrected'   

    meta_data = pd.read_csv(cfg.dataset.label)[['subjectkey',task_col_name]].dropna()
    subj_ids_with_label = [subj_id in meta_data['subjectkey'].values for subj_id in subj_ids]
    #subj_ids[subj_ids_with_label]
    final_subj_ids = subj_ids[subj_ids_with_label]
    pearson_data = np.nan_to_num(pearson_data.astype(float))
    final_pearson = pearson_data[subj_ids_with_label,:,:]

    
    if task == 'age':
        target_mean = 118.95
        target_std = 7.46
    elif task == 'int_total':
        target_mean = 0.0153
        target_std = 0.8664

    labels = []
    for subj_id in subj_ids[subj_ids_with_label]:
        if task == 'sex':
            labels.append(meta_data[meta_data['subjectkey']==subj_id]['sex'].values[0])
        elif task == 'age' or task == 'int_total':
            target = meta_data[meta_data['subjectkey']==subj_id][task_col_name].values[0]
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
