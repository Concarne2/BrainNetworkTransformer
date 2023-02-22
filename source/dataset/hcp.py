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

    meta_data = pd.read_csv(cfg.dataset.label)[['subject',task]].dropna()
    subj_ids_with_label = [subj_id in meta_data['subjectkey'].values for subj_id in subj_ids]
    #subj_ids[subj_ids_with_label]
    pearson_data = np.nan_to_num(pearson_data.astype(float))
    final_pearson = pearson_data[subj_ids_with_label,:,:]

    labels = []
    for subj_id in subj_ids[subj_ids_with_label]:
        labels.append(meta_data[meta_data['subjectkey']==subj_id][task].values[0])
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
    return final_timeseries, final_pearson, labels, task
