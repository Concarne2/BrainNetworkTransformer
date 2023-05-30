from .transformer import GraphTransformer
from .transformer_regression import GraphTransformerRegression
from omegaconf import DictConfig
from .brainnetcnn import BrainNetCNN
from .brainnetcnn_regression import BrainNetCNNRegression
from .fbnetgen import FBNETGEN
from .BNT.bnt import BrainNetworkTransformer
from .BNT.bnt_regression import BrainNetworkTransformerRegression


def model_factory(config: DictConfig):
    if config.model.name in ["LogisticRegression", "SVC"]:
        return None
    return eval(config.model.name)(config).cuda()
