from __future__ import absolute_import

from .TCLNet import TCLNet
#from .TCLNetExtended import TCLNetExtended
from .GRL_DomainClassifier import DANN_GRL_model
from .resnet import resnet50

__factory = {
        'TCLNet': TCLNet,
        #'TCLNetExtended': TCLNetExtended,
        'grl': DANN_GRL_model,
        'resnet50': resnet50
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
