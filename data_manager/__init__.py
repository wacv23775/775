from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from data_manager.video.mars import Mars
from data_manager.video.duke import DukeMTMCVidReID
from data_manager.video.ilidsvid import iLIDSVID
from data_manager.video.prid import PRID2011

from data_manager.image.market1501 import Market1501
from data_manager.image.msmt17 import MSMT17
from data_manager.image.ilids import iLIDS
from data_manager.image.prid import PRID
from data_manager.image.cuhk03 import CUHK03


__vidreid_factory = {
    'mars': Mars,
    'duke': DukeMTMCVidReID,
    'ilids-vid': iLIDSVID,
    'prid2011': PRID2011
}

__imgreid_factory = {
    'market1501': Market1501,
    'cuhk03': CUHK03,
    'msmt17': MSMT17,
    'ilids': iLIDS,
    'prid': PRID,
}


def get_names():
    return list(__vidreid_factory.keys() | __imgreid_factory.keys())

def init_dataset(name, **kwargs):
    """if name not in list(__vidreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__vidreid_factory.keys())))
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
"""
    if (name not in list(__vidreid_factory.keys()) and name not in list(__imgreid_factory.keys())):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(
            name, list(__imgreid_factory.keys()) + list(__vidreid_factory.keys())
                                                                                        )
                        )

    if name in list(__vidreid_factory.keys()):
        return __vidreid_factory[name](**kwargs)
    if name in list(__imgreid_factory.keys()):
        return __imgreid_factory[name](**kwargs)
