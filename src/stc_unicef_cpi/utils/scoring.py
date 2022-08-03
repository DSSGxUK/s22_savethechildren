import numpy as np


def mae(true, pred):
    """Calculate mean absolute error

    :param true: _description_
    :type true: _type_
    :param pred: _description_
    :type pred: _type_
    :return: _description_
    :rtype: _type_
    """
    return np.mean(np.abs(true - pred))
