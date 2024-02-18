import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, ttest_1samp
import scipy.special as special


def ppcor_python(C, args):
    """
    Python implementation of ppcor.
    This function calculates the sample linear partial correlation coefficients between pairs of variables in C,
    controlling for the remaining variables in C.

    :param C: array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable

    :return: array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    C = np.reshape(C, (-1, args.b_manual_nodes))
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)  # sample linear partial correlation coefficients

    corr = np.corrcoef(C, rowvar=False)  # Pearson product-moment correlation coefficients.
    corr_inv = np.linalg.inv(corr)  # the (multiplicative) inverse of a matrix.

    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            pcorr_ij = -corr_inv[i, j] / (np.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            # idx = np.ones(p, dtype=np.bool)
            # idx[i] = False
            # idx[j] = False
            # beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            # beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            # res_j = C[:, j] - C[:, idx].dot( beta_i)
            # res_i = C[:, i] - C[:, idx].dot(beta_j)

            # corr = stats.pearsonr(res_i, res_j)[0]
            # P_corr[i, j] = corr
            # P_corr[j, i] = corr
            P_corr[i, j] = pcorr_ij
            P_corr[j, i] = pcorr_ij

    return P_corr

def ppcor_python_p_values(C, args):
    """
    Python implementation of ppcor.
    This function calculates the sample linear partial correlation coefficients between pairs of variables in C,
    controlling for the remaining variables in C.

    :param C: array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable

    :return: array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    C = np.reshape(C, (-1, args.b_manual_nodes))
    p = C.shape[1]
    length = C.shape[0]
    P_corr = np.zeros((p, p), dtype=np.float)  # sample linear partial correlation coefficients

    corr = np.corrcoef(C, rowvar=False)  # Pearson product-moment correlation coefficients.
    corr_inv = np.linalg.inv(corr)  # the (multiplicative) inverse of a matrix.

    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            pcorr_ij = -corr_inv[i, j] / (np.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            # idx = np.ones(p, dtype=np.bool)
            # idx[i] = False
            # idx[j] = False
            # beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            # beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            # res_j = C[:, j] - C[:, idx].dot( beta_i)
            # res_i = C[:, i] - C[:, idx].dot(beta_j)

            # corr = stats.pearsonr(res_i, res_j)[0]
            # P_corr[i, j] = corr
            # P_corr[j, i] = corr
            P_corr[i, j] = pcorr_ij
            P_corr[j, i] = pcorr_ij

    dof = length - 2
    with np.errstate(divide='ignore'):
        # clip the small negative values possibly caused by rounding
        # errors before taking the square root
        t = P_corr * np.sqrt((dof/((P_corr+1.0)*(1.0-P_corr))).clip(0))
    t, prob = _ttest_finish(dof, t)
    return prob


def ppcor_python_raw(C):
    """
    Python implementation of ppcor.
    This function calculates the sample linear partial correlation coefficients between pairs of variables in C,
    controlling for the remaining variables in C.

    :param C: array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable

    :return: array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    # C = np.reshape(C, (-1, args.b_manual_nodes))
    p = C.shape[1]
    length = C.shape[0]
    P_corr = np.zeros((p, p), dtype=np.float)  # sample linear partial correlation coefficients

    corr = np.corrcoef(C, rowvar=False)  # Pearson product-moment correlation coefficients.
    corr_inv = np.linalg.inv(corr)  # the (multiplicative) inverse of a matrix.

    for i in range(p):
        P_corr[i, i] = 1
        for j in range(i + 1, p):
            pcorr_ij = -corr_inv[i, j] / (np.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            # idx = np.ones(p, dtype=np.bool)
            # idx[i] = False
            # idx[j] = False
            # beta_i = linalg.lstsq(C[:, idx], C[:, j])[0]
            # beta_j = linalg.lstsq(C[:, idx], C[:, i])[0]

            # res_j = C[:, j] - C[:, idx].dot( beta_i)
            # res_i = C[:, i] - C[:, idx].dot(beta_j)

            # corr = stats.pearsonr(res_i, res_j)[0]
            # P_corr[i, j] = corr
            # P_corr[j, i] = corr
            P_corr[i, j] = pcorr_ij
            P_corr[j, i] = pcorr_ij
    dof = length - 2
    with np.errstate(divide='ignore'):
        # clip the small negative values possibly caused by rounding
        # errors before taking the square root
        t = P_corr * np.sqrt((dof/((P_corr+1.0)*(1.0-P_corr))).clip(0))
    t, prob = _ttest_finish(dof, t)
    return prob

def _ttest_finish(df, t, alternative='two-sided'):
    """Common code between all 3 t-test functions."""
    # We use ``stdtr`` directly here as it handles the case when ``nan``
    # values are present in the data and masked arrays are passed
    # while ``t.cdf`` emits runtime warnings. This way ``_ttest_finish``
    # can be shared between the ``stats`` and ``mstats`` versions.

    if alternative == 'less':
        pval = special.stdtr(df, t)
    elif alternative == 'greater':
        pval = special.stdtr(df, -t)
    elif alternative == 'two-sided':
        pval = special.stdtr(df, -np.abs(t))*2
    else:
        raise ValueError("alternative must be "
                         "'less', 'greater' or 'two-sided'")

    if t.ndim == 0:
        t = t[()]
    if pval.ndim == 0:
        pval = pval[()]

    return t, pval


def ppcor_python_spearman(C):
    """
    Python implementation of ppcor.
    This function calculates the sample linear partial correlation coefficients between pairs of variables in C,
    controlling for the remaining variables in C.

    :param C: array-like, shape (n, p)
        Array with the different variables. Each column of C is taken as a variable

    :return: array-like, shape (p, p)
        P[i, j] contains the partial correlation of C[:, i] and C[:, j] controlling
        for the remaining variables in C.
    """
    C = np.asarray(C)
    # C = np.reshape(C, (-1, args.b_manual_nodes))
    p = C.shape[1]
    P_corr = np.zeros((p, p), dtype=np.float)  # sample linear partial correlation coefficients

    P_corr = spearmanr(C, axis=0)[1]

    return P_corr

def reframe_latent(z, args):
    """
    Align each node latent features in every time step with other nodes latent features from previous time steps.
    The return list is a list of z, each z is an array of shape (timesteps - lag_time - 1, b_manual_nodes)
    :param z:
    :param args:
    :return: a list of z, each z is an array of shape (timesteps - lag_time - 1, b_manual_nodes)
    """
    return_ls = []
    # print("Working on reframe latent features for directions in the graph.")
    for node in range(args.b_manual_nodes):
        node_z = z[:, node]
        node_z = node_z[1:]
        z_shorten = z[:-1, :]
        z_shorten[:, node] = node_z
        return_ls.append(np.reshape(z_shorten, (-1, args.b_manual_nodes)))
    return return_ls


def grab_the_adjacency_matrix(graph_cache, args):
    """
    Grab the adjacency matrix from the graph_cache.
    :param graph_cache: a list of adjacency matrices
    :param args:
    :return:
    """
    # print("Working on grabbing the adjacency matrix from the graph_cache.")
    adj_matrix = np.zeros((args.b_manual_nodes, args.b_manual_nodes))
    if args.use_future_latent:
        for i, matrix in enumerate(graph_cache):
            adj_matrix[:, i] = matrix[:, i]
    else:
        adj_matrix = graph_cache[0]
    return adj_matrix


def make_it_array(llist):
    new_ls = []
    for ordx, i in enumerate(llist):
        # print("du: ", ordx)
        cache = torch.cat(i, dim=0)
        # print("shape of cache: ", cache.shape)
        new_ls.append(cache)
    return torch.cat(new_ls, dim=0).cpu().detach().numpy()

def structural_inference_pipeline(predict, args):
    """
    The pipeline for structural inference with ppcor.
    The results will be a list of adjacency matrices.
    Each adjacency matrix corresponds to a trajectory.
    :param predict:
    :param args:
    :return: a list of adjacency matrices
    """
    # predict = np.concatenate([x.cpu().detach().numpy() for x in predict], axis=0)
    predict = make_it_array(predict)
    # print("predict: ", predict.shape)
    # predict = np.reshape(predict, (-1, args.timesteps - args.lag_time, args.b_manual_nodes, 2))
    predict = np.reshape(predict, (-1, args.timesteps, args.b_manual_nodes, 2))
    # print("after reshape, predict: ", predict.shape)

    res = []
    for i in predict[:, :-1, :, 0]:
        # print(len(i))
        # print(type(i))
        # print(type(i[0]))
        # print(type(i[1]))

        # cache = np.concatenate((i[0].cpu().detach().numpy(), i[1].cpu().detach().numpy()), axis=0)
        # cache = np.concatenate(i[0].cpu().detach().numpy(), axis=0)
        cache = i
        # print(cache.shape)
        cache = np.reshape(cache, (-1, args.timesteps - args.lag_time, args.b_manual_nodes))
        # cache = np.reshape(cache, (-1, args.timesteps, args.b_manual_nodes))
        # z = cache[:, 0]
        z = np.reshape(cache, (-1, args.timesteps - args.lag_time, args.b_manual_nodes))
        # z = np.reshape(cache, (-1, args.timesteps, args.b_manual_nodes))
        for traj in z:
              # be careful with the nodes order
            # print("traj before ppcor: ", traj.shape)
            if args.use_future_latent:
                feat_ls = reframe_latent(traj, args)
                # print("number of nodes: ", feat_ls[0].shape[-1])
            else:
                feat_ls = [np.reshape(z, (-1, args.b_manual_nodes))]
                # print("feat_ls: ", len(feat_ls), feat_ls[0].shape)
            # print("Ready for ppcor!")
            graph_cache = []
            if args.use_future_latent:
                # print("Using future latent features for ppcor.")
                for j, feat in enumerate(feat_ls):
                    # print("Working on node: ", j)
                    # print("feat: ", feat.shape)
                    interaction_graph = ppcor_python_p_values(np.reshape(feat, (args.b_manual_nodes, -1)), args)
                    graph_cache.append(interaction_graph)
                    # print("shape of interaction graph: ", interaction_graph.shape)
                graph_cache = np.array(graph_cache)  # a_ij -> an edge from i to j.
            else:
                graph_cache = graph_cache.append(ppcor_python_p_values(np.reshape(feat_ls[0], (args.b_manual_nodes, -1)), args))
            res.append(grab_the_adjacency_matrix(graph_cache, args))
    return res


def calculate_auroc(adj_matrix_ls, true_adj_matrix):
    """
    Calculate the AUROC for each trajectory.
    :param adj_matrix_ls: a list of adjacency matrices
    :param true_adj_matrix:
    :return: average AUROC
    """
    res = []
    for adj_matrix in adj_matrix_ls:
        i = np.nan_to_num(adj_matrix, nan=0.0)
        res.append(roc_auc_score(true_adj_matrix.flatten(), i.flatten()))
    res = np.array(res)
    print("AVG AUROC: ", res.mean())
    return res.mean(), res


def store_adj_and_results(adj_matrix_ls, true_adj_matrix, auroc_score, args):
    """
    Store the adjacency matrices and the AUROC scores.
    :param adj_matrix_ls: list of adjacency matrices, [ndarrays]
    :param true_adj_matrix: ground truth adjacency matrix, ndarray
    :param auroc_score: list of AUROC scores, [floats]
    :param args:
    :return: None
    """
    np.save(os.path.join(args.res_folder_path, "adj_matrix_ls.npy"), np.array(adj_matrix_ls))
    np.save(os.path.join(args.res_folder_path, "true_adj_matrix.npy"), true_adj_matrix)
    np.save(os.path.join(args.res_folder_path, "auroc_score.npy"), np.array(auroc_score))
    print("Results saved at ", args.res_folder_path)


class BatchSampler(object):
    """Wraps another sampler to yield a mini-batch of indices.

    Parameters
    ----------
    sampler : Sampler
        Base sampler.
    batch_size : int
        Size of mini-batch.
    drop_last : bool
        If ``True``, the sampler will drop the last batch if its size
        would be less than ``batch_size``

    Example
    -------
    >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x - x.mean()
    r = np.correlate(x, x, mode='full')[-n:]
    assert np.allclose(r, np.array(
        [(x[:n - k] * x[-(n - k):]).sum() for k in range(n)]))
    result = r / (variance * (np.arange(n, 0, -1)))
    return result


def initialize_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.xavier_uniform_(m.weight.data)
    elif isinstance(m, nn.GRU):
        for weights in m.all_weights:
            for weight in weights:
                if len(weight.size()) > 1:
                    init.xavier_uniform_(weight.data)

## Utils to handle newer PyTorch Lightning changes from version 0.6
## ==================================================================================================== ##


def data_loader(fn):
    """
    Decorator to handle the deprecation of data_loader from 0.7
    :param fn: User defined data loader function
    :return: A wrapper for the data_loader function
    """

    def func_wrapper(self):
        try: # Works for version 0.6.0
            return pl.data_loader(fn)(self)

        except: # Works for version > 0.6.0
            return fn(self)

    return func_wrapper


if __name__ == '__main__':
    print("This is utils.py")
