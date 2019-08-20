import numpy as np
import scipy.sparse as sp

def np_row_normalize(x: np.ndarray):
    row_sum = np.sum(x, axis = 1).reshape(-1, 1)
    ret = x / row_sum
    ret[np.isnan(ret)] = 0
    return ret

def np_col_normalize(x: np.ndarray):
    col_sum = np.sum(x, axis = 0).reshape(1, -1)
    ret = x / col_sum
    ret[np.isnan(ret)] = 0
    return ret

def np_sym_normalize(x: np.ndarray):
    col_sum = np.sum(x, axis = 1)
    col_sum_sqrt = np.sqrt(col_sum)
    ret = (x / col_sum_sqrt.reshape(-1, 1)) / col_sum_sqrt.reshape(1, -1)
    ret[np.isnan(ret)] = 0
    return ret
    
def np_sym_right_half_normalize(x: np.ndarray):
    col_sum = np.sum(x, axixs = 1).reshape(1, -1)
    col_sum_sqrt = np.sqrt(col_sum)
    ret = x / col_sum_sqrt.reshape(1, -1)
    ret[np.isnan(ret)] = 0
    return ret

def sp_row_normalize(x):
    rowsum = x.toarray().sum(1)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(x)

def sp_col_normalize():
    pass

def sp_sym_normalize(x):
    rowsum = x.toarray().sum(1)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return r_mat_inv_sqrt.dot(x).dot(r_mat_inv_sqrt)

def sp_sym_right_half_normalize(x):
    rowsum = x.toarray().sum(1)
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return x.dot(r_mat_inv_sqrt)