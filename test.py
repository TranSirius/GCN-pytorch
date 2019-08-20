from utils.data_utils import *
from utils.preprocess import *

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')

r = sp_row_normalize(adj)
print(r)