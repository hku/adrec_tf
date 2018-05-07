import numpy as np
from scipy.sparse import hstack
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

from cfg import cfg
from util import read_raw_data, get_selected_input

from tffm import TFFMClassifier

print("read features...")
features, y = read_raw_data("encoded_train_sample.data")
y = (y + 1)/2

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)


for train_index, test_index in sss.split(features, y):
	y_train = y[train_index]
	y_test = y[test_index]
	# print("train len: %d, test len: %d" % (len(yy_train), len(yy_test)))
	# print("train sum: %d, test sum: %d" % (sum(yy_train), sum(yy_test)))
	features_train = features[train_index]
	features_test = features[test_index]	

	sparse_u_train = get_selected_input(features_train, "u_all", cfg)
	sparse_ad_train = get_selected_input(features_train, "ad_all", cfg)
	
	sparse_u_test = get_selected_input(features_test, "u_all", cfg)
	sparse_ad_test = get_selected_input(features_test, "ad_all", cfg)

	sparse_x_train = hstack([sparse_u_train, sparse_ad_train]).tocsr()
	sparse_x_test = hstack([sparse_u_test, sparse_ad_test]).tocsr()


order = 2
model = TFFMClassifier(
    order=order, 
    rank=10, 
    optimizer=tf.train.AdamOptimizer(learning_rate=0.001), 
    n_epochs=50, 
    batch_size=50,
    init_std=0.001,
    reg=0.01,
    input_type='sparse',
    seed=42
)
model.fit(sparse_x_train, y_train, show_progress=False)
predictions = model.predict(sparse_x_test)
print('[order={}] accuracy: {}'.format(order, roc_auc_score(y_test, predictions)))
model.destroy()