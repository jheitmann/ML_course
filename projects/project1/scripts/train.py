
import numpy as np
from proj1_helpers import load_csv_data
from implementations import least_squares_SGD

yb, input_data, ids = load_csv_data('all/train.csv', step=50)
losses, w = least_squares_SGD(yb, input_data, initial_w=np.zeros(30), batch_size=1, max_iters=30, gamma=0.5)
with open('output.txt', 'w') as fp:
    print(yb, input_data, losses, w, file=fp)
