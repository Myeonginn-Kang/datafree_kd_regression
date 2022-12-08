import sys, csv
import pandas as pd
import numpy as np
from proposed import mnnet
from sampling import sampling
from DI import DI
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

dname = sys.argv[1]
iter_id = sys.argv[2]
size = sys.argv[3]
method = sys.argv[4]

np.random.seed(27+int(iter_id))
tf.set_random_seed(27+int(iter_id))

dataset = dname+'.csv'
dataframe = pd.read_csv(dataset, header=None)
data = dataframe.values

n_data = data.shape[0]
dim_x = data.shape[1] - 1
X_data = data[:,0:dim_x]
Y_data = data[:,dim_x:dim_x+1]

X_trn, X_tst, Y_trn, Y_tst = train_test_split(X_data, Y_data, test_size=None, train_size = 5000, random_state = 27+int(iter_id))
scalerX = StandardScaler()
X_trn = scalerX.fit_transform(X_trn)
X_tst = scalerX.transform(X_tst)
scalerY = StandardScaler()
Y_trn = scalerY.fit_transform(Y_trn)
Y_tst = scalerY.transform(Y_tst)

teacher_path = 'teacher'+str(int(iter_id))+'_'+str(dname)+'.ckpt'

if method == 'proposed':
    model = mnnet(dim_x, s_size=int(size))
elif method == 'sampling':
    model = samplingt(dim_x, s_size=int(size))
elif method == 'DI':
    model = DI(dim_x, s_xize=int(size))
    
student = model.train(X_trn, Y_trn, X_tst, Y_tst, teacher_path)

result = open(str(method)+str(iter_id)+'_'+str(dname)+'_size'+str(size)+'.csv','w')
wr = csv.writer(result)
wr.writerow(student)
result.close()
