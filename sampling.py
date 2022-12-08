import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.metrics import mean_squared_error


class sampling(object):
    def __init__(self, dim_x, s_size, batch_size=50):
        
        self.dim_x = dim_x
        self.s_size = s_size
        self.batch_size = batch_size
        
        # variables
        self.G = tf.Graph()
        self.G.as_default()

        # teacher (does not necessarily be a neural network)
        self.x = tf.placeholder(tf.float32, shape = (None, self.dim_x))
        self.y = tf.placeholder(tf.float32, shape = (None, 1))
        self.y_pred = self.teacher(self.x)
        self.vars_T = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='teacher')

        self.loss_T = tf.reduce_mean(tf.square(tf.subtract(self.y, self.y_pred)))
        self.train_T = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss_T, var_list=self.vars_T)
        self.saver_T = tf.train.Saver(var_list=self.vars_T)

        self.s_y_pred = self.student(self.x, s_size, 'student')
        self.vars_S = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='student')

        self.loss_S = tf.reduce_mean(tf.square(tf.subtract(self.y_pred, self.s_y_pred)))

        self.train_S = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss_S, var_list=self.vars_S)
        self.saver_S = tf.train.Saver(var_list=self.vars_S)

        self.sess = tf.Session()
                  
    def test(self, X_tst, Y_tst):
        Y_tst_hat_s = self.sess.run(self.s_y_pred, feed_dict = {self.x: X_tst})
        tst_rmse_s = mean_squared_error(Y_tst, Y_tst_hat_s)**0.5    
        return tst_rmse_s
    
    def train(self, X_tst, Y_tst, teacher_path, iteration=2000, n_s=10):
        self.sess.run(tf.global_variables_initializer())
        self.saver_T.restore(self.sess, teacher_path)
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == len(self.vars_S+self.vars_T)

        for epoch in range(iteration):
            for i in range(n_s):
                self.sess.run(self.train_S, feed_dict={self.x: np.random.randn(self.batch_size, self.dim_x)})    
    
    def teacher(self, x, reuse=False):
        with tf.variable_scope('teacher', reuse=reuse):
            for _ in range(1):
                x = tf.layers.dense(x, 500, activation = tf.nn.tanh)                               
            o = tf.layers.dense(x, 1)
        return o

    def student(self, x, s_size, name='', reuse = False):
        with tf.variable_scope(name, reuse=reuse):
            for _ in range(1):
                x = tf.layers.dense(x, s_size, activation = tf.nn.tanh)    
            o = tf.layers.dense(x, 1)
        return o