import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from sklearn.metrics import mean_squared_error

class DI(object):
    def __init__(self, dim_x, s_size, delta=0.00001, dim_z=50, batch_size=50):
        
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.delta = delta
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

        # generator & student
        self.y_prior = tf.placeholder(tf.float32, shape = (None, 1))
        self.z = tf.placeholder(tf.float32, shape = (None, self.dim_z))

        self.latent = self.generator(self.z, self.y_prior, self.dim_x)
        self.latent_mu, self.latent_lsgms = tf.split(self.latent, [self.dim_x, self.dim_x], 1)
        self.latent_epsilon = tf.random_normal([self.batch_size, self.dim_x], 0., 1.)
        self.g_x = tf.add(self.latent_mu, tf.multiply(tf.exp(0.5*self.latent_lsgms), self.latent_epsilon))
        self.t_pred = self.teacher(self.g_x, reuse=True)

        self.vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')    
        self.loss_G = tf.reduce_mean(tf.square(tf.subtract(self.y_prior, self.t_pred))) + self.delta*tf.reduce_mean(self.kld(self.latent_mu, self.latent_lsgms)) 
        self.train_G = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss_G, var_list=self.vars_G)

        self.s_t_pred = self.student(self.g_x, self.s_size, 'student')
        self.s_y_pred = self.student(self.x, self.s_size, 'student', reuse=True)
        self.vars_S = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='student')

        self.frac_S = tf.placeholder(tf.float32)
        self.loss_S = self.frac_S * tf.reduce_mean(tf.square(tf.subtract(self.t_pred, self.s_t_pred))) + (1-self.frac_S) * tf.reduce_mean(tf.square(tf.subtract(self.y_pred, self.s_y_pred)))
        self.train_S = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.loss_S, var_list=self.vars_S)
        self.saver_S = tf.train.Saver(var_list=self.vars_S)

        self.sess = tf.Session()
    
    def kld(self, mu, lsgm):
        a = tf.exp(lsgm) + tf.square(mu)
        b = 1 + lsgm 
        kld = 0.5 * tf.reduce_sum(a - b, 1) 
        return kld
  
    def frac(self, epoch):
        frac = 1 - (epoch/2000)         
        return frac
                
    def test(self, X_tst, Y_tst):
        Y_tst_hat_s = self.sess.run(self.s_y_pred, feed_dict = {self.x: X_tst})
        tst_rmse_s = mean_squared_error(Y_tst, Y_tst_hat_s)**0.5    
        return tst_rmse_s
    
    def train(self, X_tst, Y_tst, teacher_path, iteration=2000, n_s=10):
        self.sess.run(tf.global_variables_initializer())
        self.saver_T.restore(self.sess, teacher_path)
        assert len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)) == len(self.vars_G+self.vars_S+self.vars_T)

        for epoch in range(iteration):
            noise_y = np.random.randn(self.batch_size, 1)
            self.sess.run(self.train_G, feed_dict={self.y_prior: noise_y, self.z: np.random.randn(self.batch_size, self.dim_z)})

        for epoch in range(iteration):
            frac = self.frac(epoch)
            for i in range(n_s):  
                self.sess.run(self.train_S, feed_dict={self.y_prior: np.random.randn(self.batch_size, 1), self.z: np.random.randn(self.batch_size, self.dim_z), self.x:np.random.randn(self.batch_size, self.dim_x), self.frac_S:frac})
            
    def teacher(self, x, reuse=False):
        with tf.variable_scope('teacher', reuse=reuse):
            for _ in range(1):
                x = tf.layers.dense(x, 500, activation = tf.nn.tanh)                               
            o = tf.layers.dense(x, 1)
        return o
    
    def generator(self, z, y_prior, dim, reuse=False):
        with tf.variable_scope('generator', reuse=reuse):
            a = tf.concat([z, y_prior], 1)
            for _ in range(1):
                x = tf.layers.dense(a, 500, activation = tf.nn.tanh)
            o = tf.layers.dense(z, 2*dim)
        return o

    def student(self, x, s_size, name='', reuse = False):
        with tf.variable_scope(name, reuse=reuse):
            for _ in range(1):
                x = tf.layers.dense(x, s_size, activation = tf.nn.tanh)    
            o = tf.layers.dense(x, 1)
        return o