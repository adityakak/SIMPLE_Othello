import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# from tensorflow import keras
# from keras.layers import BatchNormalization, Activation, Flatten, Conv2D, Add, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, Flatten, Conv2D, Add, Dense, Dropout, Lambda

from stable_baselines.common.policies import ActorCriticPolicy
from stable_baselines.common.distributions import CategoricalProbabilityDistributionType, CategoricalProbabilityDistribution
from stable_baselines import logger
import tensorflow as tf

ACTIONS = 64

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
        super(CustomPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=True)

        with tf.variable_scope("model", reuse=reuse):
            
            obs, legal_actions = split_input(self.processed_obs, 1)
            legal_actions_reshaped = tf.reshape(legal_actions, shape=(-1, 64))
                        
            # print(f'Original Obs {self.processed_obs}')
            # print(f'Obs {obs}')
            # print(f'Legal_Actions Original {legal_actions}')
            # print(f'Legal_Actions Reshaped {legal_actions_reshaped}')
            
            # extracted_features = resnet_extractor(self.processed_obs, **kwargs)
            # self._policy = policy_head(extracted_features)
            
            extracted_features = resnet_extractor(obs, **kwargs)
            self._policy = policy_head(extracted_features, legal_actions_reshaped)
            self._value_fn, self.q_value = value_head(extracted_features)

            self._proba_distribution = CategoricalProbabilityDistribution(self._policy)

            
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        else:
            action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
    
def split_input(obs, split):
    return obs[:, :, :, :2], obs[:, :, :, 2:]

def value_head(y):
    # y = convolutional(y, 4, 1)
    # y = Flatten()(y)
    # y = dense(y, 256, batch_norm = False)
    # vf = dense(y, 1, batch_norm = False, activation = 'tanh', name='vf')
    # q = dense(y, 64, batch_norm = False, activation = 'tanh', name='q')

    # y = convolutional(y, 32, 3)
    y = convolutional(y, 4, 1)
    # y = convolutional(y, 64, 1)
    # print(f'Value head y: {y}')
    y = Flatten()(y)
    y = dense(y, 256, batch_norm=False)
    vf = dense(y, 1, batch_norm=False, activation='tanh', name='vf')
    q = dense(y, 64, batch_norm=False, activation='tanh', name='q')
    return vf, q


def policy_head(y, legal_actions):
    # y = convolutional(y, 4, 1)
    # y = Flatten()(y)
    # policy = dense(y, 64, batch_norm = False, activation = None, name='pi')

    y = convolutional(y, 4, 1)
    # y = convolutional(y, 64, 3)
    # y = convolutional(y, 128, 3)
    y = Flatten()(y)
    # print(f'Policy head y: {y}')
    policy = dense(y, 64, batch_norm=False, activation=None, name='pi')
    # print(f"Unmasked Policy {policy}")
    mask = Lambda(lambda x: (1 - x) * -1e8)(legal_actions)   
    
    policy = Add()([policy, mask])
    # print(f"Masked Policy {policy}")
    return policy


def resnet_extractor(y, **kwargs):

    y = convolutional(y, 32, 4)
    for _ in range(5):
        y = residual(y, 32, 4)

    # y = convolutional(y, 64, 3)
    # y = convolutional(y, 128, 3)
    # for _ in range(7):  # Use more residual blocks
    #     y = residual(y, 128, 3)
    return y



def convolutional(y, filters, kernel_size):
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Activation('relu')(y)
    return y

def residual(y, filters, kernel_size):
    shortcut = y

    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Activation('relu')(y)

    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(y)
    y = BatchNormalization(momentum = 0.9)(y)
    y = Add()([shortcut, y])
    y = Activation('relu')(y)

    return y


def dense(y, filters, batch_norm = True, activation = 'relu', name = None):

    if batch_norm or activation:
        y = Dense(filters)(y)
    else:
        y = Dense(filters, name = name)(y)
    
    if batch_norm:
        if activation:
            y = BatchNormalization(momentum = 0.9)(y)
        else:
            y = BatchNormalization(momentum = 0.9, name = name)(y)

    if activation:
        y = Activation(activation, name = name)(y)
    
    return y