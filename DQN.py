import gym
from collections import deque
import os
import tensorflow as tf
import numpy as np
import gym
import  tensorflow.contrib.layers as layers
import random

GAME = 'Pong'
ACTIONS = [0,2,3] # Open AI gym Atari env: 0 :'still', 2:'up', 3 'down'
nACTIONS=len(ACTIONS) #Number of actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 5000. # timesteps to observe before training
EXPLORE = 10000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1 # final value of epsilon
INITIAL_EPSILON = 1.0 # starting value of epsilon
REPLAY_MEMORY = 100000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
K = 4 # only select an action every Kth frame, repeat prev for others


def greyscale(state):
    """
    Preprocess state (210, 160, 3) image into
    a (80, 80, 1) image in grey scale
    """
    state = np.reshape(state, [210, 160, 3]).astype(np.float32)

    # grey scale
    state = state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114

    # karpathy
    state = state[35:195]  # crop
    state = state[::2,::2] # downsample by factor of 2

    return state.astype(np.uint8)



def createNetwork():
    # input layer
    state = tf.placeholder("float", [None, 80, 80, 4])
    with tf.variable_scope('CNN2'):
        X = layers.conv2d(state, 32, 8, stride=4, )
        X = layers.conv2d(X, 64, 4, stride=2, )
        X = layers.conv2d(X, 64, 3, stride=1, )
        X = layers.flatten(X)
        X = layers.fully_connected(X, 512)
        Q_value = layers.fully_connected(X, nACTIONS, activation_fn=None)

    return state, Q_value

def trainNetwork(state, Q_value, sess,total_episode):
    # define the cost function
    a = tf.placeholder("float",[None,nACTIONS])
    y = tf.placeholder("float", [None])
    Qvalue_action = tf.reduce_sum(tf.multiply(Q_value, a), reduction_indices = 1)
    cost = tf.reduce_mean(tf.square(y -Qvalue_action))
    train_step = tf.train.AdamOptimizer(5e-5).minimize(cost)

    # open up a game state to communicate with emulator
    env=gym.make('Pong-v0')
    
   
    #loading networks
    saver = tf.train.Saver()
    #sess.run(tf.initialize_all_variables())
    saver.restore(sess, 'C:\\Users\\maido\\Desktop\\DQN\\save2\\model')

    epsilon = INITIAL_EPSILON
    
    
    # store the previous observations in replay memory
    D = deque(maxlen=REPLAY_MEMORY)
    t = 0
    for ep in range(total_episode):
        x_t=env.reset()
        #convert 210 160 3 into grey scale 80 80 1
        x_t=greyscale(x_t)
        #feed in 4 frames at a time
        s_t = np.stack([x_t]*4, axis = 2)
        
        Return=0
        terminal=False
        while not terminal:
            # choose an action epsilon greedily
            Q_value_t = Q_value.eval(feed_dict = {state :[ s_t.astype(np.float32)]},session=sess)
            a_t = np.zeros([nACTIONS])
            action_index=0
            if random.random() <= epsilon or t <= OBSERVE:
                action_index  = random.randrange(nACTIONS)
                a_t[action_index] = 1
            else:
                action_index  = np.argmax(Q_value_t)
                a_t[action_index] = 1
    
            # scale down epsilon
            if t> (OBSERVE+EXPLORE):
                epsilon=FINAL_EPSILON
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
    
            # run the selected action and observe next state and reward
            x_t1, r_t, terminal,_ = env.step(ACTIONS[action_index])
            Return+=r_t
          
            x_t1 = greyscale(x_t1)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            s_t1 = np.append(x_t1 ,s_t[:,:,0:3], axis = 2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if terminal:
                 print ('Ep',ep,"/TIMESTEP", t, "/ ACTION", a_t, "/ REWARD", Return, 
               "/ Q_MAX %e" % np.max( Q_value_t))
            
    
            # only train if done observing
            if t > OBSERVE and t%K == 0:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)
    
                # get the batch variables
                s_j_batch = [d[0].astype(np.float32) for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3].astype(np.float32) for d in minibatch]
    
                y_batch = []
                Q_value_j1_batch = Q_value.eval(feed_dict = {state : s_j1_batch},session=sess)
                for i in range(0, len(minibatch)):
                    # if terminal only equals reward
                    if minibatch[i][4]:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(Q_value_j1_batch[i]))
    
                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    state : s_j_batch},session=sess)
    
            # update the old values
            s_t = s_t1
            t += 1
    
            # save progress every 3000 iterations
            if t % 3000 == 0:
                saver.save(sess, 'C:\\Users\\maido\\Desktop\\DQN\\save2\\model')

           



def playGame(total_episode):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    state, Q_value = createNetwork()
    trainNetwork(state, Q_value ,sess,total_episode)




playGame(20000)

def test_game(n_episodes):
    env=gym.make('Pong-v0')
    sess = tf.Session()
    state, Q_value = createNetwork()
    saver = tf.train.Saver()
    saver.restore(sess, 'C:\\Users\\maido\\Desktop\\DQN\\save2\\model')
   
    

    for episode in range(n_episodes):
        x_t=env.reset()
        x_t=greyscale(x_t)
        s_t = np.stack([x_t]*4, axis = 2)
        terminated = False
        Return=0
       
        while not terminated:
            action = np.argmax(sess.run(Q_value,feed_dict = {state :[ s_t.astype(np.float32)]}))

            x_t , reward, terminated, info = env.step(ACTIONS[action])
            x_t = greyscale(x_t)
            x_t = np.reshape(x_t, (80, 80, 1))
            s_t = np.append(x_t ,s_t[:,:,0:3], axis = 2)

            Return+=reward
       
            env.render()
        print(Return)
        
test_game(5)

