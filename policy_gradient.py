import gym
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

def discount_and_norm_rewards(r,GAMMA):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * GAMMA + r[t]
        discounted_r[t] = running_add
     # normalize episode rewards
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r+1e-5)
    return discounted_r



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
    state = tf.placeholder("float", [None, 80, 80])
    with tf.variable_scope('NN'):
        X = layers.flatten(state)
        X = layers.fully_connected(X, 200, activation_fn=tf.nn.relu)
        output = layers.fully_connected(X, nACTIONS, activation_fn=None)

    return state,output

def trainNetwork(state, output, sess,total_episode):
    # define the cost function
    reward_holder = tf.placeholder(shape=[None],dtype=tf.float32)
    action_holder =  tf.placeholder(tf.int32,[None])
        
    all_act_prob = tf.nn.softmax(output, name='act_prob')  # use softmax to convert to probability
    
    neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=action_holder)
    loss = tf.reduce_mean(neg_log_prob * reward_holder)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # open up a game state to communicate with emulator
    env=gym.make('Pong-v0')
    
    #loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    #saver.save(sess, 'C:\\Users\\maido\\Desktop\\Policy_Gradient\\save\\model')

    
    t = 0
    for ep in range(total_episode):
        obs=env.reset()
        Return=0
        terminal=False
        
        state_history=[]
        action_history=[]
        reward_history=[]
        prev_s = None
        while not terminal:
            cur_s= greyscale(obs)
            input_s = cur_s - prev_s if prev_s is not None else np.zeros_like(cur_s)
            prev_s = cur_s
            prob_weights = sess.run(all_act_prob, feed_dict={state: [input_s]})
            action_index =np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())
            
            obs, reward, terminal, _ = env.step(ACTIONS[action_index])
            Return+=reward
          
            

           # ep_history.append((s_t, a, r_t, s_t1))
            state_history.append(input_s)
            action_history.append(action_index)
            reward_history.append(reward)
            if terminal:
                print ('Ep',ep,"/TIMESTEP", t, "/ ACTION", action_index, "/ REWARD", Return) 
                #Update the network.
                state_history = np.array(state_history)
                reward_history= discount_and_norm_rewards(reward_history,GAMMA)
                train_step.run(feed_dict={reward_holder:reward_history,
                                          action_holder:action_history,
                                          state:state_history},session=sess)
            
            # increment timestep
            t += 1
    
            # save progress every 3000 iterations
            if t % 30000 == 0:
                saver.save(sess, 'C:\\Users\\maido\\Desktop\\Policy_Gradient\\save\\model')
        
        
        
def playGame(total_episode):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    state, output = createNetwork()
    trainNetwork(state, output ,sess,total_episode)




playGame(20000)        
        
        
def test_game(n_episodes):
    env=gym.make('Pong-v0')
    sess = tf.Session()
    state, output = createNetwork()
    saver = tf.train.Saver()
    saver.restore(sess, 'C:\\Users\\maido\\Desktop\\Policy_Gradient\\save\\model')
    all_act_prob = tf.nn.softmax(output, name='act_prob')
    

    for episode in range(n_episodes):
        obs=env.reset()
        terminal=False
        prev_s = None
        Return=0
       
        while not terminal:
            cur_s= greyscale(obs)
            input_s = cur_s - prev_s if prev_s is not None else np.zeros_like(cur_s)
            prev_s = cur_s
            prob_weights = sess.run(all_act_prob, feed_dict={state: [input_s]})
            action_index =np.argmax(prob_weights.ravel())
            
            obs, reward, terminal, _ = env.step(ACTIONS[action_index])
    
            Return+=reward
       
            env.render()
        print(Return)
        
test_game(5)
    
        
        
        