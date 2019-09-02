import gym
import os
import tensorflow as tf
import numpy as np
import gym
import tensorflow.layers as layers
import random
from collections import deque

GAME = 'CartPole'
ACTIONS = [0, 1]  # Open AI gym Atari env: 0 :'up', 1 'down'
nACTIONS = len(ACTIONS)  # Number of actions
GAMMA = 0.99  # decay rate of past observations





def create_actor_network():
    # input layer
    state = tf.placeholder("float", [None, 4])
    with tf.variable_scope('actor'):
        X = layers.dense(state, 24, activation=tf.nn.relu,
                         kernel_initializer='he_uniform')
        output = layers.dense(X, nACTIONS, activation=None,
                         kernel_initializer='he_uniform')

    return state, output

def create_critic_network(state,reuse=False):

    with tf.variable_scope('critic',reuse=reuse):
        X = layers.dense(state, 24, activation=tf.nn.relu,
                         kernel_initializer='he_uniform')
        value = layers.dense(X, 1, activation=None,
                         kernel_initializer='he_uniform')

    return value




def trainNetwork(state,output,value, sess, total_episode):

    # define the loss function for critic model
    target_critic=tf.placeholder(tf.float32,[None,1])
    critic_loss = tf.reduce_mean(tf.square(value-target_critic))

    # define the loss function for actor model
    target_actor=tf.placeholder(tf.float32,[None,nACTIONS])
    all_act_prob = tf.nn.softmax(output, name='act_prob')  # use softmax to convert to probability

    actor_loss = tf.keras.losses.categorical_crossentropy(y_pred=output,
                                                y_true=target_actor,from_logits=True)

    #define optimizer and trainable var list
    var_actor=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'actor')
    var_critic=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'critic')
    train_actor = tf.train.AdamOptimizer(1e-3).minimize(actor_loss,var_list=var_actor)

    train_critic = tf.train.AdamOptimizer(5e-3).minimize(critic_loss,var_list=var_critic)
    # open up a game state to communicate with emulator
    env = gym.make('CartPole-v0')

    # loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    #saver.save(sess, 'C:\\Users\\maido\\Desktop\\Actor Critic\\save\\model')

    t = 0
    Return_list=deque(maxlen=1)
    for ep in range(1,total_episode+1):
        obs = env.reset()
        terminal = False
        episode_rewards = 0
        while not terminal:

            cur_s_t = obs

            prob_weights = sess.run(all_act_prob, feed_dict={state: [cur_s_t]})
            action_index = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())


            obs, reward, terminal, _ = env.step(ACTIONS[action_index])
            episode_rewards += reward

            # if an action make the episode end, then gives penalty of -100
            reward = reward if not terminal or episode_rewards == 199 else -100
            next_s_t = obs

            critic_target = np.zeros((1, 1))
            actor_target = np.zeros((1, nACTIONS))

            v_cur = sess.run(value, {state: [cur_s_t]})[0]
            v_next = sess.run(value, {state: [next_s_t]})[0]

            critic_target[0][0] = reward + GAMMA * v_next*(1-terminal)
            actor_target[0][action_index] = reward + GAMMA * v_next*(1-terminal) - v_cur

            train_critic.run({state: [cur_s_t],
                              target_critic: critic_target}, session=sess)

            train_actor.run({state: [cur_s_t],
                             target_actor: actor_target}, session=sess)


            if terminal:
                Return_list.append(episode_rewards)


            # increment timestep
            t += 1


            # save progress every 3000 iterations
            if t % 30000 == 0:
                saver.save(sess, 'C:\\Users\\maido\\Desktop\\Actor Critic\\save\\model')
        #Print out Performance of the model every 300 epochs
        if ep % 1 == 0:
            print('Ep', ep, "/TIMESTEP", t, "/ Average Returns", np.mean(Return_list))


def playGame(total_episode):
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    state, output = create_actor_network()
    value = create_critic_network(state)
    trainNetwork(state, output, value, sess, total_episode)


playGame(20000)
