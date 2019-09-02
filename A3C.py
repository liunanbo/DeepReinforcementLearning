import numpy as np
import tensorflow as tf
import tensorflow.layers as layers
import os
from scipy.misc import imresize
import gym
import threading

def copy_src_to_dst(from_scope,to_scope):
    from_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,from_scope)
    to_vars=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,to_scope)
    op_holder=[]

    for from_var,to_var in zip(from_vars,to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder




def preprocess(obs):
    #crop top and bottom
    obs=obs[35:195]
    obs=imresize(obs,(80,80),interp='nearest')

    H, W, _ = obs.shape

    R = obs[..., 0]
    G = obs[..., 1]
    B = obs[..., 2]
    cond = (R == 144) & (G == 72) & (B == 17)

    obs = np.zeros((H, W))
    obs[~cond] = 1

    obs=np.expand_dims(obs,2)
    return obs

def discounted_rewards(r,GAMMA=0.99):
    discounted_r=np.zeros_like(r)
    running_add=0
    for t in reversed(range(0,len(r))):

        if r[t]!=0:
            running_add=0
        running_add=running_add*GAMMA+r[t]
        discounted_r[t]=running_add

    discounted_r-=np.mean(discounted_r)
    discounted_r/=np.std(discounted_r)+1e-8
    return discounted_r


class A3C_Network(object):

    def __init__(self,name):
        with tf.variable_scope(name):
            self.GAMMA=0.99
            self.nActions=3
            self.state_shape=[80,80,1]



            self.state=tf.placeholder(tf.float32, [None,*self.state_shape],name='state')

            net=layers.conv2d(self.state,32,8,4,activation=tf.nn.relu)
            net=layers.conv2d(net,64,4,2,activation=tf.nn.relu)
            net = layers.conv2d(net, 64, 3, 1, activation=tf.nn.relu)
            net=tf.contrib.layers.flatten(net)
            net=layers.dense(net,512,activation=tf.nn.relu)

            #Use Shared network to define actor logtis
            actor_logits=layers.dense(net,self.nActions,activation=None,name='actor_logits')
            self.action_prob=tf.nn.softmax(actor_logits,name='action_prob')


            #Use Shared network to define critic value
            self.critic_value=tf.squeeze(layers.dense(net,1,activation=None,name='critic_value'))


            self.actions= tf.placeholder(tf.int32,[None],name='actions')
            self.advantage=tf.placeholder(tf.float32,[None],name='advantage')
            self.Return=tf.placeholder(tf.float32,[None],name='Return')

            action_onehot = tf.one_hot(self.actions, self.nActions, name="action_onehot")
            single_action_prob = tf.reduce_mean(self.action_prob * action_onehot, axis=1)

            #Create actor Loss
            entropy=tf.reduce_sum(-self.action_prob*tf.log(self.action_prob+1e-7),axis=1)

            log_action_prob= tf.log(single_action_prob + 1e-7)
            actor_loss= -tf.reduce_mean(log_action_prob*self.advantage+entropy*0.005)

            #Create Critic loss
            critic_loss=tf.losses.mean_squared_error(labels=self.Return
                                                     ,predictions=self.critic_value)

            #define Total Loss
            self.total_loss=actor_loss+critic_loss*0.5

            self.optimizer=tf.train.RMSPropOptimizer(learning_rate=1e-3,decay=0.99)

        var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=name)

        self.gradients=self.optimizer.compute_gradients(self.total_loss,var_list)
        self.gradients_placeholders=[]

        for grad,var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype,shape=var.get_shape()),var))
        self.apply_gradients= self.optimizer.apply_gradients(self.gradients_placeholders)

        self.graph_var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)




class Agent(threading.Thread):
    def __init__(self,env,Global_network,sess,coord,name,saver,save_path):

        super(Agent,self).__init__()  # Initialize Parent Class : threading.Thead
        self.local=A3C_Network(name)
        self.global_to_local=copy_src_to_dst('global',name)
        self.Global_network = Global_network
        self.env=env
        self.state_shape=[80,80,1]

        self.sess=sess
        self.nActions=3
        self.coord=coord
        self.name=name
        self.state_shape=[80,80,1]
        self.action_space=[1,2,3] # 1: 'Still' , 2: 'Up',  3: 'Down'
        self.saver = saver
        self.save_path = save_path

    def print(self,reward):
        print('Agent :'+self.name+', reward= %d' %reward)

    def choose_action(self, states):
        states = np.reshape(states, [-1, *self.state_shape])
        action = self.sess.run(self.local.action_prob, {self.local.state: states})
        action = np.squeeze(action)
        action_index = np.random.choice(np.arange(self.nActions), p=action)

        return action_index

    def train(self,states,actions,rewards):
        states= np.array(states)
        actions= np.array(actions)
        rewards= np.array(rewards)

        value=self.sess.run(self.local.critic_value,{self.local.state : states})

        Return= discounted_rewards(rewards,GAMMA=0.99)



        advantage= Return-value
        advantage-= np.mean(advantage)
        advantage/= np.std(advantage)+1e-8

        gradients= self.sess.run(self.local.gradients, {
            self.local.state: states,
            self.local.actions : actions,
            self.local.Return: Return,
            self.local.advantage: advantage
        })

        feed_dict={}
        for(grad,_),(placeholder,_) in zip(gradients,self.Global_network.gradients_placeholders):
            feed_dict[placeholder]=grad

        self.sess.run(self.Global_network.apply_gradients,feed_dict)


    def play_episode(self):


        self.sess.run(self.global_to_local)
        states,actions,rewards = [],[],[]
        obs = self.env.reset()
        s=preprocess(obs)
        state_diff = s

        terminate=False
        total_reward = 0
        time_step = 0
        while not terminate:
            action_index=self.choose_action(state_diff)
            a = self.action_space[action_index]
            obs2, r, terminate, _ = self.env.step(a)
            s2 = preprocess(obs2)
            total_reward += r

            states.append(state_diff)
            actions.append(action_index)
            rewards.append(r)

            state_diff = s2 - s
            s = s2

            if r == -1 or r == 1 or terminate:
                time_step += 1

                if time_step >= 5 or terminate:
                    self.train(states,actions,rewards)
                    self.sess.run(self.global_to_local)
                    states, actions, rewards = [], [], []
                    time_step=0

        self.print(total_reward)

    def run(self):
        while not self.coord.should_stop():

            self.play_episode()

            # Save Global Model  After every Episode
            self.saver.save(self.sess, self.save_path)


def run_model(n_threads=8):
    try:
        tf.reset_default_graph()
        sess=tf.InteractiveSession()
        coord=tf.train.Coordinator()

        checkpoint_dir='save'

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)


        save_path = '.'+os.sep+checkpoint_dir+os.sep+'Pong_save'

        #Define Global network
        global_network=A3C_Network(name='global')

        thread_list=[]
        env_list=[]

        #Define global network Saver
        saver=tf.train.Saver(var_list=global_network.graph_var_list)
        for id in range(1,n_threads+1):
            # Create local environment
            env=gym.make('Pong-v0')

            # Define local network
            single_agent=Agent(env=env,Global_network=global_network,
                               sess=sess,coord=coord,name='thread_%d'%id,
                               saver=saver,save_path=save_path)
            thread_list.append(single_agent)
            env_list.append(env)

        if tf.train.latest_checkpoint(checkpoint_dir):

            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir))
            print('Model restored to global')

        else:
            sess.run(tf.global_variables_initializer())
            print('No Model is found!')

        for t in thread_list:
            t.start()

        print('Ctrl + C to Close')
        coord.wait_for_stop()
    except KeyboardInterrupt:
        # Save Global Model Parameters

        saver.save(sess,save_path)
        print('Checkpoint Saved to %s'%save_path)

        print('Closing threads')
        coord.request_stop()
        coord.join(thread_list)

        print('Closing environments')
        for env in env_list:
            env.close()

        sess.close()



if __name__ == '__main__':
    run_model(n_threads=8)





























































