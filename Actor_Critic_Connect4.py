# -*- coding: utf-8 -*-
"""
Created on Thu May 18 11:30:10 2023

@author: Matthew
"""

from ConnectFour_Learning_V2 import CfourEnv
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


#%% Actor - Critic Hyperparameters
seed = 42
gamma = 0.9
max_steps_per_episode = 10000
env = CfourEnv()
eps = np.finfo(np.float32).eps.item()

#%% Actor - Critic Network Implementation
num_inputs = 42
num_actions = 7
num_hidden = 256

inputs = layers.Input(shape = (num_inputs,1))
common = layers.Dense(num_hidden, activation = "relu")(inputs)
actor = layers.Dense(num_actions, activation = "softmax")(common)
critic = layers.Dense(1)(common)

model = keras.Model(inputs = inputs, outputs = [actor, critic])

#%% Network Training
optimizer = keras.optimizers.Adam(learning_rate=0.01) ##Look into different optimizers?
huber_loss = keras.losses.Huber() ##Look into different loss functions?
action_probs_history = []
critic_value_history = []
rewards_history = []
running_reward = 0
episode_count = 0

while True:
    state = env.reset()
    episode_reward = 0
    with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            state = tf.convert_to_tensor(state.flatten())
            state = tf.expand_dims(state,0)
            
            #predict action probabilities and estimated future rewards from environment state
            action_probs, critic_value = model(state)
            critic_value_history.append(critic_value[0,0])
            
            #sample actionfrom probability distribution 
            probs = np.array(action_probs[0,0,:])
            whereNaN = np.isnan(probs)
            probs[whereNaN] = 0
            probs /= probs.sum()
            action = np.random.choice(num_actions, p=probs)
            action_probs_history.append(tf.math.log(action_probs[0,action])) #why are we appending the log instead of the actual value?
            
            #apply sampled action in our environment
            state, reward, done, _ = env.step(action)
            rewards_history.append(reward)
            episode_reward += reward
            
            if done:
                break
        # update running reward to check condition for solving
        running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward
        
        # Calculate expected value from rewards
        # - At each timestep what was the total reward received after that timestep
        # - Rewards in the past are discounted by multiplying them with gamma
        # - These are the labels for our critic    
        
        returns = []
        discounted_sum = 0
        for r in rewards_history[::-1]:
            discounted_sum = r + gamma * discounted_sum
            returns.insert(0,discounted_sum)
            
        #normalize returns
        returns = np.array(returns)
        returns = (returns - np.mean(returns)) / (np.std(returns) + eps)
        returns = returns.tolist()
        
        #calculating loss values to update network
        history = zip(action_probs_history, critic_value_history, returns)
        actor_losses = []
        critic_losses = []
        for log_prob, value, ret in history:
            # At this point in history, the critic estimated that we would get a
            # total reward = `value` in the future. We took an action with log probability
            # of `log_prob` and ended up recieving a total reward = `ret`.
            # The actor must be updated so that it predicts an action that leads to
            # high rewards (compared to critic's estimate) with high probability.
            diff = ret - value
            actor_losses.append(-log_prob * diff) #actor loss
            
            #the critic must be updated so that it predicts a better estimate of future rewards
            critic_losses.append(huber_loss(tf.expand_dims(value,0), tf.expand_dims(ret,0)))
            
        #backpropogation
        loss_value = sum(actor_losses) + sum(critic_losses)
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        #clear los and reward history
        action_probs_history.clear()
        critic_value_history.clear()
        rewards_history.clear()
        
        #log details
        episode_count += 1
        if episode_count %10 == 0:
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))
            
        if running_reward >200:
            print("solved at episode {}!".format(episode_count))
            break
        
        if episode_count >= 10000:
           print('Failed to solve in 10000 iterations')
           break
        
            
            
            
            