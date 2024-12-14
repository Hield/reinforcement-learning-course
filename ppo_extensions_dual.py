from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time
from collections import namedtuple


# Hyperparameters
BUFFER_SIZE = 5000
BUFFER_PRIORITY_DECAY = 0.98
BUFFER_BATCH_SIZE = 32
BUFFER_ALPHA = 0.6
BUFFER_BETA = 0.1
DUAL_CLIP_VAL = 5.0
SIL_LOSS_WEIGHT = 0.25


Batch = namedtuple('Batch', ['state', 'action', 'retur'])


class ReplayBuffer(object):
    def __init__(self,
                 state_shape:tuple,
                 action_dim: int,
                 max_size=BUFFER_SIZE,
                 alpha=BUFFER_ALPHA,
                 priority_decay=BUFFER_PRIORITY_DECAY):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.priority_decay = priority_decay

        dtype = torch.float32 # unit8 is used to store images
        self.state = torch.zeros((max_size, state_shape), dtype=dtype)
        self.action = torch.zeros((max_size, action_dim), dtype=dtype)
        self.retur = torch.zeros(max_size, dtype=dtype)
        
        self.priorities = np.zeros((max_size,), dtype=np.float32)
    
    def _to_tensor(self, data, dtype=torch.float32):   
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype)
        return torch.tensor(data, dtype=dtype)

    def add(self, state, action, retur, priority=1.0):
        if self.ptr % 100 == 0:  # Decay every 100 transitions
            self.decay_priorities()
        self.state[self.ptr] = self._to_tensor(state, dtype=self.state.dtype)
        self.action[self.ptr] = self._to_tensor(action)
        self.retur[self.ptr] = self._to_tensor(retur)
        
        self.priorities[self.ptr] = max(priority, 1e-6)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size=BUFFER_BATCH_SIZE, beta=BUFFER_BETA, device='cpu'):
        if self.size == 0:
            raise ValueError("The replay buffer is empty!")
        
        # Compute probabilities for sampling
        scaled_priorities = self.priorities[:self.size] ** self.alpha
        sampling_probs = scaled_priorities / np.sum(scaled_priorities)
        
        # Sample indices based on probabilities
        indices = np.random.choice(self.size, batch_size, p=sampling_probs)

        batch = Batch(
            state = self.state[indices].to(device),
            action = self.action[indices].to(device),
            retur = self.retur[indices].to(device),
        )
        
        # Compute importance-sampling weights
        weights = (self.size * sampling_probs[indices]) ** -beta
        weights /= weights.max()  # Normalize weights
        return batch, indices, torch.tensor(weights, dtype=torch.float32).to(device)
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled transitions based on TD errors.
        """
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = max(abs(error), 1e-6)
            
    def decay_priorities(self):
        """Decay priorities of all items in the buffer."""
        self.priorities[:self.size] *= self.priority_decay


class PPOExtension(PPOAgent):
    def __init__(self, config=None):
        super(PPOExtension, self).__init__(config)
        self.sil_buffer = ReplayBuffer(self.observation_space_dim, self.action_space_dim)
        self.ratio_of_episodes = 1

    def store_sil(self, states, actions, returns):
        for i in range(len(states)):
            self.sil_buffer.add(states[i], actions[i], returns[i])

    
    def sil_update(self):
        M=1
        
        for _ in range(M):
            # beta goes from 0.1 to 1
            beta = 0.1 + (1 - self.ratio_of_episodes) * 0.9
            sil_batch, sil_indices, sil_weights = self.sil_buffer.sample(beta=beta)
        
            sil_states = sil_batch.state
            sil_actions = sil_batch.action
            sil_returns = sil_batch.retur

            sil_action_dists, sil_values = self.policy(sil_states)
            sil_values = sil_values.squeeze()

            # Calculate raw SIL advantages for updating priorities
            sil_advantages = sil_returns - sil_values
            sil_advantages = torch.clamp(sil_advantages, min=0)

             # Detach here not to involve value gradient
            detached_sil_advantages = sil_advantages.detach()

            # Update priorities in the buffer (use SIL advantages after clamping since we don't want worse values)
            self.sil_buffer.update_priorities(sil_indices, detached_sil_advantages.numpy())
            sil_log_probs = sil_action_dists.log_prob(sil_actions).sum(axis=-1)

            # Scale by importance-sampling weights
            sil_policy_loss = -torch.mean(sil_log_probs * detached_sil_advantages * sil_weights) 
            sil_value_loss = 0.5 * torch.mean(sil_advantages ** 2 * sil_weights)
            
            sil_loss = sil_policy_loss + 0.01 * sil_value_loss
            self.optimizer.zero_grad()
            sil_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.5)
            self.optimizer.step()
    
            
    # old functions
    def ppo_update(self, states, actions, rewards, next_states, dones, old_log_probs, targets):
        # Calculate PPO loss
        action_dists, values = self.policy(states)
        values = values.squeeze()
        new_action_probs = action_dists.log_prob(actions).sum(axis=-1) 
        ratio = torch.exp(new_action_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1-self.clip, 1+self.clip)

        advantages = targets - values
        advantages -= advantages.mean()
        advantages /= advantages.std()+1e-8
        advantages = advantages.detach()
        
        # Dual clip implementation
        dual_clip_val = torch.tensor(DUAL_CLIP_VAL, dtype=advantages.dtype, device=advantages.device)
        L_normal = torch.min(ratio * advantages, clipped_ratio * advantages)
        policy_objective = torch.where(
            advantages < 0,
            torch.max(L_normal, dual_clip_val * advantages),
            L_normal
        )
        policy_objective = -policy_objective.mean()

        value_loss = F.mse_loss(values, targets, reduction="mean")

        entropy = action_dists.entropy().mean()
        ppo_loss = policy_objective + 0.5*value_loss - 0.01*entropy
        '''
        # Calculate SIL loss
        sil_batch, sil_indices, sil_weights = self.sil_buffer.sample()
        
        sil_states = sil_batch.state
        sil_actions = sil_batch.action
        sil_returns = sil_batch.retur
        
        sil_action_dists, sil_values = self.policy(sil_states)
        sil_values = sil_values.squeeze()
        
        # Calculate raw SIL advantages for updating priorities
        sil_advantages = sil_returns - sil_values
        sil_advantages = torch.clamp(sil_advantages, min=0)

         # Detach here not to involve value gradient
        detached_sil_advantages = sil_advantages.detach()
        
        # Update priorities in the buffer (use SIL advantages after clamping since we don't want worse values)
        self.sil_buffer.update_priorities(sil_indices, detached_sil_advantages.numpy())
        sil_log_probs = sil_action_dists.log_prob(sil_actions).sum(axis=-1)
        
        # Scale by importance-sampling weights
        sil_policy_loss = -torch.mean(sil_log_probs * detached_sil_advantages * sil_weights) 
        sil_value_loss = 0.5 * torch.mean(sil_advantages ** 2 * sil_weights)
        sil_loss = sil_policy_loss + 0.01 * sil_value_loss

        '''
        loss = ppo_loss #+ SIL_LOSS_WEIGHT * sil_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.5)
        self.optimizer.step()
    
    
    def ppo_epoch(self):
        indices = list(range(len(self.states)))
        returns = self.compute_returns()
        
        #save in sil replay buffer
        self.store_sil(self.states, self.actions, returns)
        
        while len(indices) >= self.batch_size:
            # Sample a minibatch
            batch_indices = np.random.choice(indices, self.batch_size,
                    replace=False)
            
            # Do the update
            self.ppo_update(self.states[batch_indices], self.actions[batch_indices],
                self.rewards[batch_indices], self.next_states[batch_indices],
                self.dones[batch_indices], self.action_log_probs[batch_indices],
                returns[batch_indices])
            
            self.sil_update()
            # Drop the batch indices
            indices = [i for i in indices if i not in batch_indices]
            
    
    def train_iteration(self, ratio_of_episodes):
        self.ratio_of_episodes = ratio_of_episodes
        update_info = super().train_iteration(ratio_of_episodes)
        return update_info
