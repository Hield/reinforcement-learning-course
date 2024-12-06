from .agent_base import BaseAgent
from .ppo_utils import Policy
from .ppo_agent import PPOAgent

import utils.common_utils as cu
import torch
import numpy as np
import torch.nn.functional as F
import time

class PPOExtension(PPOAgent):
    
    def __init__(self, config=None):
        super(PPOExtension, self).__init__(config)
        self.sil_buffer = []

    def store_sil(self,states,actions,returns):
        self.sil_buffer.extend([(states[i], actions[i], returns[i]) for i in range(0, len(states))])

        
        
    def sil_update(self):
        if not self.sil_buffer:
            return 0.0  # do not add loss if sil buffer is empty
        
        M=2
        sil_batch_size = 16
        
        
        
        for _ in range(M):
            index = np.random.choice(len(self.sil_buffer), sil_batch_size,replace=False)
            sil_mini_batch = [self.sil_buffer[i] for i in index]
            states, actions, returns = zip(*sil_mini_batch)
            
            states = torch.stack(states)
            actions = torch.stack(actions)
            returns = torch.stack(returns)

            action_dists, values = self.policy(states)
            values = values.squeeze()
            
            advantages = returns - values
            advantages = torch.clamp(advantages, min=0)

            log_probs = action_dists.log_prob(actions).sum(axis=-1)
            sil_policy_loss = -torch.mean(log_probs * advantages)
            sil_loss_value = 0.5 * torch.mean(advantages ** 2)
            
            beta_sil = 1
            
            sil_loss = sil_policy_loss + beta_sil * sil_loss_value
            
            self.optimizer.zero_grad()
            sil_loss.backward()
            self.optimizer.step()

            
# old functions
    
    
    
    def ppo_epoch(self):
        indices = list(range(len(self.states)))
        returns = self.compute_returns()
        
        #save in sil replay buffer
        self.store_sil(self.states,self.actions,returns)
        
        while len(indices) >= self.batch_size:
            # Sample a minibatch
            batch_indices = np.random.choice(indices, self.batch_size,
                    replace=False)

            # Do the update
            self.ppo_update(self.states[batch_indices], self.actions[batch_indices],
                self.rewards[batch_indices], self.next_states[batch_indices],
                self.dones[batch_indices], self.action_log_probs[batch_indices],
                returns[batch_indices])
            
            #new additional update function
            self.sil_update()

            # Drop the batch indices
            indices = [i for i in indices if i not in batch_indices]
        
