import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

from collections import namedtuple, deque
from itertools import count


import threading
from concurrent.futures import ThreadPoolExecutor

import math


Transition = namedtuple('Transition',
                        ('state', 'reward', 'action', 'c_flag', 'used_quant'))

VQ_State = namedtuple('VQ_State',
                        ('state', 'returns', 'action'))


eps=1e-10
_eps = torch.tensor(eps, device='cuda')


def get_highest_l1_vectors(vectors, ammount=256):
    distances = torch.cdist(vectors, vectors)
    
    # Initialize a list to store the sampled indices
    sampled_indices = []
    
    # Start by randomly selecting the first index
    sampled_indices.append(torch.randint(0, vectors.size(0), (1,)).item())
    
    # Repeat until you have sampled 3 vectors
    while len(sampled_indices) < ammount:
        # Compute the distances from the already sampled vectors to all others
        sampled_distances = distances[sampled_indices, :].min(dim=0).values
        
        # Select the index with the maximum distance as the next sampled index
        next_index = sampled_distances.argmax().item()
        # Add the index to the list of sampled indices
        sampled_indices.append(next_index)
    
    # Extract the sampled vectors
    return vectors[sampled_indices], sampled_indices



class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        n, c, h, w = x.size()
        
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (w + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                w + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:w]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)

        eps = 1.0 / (h + 2 * 4)
        arange2 = torch.linspace(-1.0 + eps,
                                        1.0 - eps,
                                        h + 2 * 4,
                                        device=x.device,
                                        dtype=x.dtype)[:h]
        arange2 = arange2.unsqueeze(0).repeat(w, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange2.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        
        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        
        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)

crop_aug = RandomShiftsAug(pad=4)



def preprocess_replay(x):
    batch, seq, cnn_shape = x.shape[0], x.shape[1], x.shape[-3:]
    x = crop_aug(x.view(-1, *cnn_shape)).view(batch, seq, *cnn_shape)

    noise = 1 + (0.05*torch.rand_like(x).clip(-2.0,2.0))
    return x * noise







class PrioritizedReplay_nSteps_Sqrt(object):
    def __init__(self, capacity, final_beta=1.0, initial_beta=0.4, total_steps=40000, prefetch_cap=8, cluster_size=256):
        self.capacity = capacity
        self.cluster_size = cluster_size
        self.memory = deque([],maxlen=capacity)
        self.total_steps=total_steps

        self.initial_beta=initial_beta
        self.final_beta=final_beta
        
        
        self.free()
        

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        self.n+=1
    

    def prioritize(self, batch_size, seq_len, step):
        with torch.no_grad():
            max_n = int(self.n -seq_len -5 -1)

            priority = self.priority.clone()[:max_n].pow(0.5)
            probs = priority/priority.sum()
            _, sorted_priorities = priority.sort()

            segment_length=(max_n//batch_size)
            idxs = torch.randint(0, segment_length, (batch_size,))
            idx = idxs + torch.arange(batch_size)*segment_length
            idx = sorted_priorities[idx]

            is_w = (1/(probs*max_n+eps)).pow(0.5)
            is_w/=is_w.max()
            
            return idx, is_w[idx]

    
    def sample(self, seq_len, batch_size, step):
        
        states, action, rewards, c_flag, used_quant, idxs = [], [], [], [], [], []
        
        idxs, is_ws = self.prioritize(batch_size, seq_len, step)
        for idx in idxs:
            batch = Transition(*zip(*list([self.memory[int(i+idx)] for i in range(max( int(seq_len*1.5), 5+1 ))])))
            
            states.append(torch.stack(batch.state).squeeze(1).cuda())
            
            rewards.append(torch.stack(batch.reward).cuda())
            action.append(torch.stack(batch.action).cuda())
            c_flag.append((~torch.stack(batch.c_flag).cuda()).float())
            used_quant.append(batch.used_quant[0].cuda())
            
        

        states = torch.stack(states)
        next_states = states.clone()
        states = preprocess_replay(states)
        next_states = preprocess_replay(next_states)
        
        rewards = torch.stack(rewards)
        action = torch.stack(action)
        c_flag = torch.stack(c_flag)
        used_quant = torch.stack(used_quant)
        is_ws = is_ws.cuda()
            
        
        return states, next_states, rewards, action, c_flag, used_quant, idxs, is_ws

    
    def prepare_vq(self, model, n=8):
        with torch.no_grad():
            vq_db = deque([],maxlen=self.capacity)
            
            for idx in range(self.n-n-1):
                batch = Transition(*zip(*list([self.memory[int(i+idx)] for i in range(n)])))
    
                state, _ = model.encode(batch.state[0].squeeze()[None,None,:].cuda())
                state = state.cpu().squeeze()
                
                
                rewards = torch.stack(batch.reward)
                action = torch.stack(batch.action)
                c_flag = (~torch.stack(batch.c_flag)).float()
    
                returns = rewards*(c_flag.cumprod(0))
                if returns.sum()>0:
                    vq_db.append(VQ_State(state, returns.squeeze(), action))
    
            
            if len(vq_db)>self.cluster_size:
                new_db = deque([],maxlen=self.capacity)
                
                db = VQ_State(*zip(*list(vq_db)))
                states = torch.stack(db.state)
                
                _, sampled_indices = get_highest_l1_vectors(states, self.cluster_size)
    
                for idx in sampled_indices:
                    new_db.append(VQ_State(db.state[idx], db.returns[idx], db.action[idx]))
                vq_db = new_db
                
            #print(f"len vq db: {len(vq_db)}")
            return vq_db
        
    
    def set_priority(self, idxs, priority, same_traj):
        for i, idx in enumerate(idxs):
            if same_traj[i]==1:
                self.priority[idx] = torch.max(_eps, priority[i].detach())
    

    def max_priority(self):
        return self.priority.max()
        
    def reset_priorities(self):
        max_priority = min(1,self.max_priority())
        for i in range(self.n):
            self.priority[i] = max_priority
    
    def free(self):
        self.memory = None
        self.memory = deque([],maxlen=self.capacity)

        self.priority = None
        self.priority = torch.tensor([0]*(self.capacity), dtype=torch.float)
        self.priority[0] = 1 # For getting initial max priority
        
        self.n=0
    

    def reset_last_k(self, current_step, k=30000):
        # When getting cuda out of memory
        
        self.memory = deque(list(self.memory)[current_step-k:current_step])
        self.priority = self.priority[current_step-k:current_step]
        self.priority = torch.cat((self.priority, torch.zeros(self.capacity-k, dtype=torch.float)))


    def __len__(self):
        return len(self.memory)