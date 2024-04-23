import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

import numpy as np

import time

from collections import namedtuple, deque
from itertools import count



from nosaveddata.nsd_utils.save_hypers import Hypers
from nosaveddata.builders.efficientzero import *

from nosaveddata.nsd_utils.bbf import network_ema



import math
from ray.util.queue import Queue
import ray






Transition = namedtuple('Transition',
                        ('state', 'reward', 'action', 'c_flag'))

eps=1e-10
_eps = torch.tensor(eps)#, device='cuda')



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




# https://github.com/YeWR/EfficientZero/blob/main/core/storage.py
class QueueStorage(object):
    def __init__(self, threshold=15, size=20):
        """Queue storage
        Parameters
        ----------
        threshold: int
            if the current size if larger than threshold, the data won't be collected
        size: int
            the size of the queue
        """
        self.threshold = threshold
        self.queue = Queue(maxsize=size)

    def push(self, batch):
        if self.queue.qsize() <= self.threshold:
            self.queue.put(batch)

    def pop(self):
        if self.queue.qsize() > 0:
            return self.queue.get()
        else:
            return None

    def __len__(self):
        return self.queue.qsize()



@ray.remote
class PrioritizedReplay_nSteps_Sqrt(object):
    def __init__(self, capacity, final_beta=1.0, initial_beta=0.4, total_steps=40000, alpha=0.5, beta=0.5):
        self.capacity = capacity
        self.memory = deque([],maxlen=capacity)
        self.total_steps=total_steps

        self.alpha=alpha
        self.beta=beta
        
    
        self.free()
        

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        self.n+=1
        self.priority[self.n] = self.max_priority()
    

    def prioritize(self, batch_size, seq_len, step):
        with torch.no_grad():
            max_n = int(self.n -seq_len -5 -1)

            priority = self.priority.clone()[:max_n].pow(self.alpha)
            probs = priority/priority.sum()
            _, sorted_priorities = priority.sort()

            segment_length=(max_n//batch_size)
            idxs = torch.randint(0, segment_length, (batch_size,))
            idx = idxs + torch.arange(batch_size)*segment_length
            idx = sorted_priorities[idx]

            is_w = (1/(probs*max_n+eps)).pow(self.beta)
            is_w/=is_w.max()
            
            return idx, is_w[idx]

    
    def sample(self, seq_len, batch_size, step):
        
        states, action, rewards, c_flag, idxs = [], [], [], [], []
        
        idxs, is_ws = self.prioritize(batch_size, seq_len, step)
        for idx in idxs:
            batch = Transition(*zip(*list([self.memory[int(i+idx)] for i in range(max(seq_len+1,5+1))])))
            
            states.append(torch.stack(batch.state).squeeze(1))
            
            rewards.append(torch.stack(batch.reward))
            action.append(torch.stack(batch.action))
            c_flag.append((~torch.stack(batch.c_flag)).float())
        

        states = torch.stack(states)
        next_states = states.clone()
        states = preprocess_replay(states)
        next_states = preprocess_replay(next_states)
        
        rewards = torch.stack(rewards)
        action = torch.stack(action)
        c_flag = torch.stack(c_flag)
        is_ws = is_ws
            
        
        return states, next_states, rewards, action, c_flag, idxs, is_ws
        
    
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

    def getlen(self):
        return len(self.memory)



@ray.remote#(num_gpus=0.125)
class ParallelBuffer(Hypers):
    def __init__(self, buffer, storage, batch_size):
        super().__init__()
        

    def run(self):
        #print(f"STARTED WORKER RUN")
        
        while True:
            
            if len(self.storage) < 4 and ray.get(self.buffer.getlen.remote()) > 1999:
                try:
                    
                    #print(f"SAMPLING DATA")
                    
                    batch = ray.get(self.buffer.sample.remote(5, self.batch_size, 0))
                    
                    #print(f"Sampled: {batch[0].shape, batch[1].shape, batch[2].shape}")
                    #print(f"Sampled: {batch[0].device}")
                    
                    self.storage.push(batch) # REQUIRES DATA STORAGE ON CPU 
                    
                except Exception as e:
                    print('Data is Deleted...')
                    print(f"{e}")
                    time.sleep(0.1)


@ray.remote(num_gpus=0.125)
class MCTS_Buffer(Hypers):
    def __init__(self, buffer, buffer_storage, storage, mcts, n_actions, model):
        super().__init__()
        
        #self.model=EfficientZero(n_actions).cuda()

    #def load_weights(self, model):
    #    self.model.load_state_dict(model.state_dict())
        
    def run(self):
        
        while True:
            
            if len(self.buffer_storage) > 0 and len(self.storage) < 4:
                try:
                    
                    #print(f"SAMPLING DATA")
                    
                    batch = self.buffer_storage.pop()
                    states, next_states, rewards, action, c_flag, idxs, is_ws = batch
                    
                    value_mcts, improved_policy, _ = self.mcts(self.model, states[:,0][:,None].cuda())
                    
                    print(f"WEIGHTS {self.model.ac.policy[2].mlp[3].weight.sum()}")

                    batch = states, next_states, rewards, action, c_flag, idxs, is_ws, value_mcts.cpu(), improved_policy.cpu()
                    
                    self.storage.push(batch) 
                    
                except Exception as e:
                    print('GPU Data is Deleted...')
                    print(f"{e}")
                    time.sleep(0.1)

