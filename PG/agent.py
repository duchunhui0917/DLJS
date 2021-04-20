import torch
from torch import optim
import torch.nn.functional as F
from PG.brain import Net
import numpy as np
import copy


class Agent:
    def __init__(self, n_input, n_output, lr=0.001, gamma=1):
        self.actor = Net(n_input=n_input, n_output=n_output)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.gamma = gamma
        self.rewards = []

    def estimate_return(self, all_rewards):
        # compute qs
        all_qs = []
        for rewards in all_rewards:
            qs = []
            q = 0
            for reward in rewards[::-1]:
                q = q * self.gamma + reward
                qs.insert(0, q)
            all_qs.append(qs)
        # compute advantages
        adv_n = copy.deepcopy(all_qs)
        max_length = max([len(adv) for adv in adv_n])
        for adv in adv_n:
            while len(adv) < max_length:
                adv.append(0)
        adv_n = np.array(adv_n)
        adv_n = adv_n - adv_n.mean(axis=0)

        adv_n__ = []
        for i in range(adv_n.shape[0]):
            original_length = len(all_qs[i])
            adv_n__.append(list(adv_n[i][:original_length]))
        adv_n = adv_n__
        # normalization
        adv_s = []
        for advantages in adv_n:
            for advantage in advantages:
                adv_s.append(advantage)
        adv_s = np.array(adv_s)
        mean = adv_s.mean()
        std = adv_s.std()
        adv_n__ = []
        for advantages in adv_n:
            advantages__ = []
            for advantage in advantages:
                advantages__.append((advantage - mean) / (std + np.finfo(np.float32).eps))
            adv_n__.append(advantages__)
        adv_n = adv_n__
        return all_qs, adv_n

    def compute_loss(self, state, action, advantage):
        logits = self.actor(state)
        logprob = - F.cross_entropy(logits, torch.tensor(action))
        return logprob * advantage

    def update_parameters(self, all_states, all_actions, all_advantages):
        for states, actions, advantages in zip(all_states, all_actions, all_advantages):
            all_loss = []
            for state, action, advantage in zip(states, actions, advantages):
                if state is None or action is None:
                    continue
                loss = - self.compute_loss(state, [action], advantage)
                all_loss.append(loss.unsqueeze(0))
                if len(all_loss) == 1000:
                    self.optimizer.zero_grad()
                    l = torch.cat(all_loss).mean()
                    l.backward()
                    self.optimizer.step()
                    all_loss = []
            if len(all_loss) > 0:
                self.optimizer.zero_grad()
                l = torch.cat(all_loss).mean()
                l.backward()
                self.optimizer.step()
        # print('advantage:', sum(advantages_list) / len(advantages_list))
        # print('log_probs:', sum(log_probs_list) / len(log_probs_list))
        # print('loss:', sum(loss_list) / len(loss_list))
