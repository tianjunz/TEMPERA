import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.utils import AddBias, init

"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions).sum(-1, keepdim=True)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return self.mean


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        # print(torch.nn.Softmax(dim=-1)(x))
        return FixedCategorical(logits=x)

class FixedMultiCategorical(torch.distributions.Distribution):
    def __init__(self, logits):
        super().__init__(validate_args=False)
        self._dists = [
            FixedCategorical(logits=logit) for logit in logits
        ]

    def log_probs(self, actions):
        return torch.stack(
            [dist.log_prob(action)
                for dist, action in zip(self._dists, torch.unbind(actions, dim=1))],
            dim=1,
        ).sum(dim=1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self._dists], dim=1).sum(dim=1)

    def sample(self, sample_shape=torch.Size()):
        assert sample_shape == torch.Size()
        return torch.stack([dist.sample() for dist in self._dists], dim=1)

    def mode(self):
        return torch.stack(
            [torch.argmax(dist.probs, dim=1) for dist in self._dists], dim=1
        )

    def random_actions(self):
        return torch.stack([dist.random_actions() for dist in self._dists], dim=-1)

class MultiCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(MultiCategorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.linear = nn.ModuleList([init_(nn.Linear(num_inputs, _num_outputs.n)) for _num_outputs in num_outputs])

    def forward(self, x):
        x = [linear(x) for linear in self.linear]
        return FixedMultiCategorical(logits=x)

class AutoFixedMultiCategorical(torch.distributions.Distribution):
    def __init__(self, logits, mlp, num_classes):
        super().__init__(validate_args=False)
        self._logits = logits
        self._mlp = mlp
        self._num_classes = num_classes

    def log_probs(self, actions):
        outputs = []
        inputs = self._logits
        self._dists = []
        for mlp, action, num_class in zip(self._mlp, torch.unbind(actions, dim=1), self._num_classes):
            output = mlp(inputs)
            dist = FixedCategorical(logits=output)
            self._dists.append(dist)
            outputs.append(dist.log_prob(action))
            inputs = torch.cat([inputs, F.one_hot(action, num_class)], dim=-1)
        return torch.stack(outputs, dim=1).sum(dim=1)

    def entropy(self):
        return torch.stack([dist.entropy() for dist in self._dists], dim=1).sum(dim=1)

    def sample(self, sample_shape=torch.Size()):
        assert sample_shape == torch.Size()
        outputs = []
        inputs = self._logits
        self._dists = []
        for mlp, num_class in zip(self._mlp, self._num_classes):
            output = mlp(inputs)
            dist = FixedCategorical(logits=output)
            self._dists.append(dist)
            action = dist.sample()
            outputs.append(action)
            inputs = torch.cat([inputs, F.one_hot(action.squeeze(-1), num_class)], dim=-1)
        return torch.stack(outputs, dim=1)

    def mode(self):
        outputs = []
        inputs = self._logits
        self._dists = []
        for mlp, num_class in zip(self._mlp, self._num_classes):
            output = mlp(inputs)
            dist = FixedCategorical(logits=output)
            self._dists.append(dist)
            action = torch.argmax(dist.probs, dim=1)
            outputs.append(action)
            inputs = torch.cat([inputs, F.one_hot(action, num_class)], dim=-1)
        return torch.stack(outputs, dim=1)

    def random_actions(self):
        return torch.stack([dist.random_actions() for dist in self._dists], dim=-1)

class AutoMultiCategorical(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(AutoMultiCategorical, self).__init__()

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            gain=0.01)

        self.mlp = nn.ModuleList([])
        self.num_classes = []
        total_num_outputs = 0
        for _num_outputs in num_outputs:
            self.mlp.append(nn.Sequential(init_(nn.Linear(num_inputs + total_num_outputs, 256)), nn.ReLU(), init_(nn.Linear(256, _num_outputs.n))))
            total_num_outputs += _num_outputs.n
            self.num_classes.append(_num_outputs.n)
        # self.linear = nn.ModuleList([init_(nn.Linear(num_inputs, _num_outputs.n)) for _num_outputs in num_outputs])

    def forward(self, x):
        return AutoFixedMultiCategorical(logits=x, mlp=self.mlp, num_classes=self.num_classes)

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
