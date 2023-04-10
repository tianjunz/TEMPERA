import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs, make_vec_envs_eval, make_vec_envs_fseval, get_num_test
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate, evaluate_lm, evaluate_fs_lm
from utils import setup_roberta
from torch import multiprocessing as mp

class Normalizer:
    _STATS_FNAME = "env_stats.pickle"

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, in_size, num_process, device='cpu', dtype=torch.float):
        device='cpu'
        self.mean = torch.zeros((num_process, in_size), device=device, dtype=dtype)
        self.std = torch.ones((num_process, in_size), device=device, dtype=dtype)
        self.num_process = num_process
        self.eps = 1e-12 if dtype == torch.double else 1e-5
        self.device = device
        self.count = self.eps + torch.zeros((num_process, in_size), device=device, dtype=dtype)

    def update_stats(self, batch_data, batch_indices):
        if isinstance(batch_data, np.ndarray):
            batch_data = torch.from_numpy(batch_data).float().to(data.device)
        batch_data = batch_data.to('cpu')
        if isinstance(batch_indices, np.ndarray):
            batch_indices = torch.from_numpy(batch_indices).to('cpu')
        for i in range(self.num_process):
            index = (batch_indices == i).nonzero()
            data = torch.gather(batch_data, dim=0, index=index)
            if data.shape[0] > 1:
                batch_mean = data.mean(0, keepdim=True)
                batch_var = data.var(0, keepdim=True)
                batch_count = data.shape[0]
                self.update_from_moments(batch_mean, batch_var, batch_count, i)

    def update_from_moments(self, batch_mean, batch_var, batch_count, index):
        delta = batch_mean - self.mean[[index]]
        tot_count = self.count[[index]] + batch_count

        new_mean = self.mean[[index]] + delta * batch_count / tot_count
        m_a = torch.square(self.std[[index]]) * (self.count[[index]])
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count[[index]] * batch_count / (self.count[[index]] + batch_count)
        new_var = M2 / (self.count[[index]] + batch_count)

        new_count = batch_count + self.count[[index]]

        self.mean[[index]] = new_mean
        self.std[[index]] = torch.sqrt(new_var)
        self.count[[index]] = new_count

    def normalize(self, val, index):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        std = torch.clamp(self.std, self.eps)
        mean = self.mean[index]
        std = std[index]
        return (val - mean.to(val.device)) / std.to(val.device)

    def denormalize(self, val):
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        std = torch.clamp(self.std, self.eps)
        return std * val.to(val.device) + self.mean.to(val.device)
 
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def main():
    ctx = mp.get_context('spawn')
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    
    # Construct params
    params = {
        'conditioned_on_correct_classes': True,
        'api_num_log_prob': args.api_num_log_prob,
        'approx': args.approx,
        'bs': 1
    }
    params['model'] = args.models
    params['dataset'] = args.datasets
    params['seed'] = args.seed
    params['num_shots'] = args.num_shots
    params['expr_name'] = ""
    params['subsample_test_set'] = args.subsample_test_set
    params['env_name'] = args.env_name
    params['verbalizer'] = args.verbalizer
    params['rew_type'] = args.rew_type
    params['example_pool_size'] = args.example_pool_size
    params['use_knn'] = args.use_knn
    params['sub_sample'] = args.sub_sample
    params['num_actors'] = args.num_actors
    params['entropy_coef'] = args.env_entropy_coef
    params['random_init'] = args.random_init
    if args.models == 'gpt2-xl':
        obs_size = 1600
    elif args.models == 'gpt2-large':
        obs_size = 1280
    elif args.models == 'gpt2-medium':
        obs_size = 1024
    elif args.models == 'roberta-large':
        obs_size = 1024
    elif args.models == 't5-large':
        obs_size = 1024
    elif args.models == 't5-11b':
        obs_size = 1024
    elif args.models == 't5-3b':
        obs_size = 1024
    else:
        assert False
    print('Experiment params ', params, flush=True)
    print('Experiment arguments ', args, flush=True)

    envs = make_vec_envs(params['seed'], params, args.max_steps, args.num_processes, args.gamma, obs_size, 0)
    envs_fseval = make_vec_envs_fseval(params['seed'], params, args.max_steps, 16, args.gamma, obs_size, 0)
    num_test_samples = get_num_test(params['seed'], params, args.max_steps, args.num_processes, args.gamma, obs_size, 0, 0%torch.cuda.device_count())
    eval_envs = []
    for i in range(params['num_actors']):
        eval_env, num_test_samples = make_vec_envs_eval(params['seed'], params, args.max_steps, args.num_processes, args.gamma, obs_size, False, i, i%torch.cuda.device_count())
        eval_envs.append(eval_env)

    num_blocks = int(envs.observation_space.shape[0]/obs_size)
    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        args.use_attention,
        'cuda',
        num_blocks,
        base_kwargs={'recurrent': args.recurrent_policy,
            'hidden_size': 1024})
    actor_critic.to(device)

    hidden_dim = 256
    if args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    else:
        assert False

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                            envs.observation_space.shape, envs.action_space,
                            actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    all_obs = []
    all_rews = []
    all_indexs = []
    if args.normalize_rew:
        rew_normalizer = Normalizer(1, 16 * len(params['label_dict'].keys()))
    else:
        rew_normalizer = None
    for j in range(num_updates):

        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Observation reward and next obs
            subset_idxs = envs.venv.envs[0].subset_idxs
            all_indexs.append(copy.deepcopy(subset_idxs))
            obs, reward, done, infos = envs.step(action)
            if args.normalize_obs:
                actor_critic.base.normalizer.update_stats(obs)
            all_rews.append(copy.deepcopy(reward))
            all_obs.append(copy.deepcopy(obs))
        
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
            if done[0]:
                episode_rewards.append(info['episode_r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, torch.Tensor(subset_idxs).unsqueeze(-1), value, reward, masks, bad_masks)

        if args.normalize_obs:
            all_obs = []
        if args.normalize_rew:
            rew_normalizer.update_stats(torch.cat(all_rews, dim=0), torch.from_numpy(np.concatenate(all_indexs, axis=0)))
        if args.normalize_rew:
            rollouts.update_rew(rew_normalizer)
        if args.normalize_rew:
            all_indexs = []
            all_rews = []

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                args.gae_lambda, rew_normalizer, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss), flush=True)

        if (args.eval_interval is not None and j % args.eval_interval == 0):

            # Validation set
            evaluate_fs_lm(actor_critic, None, envs_fseval, args.seed, 
                    16, 16, params, args, obs_size)

            actor_critic.to('cpu')
            results = ctx.Queue()
            orig_results = ctx.Queue()
            tp = ctx.Queue()
            fp = ctx.Queue()
            fn = ctx.Queue()
            env_queue = ctx.Queue()
            evaluate_processes = []
            for i in range(params['num_actors']):
                eval = ctx.Process(
                    target=evaluate_lm,
                    args=(i, actor_critic, None, eval_envs[i], args.seed, 
                    args.num_processes, num_test_samples, orig_results, results, env_queue, params, args, obs_size, tp, fp, fn))
                eval.start()
                evaluate_processes.append(eval)
            for eval in evaluate_processes:
                eval.join()
            results_list = []
            orig_results_list = []
            tp_list = []
            fp_list = []
            fn_list = []
            for i in range(results.qsize()):
                results_list.append(results.get())
                orig_results_list.append(orig_results.get())
                tp_list.append(tp.get())
                fp_list.append(fp.get())
                fn_list.append(fn.get())
            print('Evaluation mean reward {:.5f}, original mean reward {:.5f}'.format(sum(results_list)/len(results_list), sum(orig_results_list)/len(orig_results_list)), flush=True)
            precision = sum(tp_list) / (sum(tp_list) + sum(fp_list))
            recall = sum(tp_list) / (sum(tp_list) + sum(fn_list))
            f_score = 2 * precision * recall / (precision + recall)
            print('Evaluation mean reward f score {:.5f}'.format(f_score), flush=True)
            actor_critic.to('cuda:0')

            if not args.load_ckpt and args.env_name != 'lmall': 
                file_path = 'checkpoints/'+str(args.models)+'_'+str(args.datasets)+'_'+str(args.seed)+'/'
                isExist = os.path.exists(file_path)
                if not isExist:
                    os.makedirs(file_path)
                current_prompt_embedding_pool = []
                add_current_prompt_embedding_pool = []
                current_verbalizer_embedding_pool = []
                add_current_verbalizer_embedding_pool = []
                for eval_env in eval_envs:
                    current_prompt_embedding_pool.append(eval_env.envs[0].current_prompt_embedding_pool)
                    add_current_prompt_embedding_pool.append(eval_env.envs[0].add_current_prompt_embedding_pool)
                    current_verbalizer_embedding_pool.append(eval_env.envs[0].current_verbalizer_embedding_pool)
                    add_current_verbalizer_embedding_pool.append(eval_env.envs[0].add_current_verbalizer_embedding_pool)
                current_prompt_embedding_pool = torch.cat(current_prompt_embedding_pool, dim=0)
                add_current_prompt_embedding_pool = torch.cat(add_current_prompt_embedding_pool, dim=0)
                current_verbalizer_embedding_pool = torch.cat(current_verbalizer_embedding_pool, dim=0)
                add_current_verbalizer_embedding_pool = torch.cat(add_current_verbalizer_embedding_pool, dim=0)
                torch.save(current_prompt_embedding_pool, file_path+'current_prompt_embedding_pool.pth')
                torch.save(add_current_prompt_embedding_pool, file_path+'add_current_prompt_embedding_pool.pth')
                torch.save(current_verbalizer_embedding_pool, file_path+'current_verbalizer_embedding_pool.pth')
                torch.save(add_current_verbalizer_embedding_pool, file_path+'add_current_verbalizer_embedding_pool.pth')


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()
