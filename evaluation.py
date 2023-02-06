import numpy as np
import torch
import copy

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs, make_vec_envs_eval


def evaluate(actor_critic, obs_rms, env_name, seed, num_processes, eval_log_dir,
             device):
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    while len(eval_episode_rewards) < 10:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, _, done, infos = eval_envs.step(action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))

def evaluate_lm(i, actor_critic, obs_rms, eval_envs, seed, num_processes,
             num_test, orig_results, results, env_queue, params, args, obs_size, tp, fp, fn):
    # vec_norm = utils.get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.obs_rms = obs_rms
    print('Evaluation ', i, ' started.', flush=True)
    if args.load_ckpt: 
        file_path = 'checkpoints/'+str(args.models)+'_'+str(args.datasets)+'_'+str(args.seed)+'/'
        eval_envs.envs[0].load_ckpt(file_path, i, num_test)
    else:
        if eval_envs.envs[0].embedding_prepared == False:
            # eval_envs, _ = make_vec_envs_eval(params['seed'], params, args.max_steps, args.num_processes, args.gamma, obs_size, False, i, i%torch.cuda.device_count())
            eval_envs.envs[0].prepare_embedding()
    # if eval_envs.envs[0].embedding_prepared == False:
    #     # eval_envs, _ = make_vec_envs_eval(params['seed'], params, args.max_steps, args.num_processes, args.gamma, obs_size, False, i, i%torch.cuda.device_count())
    #     eval_envs.envs[0].prepare_embedding()
    assert eval_envs.envs[0].embedding_prepared == True

    eval_episode_rewards = []

    obs = eval_envs.reset()
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size).cuda()
    # eval_masks = torch.zeros(num_processes, 1).cuda()

    total_correct = 0
    total_orig_correct = 0
    total_samples = 0
    _tp, _fp, _fn = 0, 0, 0
    for i in range(0, num_test, num_processes):
        if i + num_processes > num_test:
            idxs = np.arange(i, num_test)
        else:
            idxs = np.arange(i, i + num_processes)
        # TODO: auto this
        # eval_envs.venv.envs[0].idxs = idxs
        eval_envs.envs[0].idxs = idxs
        obs = eval_envs.reset()
        _done = False
        while not _done:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    # obs.cpu().cuda(),
                    torch.tensor(obs).float(),
                    # eval_recurrent_hidden_states.cpu().cuda(),
                    # eval_masks.cpu().cuda(),
                    None, 
                    None,
                    deterministic=True)
                    # deterministic=False)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action)
            _done = done[0]

            # eval_masks = torch.tensor(
            #     [[0.0] if done_ else [1.0] for done_ in done],
            #     dtype=torch.float32,
            #     device=device)
        total_correct += infos[0]['correct']
        total_orig_correct += infos[0]['orig_correct']
        total_samples += infos[0]['total']
        _tp += infos[0]['tp']
        _fp += infos[0]['fp']
        _fn += infos[0]['fn']
    # eval_envs.close()
    orig_results.put(total_orig_correct/total_samples)
    results.put(total_correct/total_samples)
    tp.put(_tp)
    fp.put(_fp)
    fn.put(_fn)

    print(" Evaluation using {} episodes: mean reward {:.5f}, original mean reward {:.5f}".format(
        len(eval_episode_rewards), total_correct/total_samples, total_orig_correct/total_samples), flush=True)

def evaluate_fs_lm(actor_critic, obs_rms, eval_envs, seed, num_processes,
             num_test, params, args, obs_size):
    # vec_norm = utils.get_vec_normalize(eval_envs)
    # if vec_norm is not None:
    #     vec_norm.eval()
    #     vec_norm.obs_rms = obs_rms
    print('Evaluation fs started.', flush=True)

    eval_episode_rewards = []
    obs = eval_envs.reset()
    # eval_recurrent_hidden_states = torch.zeros(
    #     num_processes, actor_critic.recurrent_hidden_state_size).cuda()
    # eval_masks = torch.zeros(num_processes, 1).cuda()

    total_correct = 0
    total_orig_correct = 0
    total_samples = 0
    for i in range(0, num_test, num_processes):
        if i + num_processes > num_test:
            idxs = np.arange(i, num_test)
        else:
            idxs = np.arange(i, i + num_processes)
        # TODO: auto this
        # eval_envs.venv.envs[0].idxs = idxs
        eval_envs.envs[0].idxs = idxs
        obs = eval_envs.reset()
        _done = False
        while not _done:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    # obs.cpu().cuda(),
                    torch.tensor(obs).float(),
                    # eval_recurrent_hidden_states.cpu().cuda(),
                    # eval_masks.cpu().cuda(),
                    None, 
                    None,
                    deterministic=True)
                    # deterministic=False)

            # Obser reward and next obs
            obs, _, done, infos = eval_envs.step(action)
            _done = done[0]

            # eval_masks = torch.tensor(
            #     [[0.0] if done_ else [1.0] for done_ in done],
            #     dtype=torch.float32,
            #     device=device)
        total_correct += infos[0]['correct']
        total_orig_correct += infos[0]['orig_correct']
        total_samples += infos[0]['total']
    # eval_envs.close()

    print(" Evaluation fs using: mean reward {:.5f}, original mean reward {:.5f}".format(total_correct/total_samples, total_orig_correct/total_samples), flush=True)