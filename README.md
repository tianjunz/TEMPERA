# TEMPERA: Test-Time Prompting via Reinforcement Learning

## Intro

This is an implementation of the method proposed in 

<a href="https://arxiv.org/pdf/2211.11890.pdf">TEMPERA: Test-Time Prompting via Reinforcement Learning</a>

## Citation
If you use this code in your own work, please cite our paper:
```
@article{zhang2022tempera,
  title={TEMPERA: Test-Time Prompting via Reinforcement Learning},
  author={Zhang, Tianjun and Wang, Xuezhi and Zhou, Denny and Schuurmans, Dale and Gonzalez, Joseph E},
  journal={arXiv preprint arXiv:2211.11890},
  year={2022}
}
```

## Installation

```
# Install Instructions
conda create -n ride python=3.8
conda activate tempera
```

## Train Tempera on GLUE and SuperGLUE benchmarks
```
python main.py --env-name "lmnoprefix" --algo ppo --use-gae --log-interval 1 --num-steps 32 --num-processes 64 --lr 6e-4 --entropy-coef 1e-2 --value-loss-coef 0.5 --num-mini-batch 32 --gamma 0.999 --gae-lambda 0.95 --num-env-steps 3000000 --use-proper-time-limits --eval-interval 10 --num_shots 4 --models roberta-large --datasets $1 --subsample_test_set 872 --max_steps 8 --approx --verbalizer --use_attention --sub_sample --rew_type step --num_actors 8 --example_pool_size 16 --seed $2 --random_init 0 --use-linear-lr-decay --normalize_obs --exploration
```

## Acknowledgements
Our vanilla RL algorithm is based on [PPO](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail) and [Calibrate before Use](https://github.com/tonyzhaozh/few-shot-learning).

## License
This code is under the CC-BY-NC 4.0 (Attribution-NonCommercial 4.0 International) license.

