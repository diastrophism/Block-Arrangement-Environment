### PPO block_arrangement Config ###

env = {
    "name": "block_arrangement",
    "stock_scale": 3,
    "utilization": 70,
    "width": 6,
    "height": 5,
    "block_size" : 1,
    "num_blocks": 30,
    "provided_ratio": 1,
    "entrance_cnt": 1,
    "entrance_position": [5,0],
    "rearrangement_occur_reward": -5,
    "arrangement_fail_reward": -1,
    "schedule_clear_reward_without_rearrangement": +1
}

agent = {
    "name": "ppo",
    "network": "discrete_policy_value",
    "head": "mlp",
    "gamma": 0.99,
    "batch_size": 32,
    "n_step": 128,
    "n_epoch": 3,
    "_lambda": 0.95,
    "epsilon_clip": 0.01,
    "vf_coef": 1.0,
    "ent_coef": 0.01,
    "clip_grad_norm": 1.0,
    "use_standardization": True,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 2.5e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 2000,
    "save_period": 2000,
    "eval_iteration": 10,
    "record": False,
    "record_period": 2000,
    #distributed setting
    "distributed_batch_size": 256,
    "update_period": agent["n_step"],
    "num_workers": 16,
}
