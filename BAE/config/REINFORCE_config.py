### REINFORCE block_arrangement Config ###

env = {
    "name": "block_arrangement",
    "action_type": "discrete",
    "stock_scale": 3,
    "utilization": 70,
    "width": 6,
    "height": 5,
    "block_size" : 1,
    "num_blocks": 20,
    "provided_ratio": 1,
    "entrance_cnt": 1,
    "entrance_position": [5,0],
    "rearrangement_occur_reward": -5,
    "arrangement_fail_reward": -1,
    "schedule_clear_reward_without_rearrangement": +1
}

agent = {
    "name": "reinforce",
    "network": "discrete_policy",
    "gamma": 0.99,
    "lr_decay": True,
}

optim = {
    "name": "adam",
    "lr": 1e-4,
}

train = {
    "training": True,
    "load_path": None,
    "run_step": 100000,
    "print_period": 2000,
    "save_period": 2000,
    "eval_iteration": 10,
    "record": True,
    "record_period": 2000
}
