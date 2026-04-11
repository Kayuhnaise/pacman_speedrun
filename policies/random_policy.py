# policies/random_policy.py

def random_policy(env, obs, info, step_index: int) -> int:
    legal = env.legal_actions(obs["pacman"])
    return env.rng.choice(legal)