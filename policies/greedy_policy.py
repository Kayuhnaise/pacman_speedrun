# policies/greedy_policy.py

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def greedy_policy(env, obs, info, step_index: int) -> int:
    pac = obs["pacman"]
    legal = env.legal_actions(pac)

    targets = list(obs["pellets"]) + list(obs["power_pellets"])
    active_ghosts = [g["pos"] for g in obs["ghosts"] if g["active"]]

    if not legal:
        return 4  # STAY

    if not targets:
        return legal[0]

    best_action = legal[0]
    best_score = float("-inf")

    scared = obs["scared_timer"] > 0

    for action in legal:
        nxt = env._move(pac, action)

        nearest_pellet_dist = min(manhattan(nxt, t) for t in targets) if targets else 0
        nearest_ghost_dist = min(manhattan(nxt, g) for g in active_ghosts) if active_ghosts else 99

        # Heuristic:
        # - prefer moving toward pellets
        # - avoid ghosts unless scared mode is active
        score = -nearest_pellet_dist

        if scared:
            score += 0.5 * (10 - nearest_ghost_dist)
        else:
            score += 1.5 * nearest_ghost_dist

        if score > best_score:
            best_score = score
            best_action = action

    return best_action