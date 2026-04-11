# policies/smart_bfs_policy.py
from collections import deque

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def smart_bfs_policy(env, obs, info, step_index):
    start = obs["pacman"]
    pellets = set(obs["pellets"])
    power = set(obs["power_pellets"])
    ghosts = [g["pos"] for g in obs["ghosts"] if g["active"]]

    if not pellets and not power:
        return STAY

    targets = pellets | power
    scared = obs["scared_timer"] > 0

    queue = deque([(start, 0)])  # (position, cost)
    parent = {start: None}
    action_map = {}

    best_target = None
    best_score = float("inf")

    while queue:
        current, dist = queue.popleft()

        if current in targets and current != start:
            # compute risk score
            ghost_penalty = 0
            if not scared and ghosts:
                ghost_penalty = min(manhattan(current, g) for g in ghosts)

            score = dist - 2 * ghost_penalty

            if score < best_score:
                best_score = score
                best_target = current

        for action in env.legal_actions(current):
            nxt = env._move(current, action)
            if nxt in parent:
                continue

            parent[nxt] = current
            action_map[nxt] = action if current == start else action_map[current]
            queue.append((nxt, dist + 1))

    if best_target:
        return action_map[best_target]

    # fallback
    legal = env.legal_actions(start)
    return legal[0] if legal else STAY