# policies/bfs_policy.py
from collections import deque

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4


def bfs_policy(env, obs, info, step_index: int) -> int:
    start = obs["pacman"]
    targets = set(obs["pellets"]) | set(obs["power_pellets"])

    if not targets:
        return STAY

    # Avoid ghost cells unless scared
    scared = obs["scared_timer"] > 0
    blocked = set()
    if not scared:
        blocked = {g["pos"] for g in obs["ghosts"] if g["active"]}

    queue = deque([start])
    parent = {start: None}
    action_from_parent = {}

    while queue:
        current = queue.popleft()

        if current in targets and current != start:
            # reconstruct first action
            while parent[current] != start and parent[current] is not None:
                current = parent[current]
            return action_from_parent[current]

        for action in env.legal_actions(current):
            nxt = env._move(current, action)
            if nxt in parent:
                continue
            if nxt in blocked:
                continue

            parent[nxt] = current
            action_from_parent[nxt] = action if current == start else action_from_parent[current]
            queue.append(nxt)

    # fallback: if BFS cannot find safe route, choose any legal move
    legal = env.legal_actions(start)
    return legal[0] if legal else STAY