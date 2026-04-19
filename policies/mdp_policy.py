UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

GAMMA = 0.5
REASONING_ITERATIONS = 30

def reward_state(
    pos: tuple[int, int],
    pellets: set[tuple[int, int]],
    power_pellets: set[tuple[int, int]],
    active_ghosts: set[tuple[int, int]],
    scared: bool,
    step_index: int,
) -> float:
    
    # Penalize empty spaces
    reward = -25.0

    # Encourage eating pellets
    if pos in pellets:
        reward += 25.0

    # Power pellets are icing on the top for PacMan. We dont want to overly reward 
    # for them since our goal is to finish in as little steps as possible.
    # Also negate this reward initially and towards an expected end (>200 steps)
    if pos in power_pellets:
        if 25 <= step_index <= 200:
            reward += 25.0

    # Ghosts
    if pos in active_ghosts:
        # Again towards the "end" we dont really care about ghosts. We want to eat all the pellets
        if scared and step_index < 175:
            reward += 100.0
        else:
            reward = -200.0

    return reward


def transition_state(action: int) -> tuple:
    # Since the board is a series of tunnels, 
    # we generally want to continue going in the same direction
    if action == UP or action == DOWN:
        return ((action, 0.8), (LEFT, 0.1), (RIGHT, 0.1))
    if action == LEFT or action == RIGHT:
        return ((action, 0.8), (UP, 0.1), (DOWN, 0.1))
    return ((STAY, 1.0),)


def belief_states(
    env,
    pellets: set[tuple[int, int]],
    power_pellets: set[tuple[int, int]],
    active_ghosts: set[tuple[int, int]],
    scared: bool,
    step_index: int,
) -> dict[tuple[int, int], float]:
    """
    Build belief/value estimates for traversable states.
    """
    states = []
    for r in range(env.rows):
        for c in range(env.cols):
            pos = (r, c)
            if not env.is_wall(pos):
                states.append(pos)

    values = {_: 0.0 for _ in states}

    for _ in range(REASONING_ITERATIONS):
        new_values: dict[tuple[int, int], float] = {}

        for state in states:
            actions = env.legal_actions(state)
            if not actions:
                new_values[state] = reward_state(
                    state, pellets, power_pellets, active_ghosts, scared, step_index
                )
                continue

            best_q = float("-inf")
            for action in actions:
                q_val = 0.0
                for cand_action, prob in transition_state(action):
                    next_state = env._move(state, cand_action)
                    reward = reward_state(
                        next_state, pellets, power_pellets, active_ghosts, scared, step_index
                    )
                    q_val += prob * (reward + GAMMA * values[next_state])

                if q_val > best_q:
                    best_q = q_val

            new_values[state] = best_q

        values = new_values

    return values


def mdp_policy(env, obs, info, step_index: int) -> int:
    """
    MDP policy to determine next moves based on rewards and possible moves.
    """
    start = obs["pacman"]
    legal_actions = env.legal_actions(start)
    if not legal_actions:
        return STAY

    pellets = set(obs["pellets"])
    power_pellets = set(obs["power_pellets"])
    scared = obs["scared_timer"] > 0
    active_ghosts = {g["pos"] for g in obs["ghosts"] if g["active"]}

    # Build belief and value estimates for traversable states.
    values = belief_states(
        env, pellets, power_pellets, active_ghosts, scared, step_index
    )

    # Act greedily according to one-step lookahead from current position.
    best_action = legal_actions[0]
    best_q_value = float("-inf")

    for action in legal_actions:
        q_val = 0.0
        for cand_action, prob in transition_state(action):
            next_state = env._move(start, cand_action)
            reward = reward_state(next_state, pellets, power_pellets, active_ghosts, scared, step_index)
            q_val += prob * (reward + GAMMA * values[next_state])

        if q_val > best_q_value:
            best_q_value = q_val
            best_action = action

    return best_action
