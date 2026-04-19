# env.py
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple
from PIL import Image, ImageDraw

Position = Tuple[int, int]

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
STAY = 4

ACTION_TO_DELTA = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
    STAY: (0, 0),
}

ACTION_NAMES = {
    UP: "UP",
    DOWN: "DOWN",
    LEFT: "LEFT",
    RIGHT: "RIGHT",
    STAY: "STAY",
}


@dataclass
class Ghost:
    pos: Position
    prev_pos: Position
    active: bool = True
    respawn_timer: int = 0


class PacmanEnv:
    """
    Custom Pac-Man grid environment for search/RL experiments.

    Rules implemented:
    - Pac-Man moves one cell per turn
    - Ghosts move one cell per turn
    - At intersections, ghosts choose a direction randomly
    - Pellets disappear when eaten
    - Power pellets activate scared mode for N turns
    - During scared mode, ghosts move every other turn
    - If Pac-Man collides with an active ghost during scared mode, the ghost is eaten
    - Eaten ghosts respawn after a delay
    - If Pac-Man collides with an active ghost outside scared mode,
      Pac-Man and active ghosts reset to start positions, pellets remain
    - After too many deaths, game ends
    - Additional ghosts spawn every K turns, up to max_ghosts
    """

    def __init__(
        self,
        board: List[str],
        max_steps: int = 500,
        max_deaths: int = 3,
        scared_duration: int = 20,
        ghost_respawn_steps: int = 10,
        ghost_spawn_interval: int = 10,
        max_ghosts: int = 4,
        seed: Optional[int] = None,
    ):
        self.original_board = board
        self.rows = len(board)
        self.cols = len(board[0])
        self.max_steps = max_steps
        self.max_deaths = max_deaths
        self.scared_duration = scared_duration
        self.ghost_respawn_steps = ghost_respawn_steps
        self.ghost_spawn_interval = ghost_spawn_interval
        self.max_ghosts = max_ghosts
        self.rng = random.Random(seed)

        self.wall_cells = set()
        self.tunnel_cells = set()
        self.initial_pellets = set()
        self.initial_power_pellets = set()

        self.pacman_start: Position = (1, 1)
        self.ghost_spawn: Position = (1, 1)

        self._parse_static_board()
        self.reset(seed=seed)

    def _parse_static_board(self):
        for r in range(self.rows):
            for c in range(self.cols):
                ch = self.original_board[r][c]
                pos = (r, c)

                if ch == "#":
                    self.wall_cells.add(pos)
                elif ch == "T":
                    self.tunnel_cells.add(pos)
                elif ch == ".":
                    self.initial_pellets.add(pos)
                elif ch == "o":
                    self.initial_power_pellets.add(pos)
                elif ch == "P":
                    self.pacman_start = pos
                elif ch == "G":
                    self.ghost_spawn = pos

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng.seed(seed)

        self.turn_count = 0
        self.deaths = 0
        self.done = False
        self.scared_timer = 0

        self.pacman_pos = self.pacman_start
        self.pellets = set(self.initial_pellets)
        self.power_pellets = set(self.initial_power_pellets)

        self.ghosts: List[Ghost] = [
            Ghost(pos=self.ghost_spawn, prev_pos=self.ghost_spawn, active=True, respawn_timer=0)
        ]

        return self._get_obs(), self._get_info()

    def _get_obs(self) -> Dict:
        return {
            "pacman": self.pacman_pos,
            "ghosts": [
                {
                    "pos": g.pos,
                    "prev_pos": g.prev_pos,
                    "active": g.active,
                    "respawn_timer": g.respawn_timer,
                }
                for g in self.ghosts
            ],
            "pellets": tuple(sorted(self.pellets)),
            "power_pellets": tuple(sorted(self.power_pellets)),
            "scared_timer": self.scared_timer,
            "turn": self.turn_count,
            "deaths": self.deaths,
        }

    def _get_info(self) -> Dict:
        return {
            "turn": self.turn_count,
            "deaths": self.deaths,
            "remaining_pellets": len(self.pellets) + len(self.power_pellets),
            "scared_timer": self.scared_timer,
        }

    def in_bounds(self, pos: Position) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, pos: Position) -> bool:
        return pos in self.wall_cells

    def _apply_tunnel(self, pos):
        if pos == (9, 0):
            return (9, self.cols - 1)
        if pos == (9, self.cols - 1):
            return (9, 0)
        return pos

    def _move(self, pos: Position, action: int) -> Position:
        dr, dc = ACTION_TO_DELTA[action]
        nxt = (pos[0] + dr, pos[1] + dc)

        if not self.in_bounds(nxt):
            return pos
        if self.is_wall(nxt):
            return pos

        nxt = self._apply_tunnel(nxt)
        return nxt

    def legal_actions(self, pos: Position) -> List[int]:
        actions = []
        for action in [UP, DOWN, LEFT, RIGHT]:
            nxt = self._move(pos, action)
            if nxt != pos:
                actions.append(action)
        if not actions:
            actions.append(STAY)
        return actions

    def is_intersection(self, pos: Position) -> bool:
        return len(self.legal_actions(pos)) >= 3

    def _spawn_new_ghost_if_needed(self):
        if (
            self.turn_count > 0
            and self.turn_count % self.ghost_spawn_interval == 0
            and len(self.ghosts) < self.max_ghosts
        ):
            self.ghosts.append(
                Ghost(pos=self.ghost_spawn, prev_pos=self.ghost_spawn, active=True, respawn_timer=0)
            )

    def _update_respawns(self):
        for ghost in self.ghosts:
            if not ghost.active:
                ghost.respawn_timer -= 1
                if ghost.respawn_timer <= 0:
                    ghost.active = True
                    ghost.pos = self.ghost_spawn
                    ghost.prev_pos = self.ghost_spawn

    def _reverse_action(self, prev_pos: Position, current_pos: Position) -> Optional[int]:
        dr = prev_pos[0] - current_pos[0]
        dc = prev_pos[1] - current_pos[1]
        for action, delta in ACTION_TO_DELTA.items():
            if delta == (dr, dc):
                return action
        return None

    def _move_ghosts(self):
        for ghost in self.ghosts:
            if not ghost.active:
                continue

            # scared ghosts move every other turn
            if self.scared_timer > 0 and self.turn_count % 2 == 1:
                continue

            legal = self.legal_actions(ghost.pos)

            if len(legal) == 1:
                chosen = legal[0]
            else:
                reverse_action = self._reverse_action(ghost.prev_pos, ghost.pos)

                # prefer continuing forward unless at an intersection
                if not self.is_intersection(ghost.pos):
                    forward_like = [a for a in legal if a != reverse_action]
                    if forward_like:
                        chosen = self.rng.choice(forward_like)
                    else:
                        chosen = self.rng.choice(legal)
                else:
                    options = [a for a in legal if a != reverse_action]
                    chosen = self.rng.choice(options if options else legal)

            old_pos = ghost.pos
            ghost.pos = self._move(ghost.pos, chosen)
            ghost.prev_pos = old_pos

    def _handle_pellet_collection(self) -> int:
        reward = 0

        if self.pacman_pos in self.pellets:
            self.pellets.remove(self.pacman_pos)
            reward += 10

        if self.pacman_pos in self.power_pellets:
            self.power_pellets.remove(self.pacman_pos)
            self.scared_timer = self.scared_duration
            reward += 50

        return reward

    def _reset_positions_after_death(self):
        self.pacman_pos = self.pacman_start
        for ghost in self.ghosts:
            if ghost.active:
                ghost.pos = self.ghost_spawn
                ghost.prev_pos = self.ghost_spawn

    def _handle_collisions(self) -> int:
        reward = 0

        for ghost in self.ghosts:
            if not ghost.active:
                continue

            if ghost.pos == self.pacman_pos:
                if self.scared_timer > 0:
                    ghost.active = False
                    ghost.respawn_timer = self.ghost_respawn_steps
                    reward += 200
                else:
                    self.deaths += 1
                    reward -= 200
                    self._reset_positions_after_death()
                    break

        return reward

    def step(self, action: int):
        if self.done:
            return self._get_obs(), 0.0, True, False, self._get_info()

        self.turn_count += 1
        reward = -1  # step penalty to encourage speed

        # Move Pac-Man
        if action not in ACTION_TO_DELTA:
            action = STAY
        self.pacman_pos = self._move(self.pacman_pos, action)

        # Collect pellets / power pellets
        reward += self._handle_pellet_collection()

        # Update ghost respawns and spawn additional ghosts
        self._update_respawns()
        self._spawn_new_ghost_if_needed()

        # Move ghosts
        self._move_ghosts()

        # Resolve collisions
        reward += self._handle_collisions()

        # Countdown scared mode
        if self.scared_timer > 0:
            self.scared_timer -= 1

        # Win / lose / timeout
        if len(self.pellets) == 0 and len(self.power_pellets) == 0:
            reward += 1000
            self.done = True

        if self.deaths > self.max_deaths:
            self.done = True

        if self.turn_count >= self.max_steps:
            self.done = True

        obs = self._get_obs()
        info = self._get_info()
        terminated = self.done
        truncated = False

        return obs, reward, terminated, truncated, info

    def render(self):
        grid = [list(row) for row in self.original_board]

        # Clear dynamic symbols from visual copy
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] in {"P", "G"}:
                    grid[r][c] = " "

        # Re-draw pellets that remain
        for r, c in self.pellets:
            grid[r][c] = "."
        for r, c in self.power_pellets:
            grid[r][c] = "o"

        # Draw Pac-Man
        pr, pc = self.pacman_pos
        grid[pr][pc] = "P"

        # Draw active ghosts
        for ghost in self.ghosts:
            if ghost.active:
                gr, gc = ghost.pos
                grid[gr][gc] = "G"

        print("\n".join("".join(row) for row in grid))
        print(
            f"Turn={self.turn_count}  Deaths={self.deaths}  "
            f"Scared={self.scared_timer}  Remaining={len(self.pellets) + len(self.power_pellets)}"
        )
        print()
    def _build_render_grid(self):
        grid = [list(row) for row in self.original_board]

        # Clear dynamic symbols from visual copy
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] in {".", "o", "P", "G"}:
                    grid[r][c] = " "

        # Re-draw pellets that remain
        for r, c in self.pellets:
            grid[r][c] = "."
        for r, c in self.power_pellets:
            grid[r][c] = "o"

        # Draw Pac-Man
        pr, pc = self.pacman_pos
        grid[pr][pc] = "P"

        # Draw active ghosts
        for ghost in self.ghosts:
            if ghost.active:
                gr, gc = ghost.pos
                grid[gr][gc] = "G"

        return grid

    def render_frame(self, cell_size: int = 24, margin: int = 10):
        """
        Return an RGB PIL image frame for saving GIF/video.
        """
        grid = self._build_render_grid()
        width = self.cols * cell_size + 2 * margin
        height = self.rows * cell_size + 2 * margin + 40

        img = Image.new("RGB", (width, height), color=(0, 0, 0))
        draw = ImageDraw.Draw(img)

        for r in range(self.rows):
            for c in range(self.cols):
                x0 = margin + c * cell_size
                y0 = margin + r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                ch = grid[r][c]

                if ch == "#":
                    draw.rectangle([x0, y0, x1, y1], fill=(30, 60, 200))
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

                    if ch == ".":
                        cx = x0 + cell_size // 2
                        cy = y0 + cell_size // 2
                        rad = max(2, cell_size // 8)
                        draw.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=(255, 255, 255))

                    elif ch == "o":
                        cx = x0 + cell_size // 2
                        cy = y0 + cell_size // 2
                        rad = max(4, cell_size // 4)
                        draw.ellipse([cx-rad, cy-rad, cx+rad, cy+rad], fill=(255, 180, 180))

                    elif ch == "P":
                        pad = 3
                        draw.ellipse([x0+pad, y0+pad, x1-pad, y1-pad], fill=(255, 255, 0))

                    elif ch == "G":
                        pad = 3
                        ghost_color = (100, 100, 255) if self.scared_timer > 0 else (255, 0, 0)
                        draw.rounded_rectangle([x0+pad, y0+pad, x1-pad, y1-pad],
                                               radius=4, fill=ghost_color)

                    elif ch == "T":
                        draw.rectangle([x0, y0, x1, y1], outline=(180, 180, 180))

        status = (
            f"Turn={self.turn_count}  Deaths={self.deaths}  "
            f"Scared={self.scared_timer}  Remaining={len(self.pellets) + len(self.power_pellets)}"
        )
        draw.text((margin, self.rows * cell_size + margin + 8), status, fill=(255, 255, 255))

        return img