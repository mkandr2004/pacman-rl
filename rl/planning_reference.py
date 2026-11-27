"""Explicit classical reference for the existing engine. No learned weights."""

from collections import deque
from heapq import heappop, heappush
from itertools import count

import pygame
import numpy as np

from rl.ghost_forecast import forecast_ghosts


DELTAS = ((-1,0), (1,0), (0,-1), (0,1))


def neighbors(position, stride):
    x, y = position
    for dx, dy in DELTAS:
        yield x+dx*stride, y+dy*stride


def component(start, walls, width, height):
    visited = {start}
    frontier = deque([start])
    while frontier:
        for cell in neighbors(frontier.popleft(), 60):
            x, y = cell
            if cell not in visited and cell not in walls and 0 <= x < width and 0 <= y < height:
                visited.add(cell)
                frontier.append(cell)
    return visited


class PlanningReference:
    def __init__(self, seed=0, danger_weight=2.5, anticipate=True, local_avoidance=True):
        self.rng = np.random.default_rng(seed)
        self.danger_weight = danger_weight
        self.anticipate = anticipate
        self.local_avoidance = local_avoidance and anticipate
        self.path = deque()
        self.expected = None
        self.key = None
        self.geometry = None
        self.layout = None
        self.visits = {}
        self.decisions = 0
        self.target_kind = None
        self.detours = deque()

    def _geometry(self, raw):
        width, height = raw["screen_size"]
        self.layout = raw["walls"]
        blocked = np.full((height, width), -1, dtype=np.int16)
        blocked[height-29:, :] = -2
        blocked[:, width-29:] = -2
        for index, (x, y) in enumerate(self.layout):
            if x in (0,width-60) or y in (0,height-60):
                value = -2
            else:
                value = index
            blocked[max(0,y-29):y+60, max(0,x-29):x+60] = value
        # Border cells take priority even where their collision areas overlap.
        blocked[:60,:] = blocked[:,:60] = -2
        blocked[height-89:,:] = blocked[:,width-89:] = -2
        self.geometry = blocked

    def _targets(self, raw):
        """Rank walls by newly accessible items on the tile graph."""
        width, height = raw["screen_size"]
        tile = lambda p: (p[0]//60*60, p[1]//60*60)
        walls = set(raw["walls"])
        start = tile(raw["pacman"])
        reachable = component(start,walls,width,height)
        items = raw["points"] + raw["coins"]
        distant = {tile(item) for item in items if tile(item) not in reachable}
        bridges = []
        if distant and (raw["bonus_positions"][2] or raw["effects"][2]):
            for wall in walls:
                x, y = wall
                if x in (0,width-60) or y in (0,height-60):
                    continue
                adjacent = tuple(neighbors(wall,60))
                if any(cell in reachable for cell in adjacent) and any(
                    cell not in reachable and cell not in walls for cell in adjacent
                ):
                    gain = len(distant & component(start,walls-{wall},width,height))
                    if gain:
                        bridges.append((gain,wall))
        if bridges and raw["effects"][2]:
            best = max(gain for gain,_ in bridges)
            return "wall", [pygame.Rect(*wall,60,60) for gain,wall in bridges if gain==best]
        reachable_items = [item for item in items if tile(item) in reachable]
        if bridges and raw["bonus_positions"][2] and (not self.anticipate or not reachable_items):
            lemons = [pygame.Rect(*point,30,30) for point in raw["bonus_positions"][2]
                      if tile(point) in reachable]
            if lemons:
                return "lemon", lemons
        targets = [pygame.Rect(*point,60,60) for point in raw["points"] if tile(point) in reachable]+[
            pygame.Rect(*point,30,30) for point in raw["coins"] if tile(point) in reachable]
        if self.anticipate and raw["ghosts"]:
            bonus_types = (0,3) if raw["health"] < 3 else (3,)
            for idx in bonus_types:
                targets.extend(pygame.Rect(*p,30,30) for p in raw["bonus_positions"][idx] if tile(p) in reachable)
        return "items", targets

    def _risk(self, env, raw):
        risk = np.zeros((env.rows, env.cols), dtype=np.float32)
        if not self.danger_weight or not raw["ghosts"]:
            return risk
        player = pygame.Rect(*raw["pacman"],30,30)
        samples = forecast_ghosts(env._game,[player]*24,seed=self.decisions)
        for time, ghosts in enumerate(samples):
            if time % 4:
                continue
            for ghost in ghosts:
                for y in range(max(1,(ghost.top-30)//60),min(env.rows-1,(ghost.bottom+29)//60+1)):
                    for x in range(max(1,(ghost.left-30)//60),min(env.cols-1,(ghost.right+29)//60+1)):
                        risk[y,x] += 1/6
        return risk

    def _plan(self, env, raw):
        speed = round(raw["effects"][3]*6)
        start = raw["pacman"]
        if self.layout != raw["walls"]:
            self._geometry(raw)
        kind, targets = self._targets(raw)
        self.target_kind = kind
        if not targets:
            return deque()
        # Dilate the target rectangles by the player's size. A* searches
        # top-left sprite coordinates, following the same frame repetition.
        height,width = self.geometry.shape
        target_grid = np.zeros((height,width),dtype=bool)
        for rect in targets:
            target_grid[max(0,rect.top-29):rect.bottom,max(0,rect.left-29):rect.right] = True
        boxes = [(r.left-29,r.top-29,r.right-1,r.bottom-1) for r in targets]
        stride = speed * env.frames_per_action
        def heuristic(x,y):
            return min(max(l-x,0,x-r)+max(t-y,0,y-b) for l,t,r,b in boxes)/stride
        risk = self._risk(env,raw)
        order = count()
        queue = [(0, next(order), 0, start)]
        costs = {start:0.}
        parents = {}
        directions = self.rng.permutation(4).tolist()
        while queue:
            _, _, cost, position = heappop(queue)
            if cost != costs[position]:
                continue
            x,y = position
            if position != start and target_grid[y,x]:
                path = deque()
                while position != start:
                    previous, action = parents[position]
                    path.appendleft((action, position))
                    position = previous
                return path
            for action in directions:
                dx,dy = DELTAS[action]
                nx,ny = x,y
                for _ in range(env.frames_per_action):
                    a,b = nx+dx*speed,ny+dy*speed
                    collision = self.geometry[b,a]
                    if collision != -1 and not (kind == "wall" and collision >= 0 and target_grid[b,a]):
                        break
                    nx,ny = a,b
                    if target_grid[ny,nx] and kind == "wall":
                        break
                new = nx,ny
                if new == position:
                    continue
                cell = ny//60,nx//60
                # Risk near several ghosts and traffic concentrated in a
                # recently exhausted area can outweigh shortest distance.
                protection = raw["effects"][0]*200/env.frames_per_action
                exposure = risk[cell] * self.danger_weight * (0.25 if cost < protection else 1)
                price = cost+1+exposure+min(3,self.visits.get(new,0))*.35
                if price < costs.get(new,float("inf")):
                    costs[new]=price
                    parents[new]=position,action
                    heappush(queue,(price+heuristic(nx,ny),next(order),price,new))
        return deque()

    def _predict_path(self, env, raw, actions):
        """Ordinary motion on the current walls for the short safety forecast."""
        speed = round(raw["effects"][3]*6)
        x,y = raw["pacman"]
        frames = []
        for action in actions:
            dx,dy = DELTAS[action]
            for _ in range(env.frames_per_action):
                nx,ny = x+dx*speed,y+dy*speed
                if self.geometry[ny,nx] == -1:
                    x,y = nx,ny
                frames.append(pygame.Rect(x,y,30,30))
        return frames

    @staticmethod
    def _clearance(player, ghost):
        return max(player.left-ghost.right,ghost.left-player.right,0)+max(
            player.top-ghost.bottom,ghost.top-player.bottom,0)

    def _avoid(self, env, raw, planned):
        player_rect = pygame.Rect(*raw["pacman"],30,30)
        ghost_rects = [ghost.get_rect() for ghost in env._game._screen_field_mapper.get_ghosts()]
        if not ghost_rects or min(self._clearance(player_rect,g) for g in ghost_rects)>65:
            return None
        protected = round(raw["effects"][0]*200)
        if protected >= 25 or raw["effects"][2]:
            return None
        nominal = [action for action,_ in self.path][:7]
        actions = [planned]+nominal
        actions += [actions[-1]]*(8-len(actions))
        def assess(sequence):
            frames = self._predict_path(env,raw,sequence)
            earliest = len(frames)+1
            minimum = 100
            for i,(rect,ghosts) in enumerate(zip(frames,forecast_ghosts(
                env._game,frames,seed=self.decisions//4
            ))):
                if i < protected:
                    continue
                if any(rect.colliderect(ghost) for ghost in ghosts):
                    earliest = i
                    break
                minimum = min(minimum,min(self._clearance(rect,g) for g in ghosts))
            return earliest,minimum,frames[-1].topleft
        imminent,_,destination = assess(actions)
        if imminent >= 24:
            return None
        best_key = (imminent >= 24,imminent,0,0)
        best = None
        for first in np.flatnonzero(env.action_masks()):
            for duration in (2,4,6):
                for second in range(4):
                    sequence=[int(first)]*duration+[second]*(8-duration)
                    earliest,minimum,end = assess(sequence)
                    if end == raw["pacman"]:
                        continue
                    distance=abs(end[0]-destination[0])+abs(end[1]-destination[1])
                    key=(earliest>=24,earliest,-distance+minimum*2,
                         -self.visits.get(end,0))
                    if key>best_key:
                        best_key=key
                        best=sequence
        if best:
            # Execute a small prefix then replan from the observed state.
            self.detours=deque(best[1:2])
            return best[0]
        return None

    def predict(self, env):
        raw = env._last_raw_observation
        position = raw["pacman"]
        self.visits[position] = self.visits.get(position,0)+1
        self.decisions += 1
        mask = env.action_masks()
        if self.detours:
            action = self.detours.popleft()
            if mask[action]:
                return action
        key = (raw["walls"],raw["points"],raw["coins"],raw["bonus_positions"],raw["effects"][3],bool(raw["effects"][2]))
        if not self.path or position != self.expected or self.key != key:
            self.path = self._plan(env,raw)
            self.key = key
        if self.path and mask[self.path[0][0]]:
            action,self.expected=self.path.popleft()
            if self.local_avoidance:
                safer = self._avoid(env,raw,action)
                if safer is not None:
                    self.path.clear()
                    return safer
            return action
        # General fallback for a target-free component, evaluated by the
        # actual legal directions and number of previous visits.
        self.path.clear()
        speed=round(raw["effects"][3]*6)*env.frames_per_action
        options=list(neighbors(position,speed))
        return min(np.flatnonzero(mask),key=lambda action:self.visits.get(options[action],0)).item()
