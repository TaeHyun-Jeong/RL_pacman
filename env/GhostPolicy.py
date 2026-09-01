class Ghost:  #7step Chase, 3step Scatter
    def __init__(self, scatter_corner=None, cycle_len=10, chase_steps=8):
        self.SCATTER = 0
        self.CHASE = 1

        try:
            cycle_len = int(cycle_len)
        except Exception:
            cycle_len = 11
        if cycle_len <= 0:
            cycle_len = 11

        if chase_steps is None:
            chase_steps = int(round(0.6 * cycle_len))
        else:
            try:
                chase_steps = int(chase_steps)
            except Exception:
                chase_steps = int(round(0.6 * cycle_len))

        if chase_steps < 0:
            chase_steps = 0
        if chase_steps > cycle_len:
            chase_steps = cycle_len

        self.cycle_len = cycle_len
        self.chase_steps = chase_steps

        self.dir_priority = [(0, -1), (-1, 0), (0, 1), (1, 0)]
        self.opposite = {(0,-1):(0,1), (0,1):(0,-1), (-1,0):(1,0), (1,0):(-1,0)}
        self.step_count = 0
        self.curr_dir = None
        self.scatter_corner = scatter_corner

    def reset(self):
        self.step_count = 0
        self.curr_dir = None

    def _is_walkable(self, env, x, y):
        return (0 <= x < env.width and 0 <= y < env.height
                and env.wall.wallmap[y, x] == 0)

    def _mode_at(self, t):
        return self.CHASE if (t % self.cycle_len) < self.chase_steps else self.SCATTER

    def _init_scatter_corner_if_needed(self, env):
        if self.scatter_corner is not None:
            return
        x, y = env.width - 2, 1
        if not self._is_walkable(env, x, y):
            best, best_d = None, None
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    xx, yy = x + dx, y + dy
                    if self._is_walkable(env, xx, yy):
                        d = (env.width-1-xx)**2 + (yy-0)**2
                        if best_d is None or d < best_d:
                            best, best_d = (xx, yy), d
            self.scatter_corner = best if best is not None else (x, y)
        else:
            self.scatter_corner = (x, y)

    def _target_tile(self, env):
        px, py = env.pacman_location
        return (px, py) if self._mode_at(self.step_count) == self.CHASE else self.scatter_corner

    def _available_actions(self, env, x, y):
        out = []
        for a, (dx, dy) in env.action.items():
            nx, ny = x + dx, y + dy
            if self._is_walkable(env, nx, ny):
                out.append((a, nx, ny, (dx, dy)))
        return out

    def _dir_priority_index(self, dvec):
        try:
            return self.dir_priority.index(dvec)
        except ValueError:
            return 9999

    def _choose_action(self, env, target, ghost_xy, allow_uturn):
        gx, gy = ghost_xy
        candidates = self._available_actions(env, gx, gy)
        if not candidates:
            return None

        pruned = []
        for a, nx, ny, dvec in candidates:
            if self.curr_dir is not None and not allow_uturn:
                if dvec == self.opposite.get(self.curr_dir, None):
                    continue
            pruned.append((a, nx, ny, dvec))
        if not pruned:
            pruned = candidates

        tx, ty = target
        best = None
        best_key = None
        for a, nx, ny, dvec in pruned:
            dist2 = (nx - tx) ** 2 + (ny - ty) ** 2
            key = (dist2, self._dir_priority_index(dvec))
            if best_key is None or key < best_key:
                best_key = key
                best = (a, nx, ny, dvec)
        return best

    def select_action(self, env):
        self._init_scatter_corner_if_needed(env)

        gx, gy = env.ghost_location
        curr_mode = self._mode_at(self.step_count)
        prev_mode = self._mode_at(self.step_count - 1) if self.step_count > 0 else None
        allow_uturn = (prev_mode is not None and curr_mode != prev_mode)

        target = self._target_tile(env)
        choice = self._choose_action(env, target, (gx, gy), allow_uturn)

        if choice is None:
            action = 0
        else:
            action, nx, ny, dvec = choice
            self.curr_dir = dvec

        self.step_count += 1
        return int(action)
