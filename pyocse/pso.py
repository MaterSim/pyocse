import numpy as np
import logging

class PSO:
    def __init__(self, obj_function, obj_args, bounds, seed=None,
                 num_particles=30, dimensions=2,
                 inertia=0.7, cognitive=1.5, social=1.5, 
                 mutation_rate=0.1, crossover_rate=0.8, 
                 max_iter=100, ncpu=1, log_file="pso.log"):
        """
        Initialize the PSO-GA hybrid optimizer.

        Parameters:
        - obj_function: Objective function to minimize.
        - bounds: Tuple (lower_bounds, upper_bounds) for each dimension.
        - num_particles: Number of particles (and individuals in GA).
        - dimensions: Number of dimensions in the search space.
        - inertia: Inertia weight for velocity update (PSO).
        - cognitive: Cognitive (personal best) weight (PSO).
        - social: Social (global best) weight (PSO).
        - mutation_rate: Probability of mutation (GA).
        - crossover_rate: Probability of crossover (GA).
        - max_iter: Maximum number of iterations.
        - ncpu: number of parallel processes
        - log_file: log file name
        """
        self.obj_function = obj_function
        self.obj_args = obj_args
        self.bounds = bounds
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_iter = max_iter
        self.ncpu = ncpu
        self.debug = True
        self.obj_args = self.obj_args + (self.ncpu, )
        self.log_file = log_file

        # Initialize logger
        logging.getLogger().handlers.clear()
        logging.basicConfig(format="%(asctime)s| %(message)s", 
                            filename=self.log_file, 
                            level=logging.INFO)
        self.logger = logging
        

        # Initialize bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])

        # Initialize particles and Rescale to (0, 1)
        self.positions = np.random.uniform(self.lower_bounds, 
                                           self.upper_bounds, 
                                           (num_particles, dimensions))
        self.positions = (self.positions - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        self.velocities = 0.1 * np.random.uniform(-1, 1, (num_particles, dimensions))

        if seed is not None:
            self.set_seeds(seed)

        # Init: evaluate the score for each particle
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.zeros(len(self.positions))
        scores = self.safe_evaluate_par()
        for i in range(self.num_particles):
            self.personal_best_scores[i] = scores[i]
            self.logger.info(f"{i} score: {scores[i]}")

        min_idx = np.argmin(self.personal_best_scores)
        self.global_best_position = self.personal_best_positions[min_idx]
        self.global_best_score = np.min(self.personal_best_scores)
        self.global_best_id = min_idx

    def set_seeds(self, seed):
        """
        reset the positions to the seed
        """
        n_seeds = len(seed)
        self.logger.debug(f"Set Seeds {n_seeds}")
        self.positions[:n_seeds] = (seed - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)

    def rescale(self, scaled_values):
        """
        Rescale values from (0, 1) back to original bounds.
        """
        return scaled_values * (self.upper_bounds - self.lower_bounds) + self.lower_bounds

    def safe_evaluate_par(self):
        p_actuals = []
        for p in self.positions:
            p_actual = self.rescale(p)
            p_actuals.append(p_actual)

        scores = self.obj_function(p_actuals, *self.obj_args)
        return scores

    def pso_step(self):
        """Perform one step of PSO."""
        for i in range(self.num_particles):
            # update velocity
            r1, r2 = np.random.rand(), np.random.rand()
            self.positions[i] += self.velocities[i]
            v = (
                 self.inertia * self.velocities[i] + 
                 self.cognitive * r1 * (self.personal_best_positions[i] - self.positions[i]) +
                 self.social * r2 * (self.global_best_position - self.positions[i]) + 
                 0.05 * np.random.uniform(-1, 1, (self.dimensions))
            )
            v /= np.abs(v).max()
            self.velocities[i] = 0.1 * v #(-0.1, 0.1)
            self.positions[i] = np.clip(self.positions[i], 0, 1)
        
        # Evaluate results in parallel 
        scores = self.safe_evaluate_par()

        for i in range(self.num_particles):
            score = scores[i]
            if score < self.personal_best_scores[i]:
                strs = f"{i} score: {score:.4f}  pbest: {self.personal_best_scores[i]:.4f}"
                self.personal_best_scores[i] = score
                self.personal_best_positions[i] = self.positions[i]
                self.logger.info(strs)

        min_idx = np.argmin(self.personal_best_scores)
        if self.personal_best_scores[min_idx] < self.global_best_score:
            self.global_best_score = self.personal_best_scores[min_idx]
            self.global_best_position = self.personal_best_positions[min_idx]
            self.global_best_id = min_idx

    def optimize(self, x0=None):
        """Perform optimization using the PSO-GA hybrid algorithm."""
        if x0 is not None:
            self.positions = x0

        for iteration in range(self.max_iter):
            self.pso_step()
            
            if iteration % 1 == 0:
                strs = f"Iteration {iteration + 1}/{self.max_iter}, "
                strs += f"ID: {self.global_best_id}, "
                strs += f"Best Score: {self.global_best_score:.4f}"
                self.logger.info(strs)
        best_position = self.rescale(self.global_best_position)
        return best_position, self.global_best_score
