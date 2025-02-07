import numpy as np
import logging
import xml.etree.ElementTree as ET
import os
from pyocse.parameters import prettify

class PSO:
    def __init__(self, obj_function, obj_args, bounds, seed=None,
                 num_particles=30, dimensions=2,
                 inertia=0.7, cognitive=1.5, social=1.5, 
                 max_iter=100, ncpu=1, log_file="pso.log", resume_file="pso.xml"):
        """
        Initialize the PSO optimizer.

        Parameters:
        - obj_function: Objective function to minimize.
        - bounds: Tuple (lower_bounds, upper_bounds) for each dimension.
        - num_particles: Number of particles.
        - dimensions: Number of dimensions in the search space.
        - inertia: Inertia weight for velocity update (PSO).
        - cognitive: Cognitive (personal best) weight (PSO).
        - social: Social (global best) weight (PSO).
        - max_iter: Maximum number of iterations.
        - ncpu: number of parallel processes
        - log_file: log file name
        - resume_file: location of saved pso.xml
        """
        self.obj_function = obj_function
        self.obj_args = obj_args
        self.bounds = bounds
        self.num_particles = num_particles
        self.dimensions = dimensions
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.max_iter = max_iter
        self.ncpu = ncpu
        self.log_file = log_file
        self.resume_file = resume_file

        # Initialize logger
        logging.getLogger().handlers.clear()
        logging.basicConfig(format="%(asctime)s| %(message)s", 
                            filename=self.log_file, 
                            level=logging.INFO)
        self.logger = logging

        # Initialize bounds
        self.lower_bounds = np.array([b[0] for b in bounds])
        self.upper_bounds = np.array([b[1] for b in bounds])

        # Initialize particles
        self.positions = np.random.uniform(self.lower_bounds, 
                                           self.upper_bounds, 
                                           (num_particles, dimensions))
        self.positions = (self.positions - self.lower_bounds) / (self.upper_bounds - self.lower_bounds)
        self.velocities = 0.1 * np.random.uniform(-1, 1, (num_particles, dimensions))

        # Load parameters if seed is provided or load from XML
        if os.path.exists(self.resume_file):
            self.load_saved_parameters()
            print("load positions from pso.xml:", os.path.abspath(self.resume_file))
        elif seed is not None:
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
        """Set initial positions using the provided seed."""
        n_seeds = len(seed)
        self.logger.info(f"Set Seeds {n_seeds}")
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

    def save_parameter(self, filename="pso.xml"):
        """Save current best parameters to an XML file in array format."""
        # Create root element
        root = ET.Element("pso")
    
        # Global Best
        gb_elem = ET.SubElement(root, "global_best")
        gb_elem.text = str(self.global_best_position.tolist())  # Convert numpy array to Python list format
    
        # Local Bests
        lb_elem = ET.SubElement(root, "local_bests")
        for i in range(self.num_particles):
            p_elem = ET.SubElement(lb_elem, f"particle_{i}")
            p_elem.text = str(self.personal_best_positions[i].tolist())
    
        # Velocities
        vel_elem = ET.SubElement(root, "velocities")
        for i in range(self.num_particles):
            v_elem = ET.SubElement(vel_elem, f"particle_{i}")
            v_elem.text = str(self.velocities[i].tolist())

        # Use prettify to get a pretty-printed XML string
        pretty_xml = prettify(root)

        # Write the pretty-printed XML to a file
        with open(filename, 'w') as f:
            f.write(pretty_xml)
        
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
        """Optimize with PSO and save results to XML."""
        if x0 is not None:
            self.positions = x0

        for iteration in range(self.max_iter):
            self.pso_step()
            self.logger.info(f"Iteration {iteration+1}/{self.max_iter}, Best Score: {self.global_best_score:.4f}")
        best_position = self.rescale(self.global_best_position)
        self.save_parameter()  # Save final optimized parameters
        return best_position, self.global_best_score

    def load_saved_parameters(self):
        
        tree = ET.parse(self.resume_file)
        root = tree.getroot()
    
        # Load global best position (convert string list to numpy array)
        self.global_best_position = np.array(
            np.fromstring(root.find("global_best").text.strip()[1:-1], sep=",")
        )
    
        # Load personal best positions
        self.personal_best_positions = np.zeros((self.num_particles, self.dimensions))
        for i, p_elem in enumerate(root.find("local_bests")):
            self.personal_best_positions[i] = np.fromstring(p_elem.text.strip()[1:-1], sep=",")
    
        # Load velocities
        self.velocities = np.zeros((self.num_particles, self.dimensions))
        for i, v_elem in enumerate(root.find("velocities")):
            self.velocities[i] = np.fromstring(v_elem.text.strip()[1:-1], sep=",")
    
        self.logger.info(f"Loaded PSO from {self.resume_file}")
