# equations of CBF and QP matrix
import numpy as np
from cvxopt import matrix, solvers # I use it for convex optimization and find u_opt that is the vector of optimal velocity

solvers.options['show_progress'] = False

class QPColregController:
    # I add the parameter enable_colregs (default at False to start progressively)
    def __init__(self, d_safe=3.0, gamma=1.2, v_max=2.5, r_threshold=0.1, enable_colregs=True):
        self.d_safe = d_safe
        self.gamma = gamma
        self.v_max = v_max
        self.r_threshold = r_threshold
        self.enable_colregs = enable_colregs 
        
        # COLREGs FINE TUNING PARAMETERS
        self.gamma_colreg = 0.8  # Softer reaction for maritime rules (prevents chattering)
        self.r_colreg = 0.2      # Independent robustness margin for COLREGs
        self.mu_smooth = 5.0     # Smoothing parameter for Eq. 6 (Log-Sum-Exp)
        
        # Eq. 5 PARAMETERS (Relaxation Term o_j)
        self.decay_rate = 5.0    # Decay rate for o_j(t) to reach zero smoothly
        self.oj_memory = {}      # Dictionary to track o_j(t) history for each obstacle
        self.dt = 0.1            # Estimated simulation time step

    # base of the Quadratic Programming Algorithm to find min(0.5*u^T*P*u+q^T*u)
    def compute_control(self, p_self, v_nominal, intruders):
        n = 2
        P = matrix(np.eye(n))
        q = matrix(-v_nominal.astype(float))

        A_list = []
        b_list = []

        # Maximum speed constraint |v_x|<=v_max & |v_y|<=v_max
        # I prevent reverse by forcing v_y >= 0
        G_speed = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        h_speed = np.array([self.v_max, self.v_max, self.v_max, 0.0]) 
        A_list.append(G_speed)
        b_list.append(h_speed)

        # Track currently visible intruders to clean up oj_memory
        current_intruder_ids = []

        for pos_rel, vel_rel in intruders:
            dist = np.linalg.norm(pos_rel)
            if dist < 0.1:
                continue
                
            # Generate a temporary ID based on relative position to track o_j(t) consistently
            intruder_id = f"{round(pos_rel[0],1)}_{round(pos_rel[1],1)}"
            current_intruder_ids.append(intruder_id)
                
            # Dynamic size of obstacles
            is_moving = np.linalg.norm(vel_rel) > 0.1
            is_ahead_physical = pos_rel[1] > -0.5 
            
            if is_moving:
                if is_ahead_physical:
                    actual_d_safe = 8.0 
                    actual_gamma = 0.5
                else:
                    actual_d_safe = 4.5 
                    actual_gamma = 1.5
            else:
                actual_d_safe = 3.0 # Safe margin for 3D hull
                actual_gamma = 1.2  # Smooth and early reaction

            # RULE R1: SAFE DISTANCE (Obstacle Avoidance Pura) with CBF
            h_R1 = dist**2 - actual_d_safe**2 #h(x)>d_safe
            
            # I don't move the object but add an orthogonal micro-bias to the gradient 
            # if the obstacle is perfectly straight ahead. It breaks the stalemate without creating trajectory changes.
            grad_h_R1 = -2 * pos_rel 
            if abs(pos_rel[0]) < 0.5 and pos_rel[1] > 0:
                grad_h_R1[0] -= 2.0 # Adds a subtle but decisive lateral push

            # I add the processed constraints to the QP (Regola R1 sempre in AND)
            G_cbf = -grad_h_R1.reshape(1, 2)
            h_cbf = np.array([actual_gamma * h_R1 - self.r_threshold])
            
            A_list.append(G_cbf)
            b_list.append(h_cbf)

            # RULES COLREGs (Verify if the obstacle is moving and if it's in front of us )
            if self.enable_colregs and is_moving and dist < 6.0:
                # I change the concept of "ahead" (is_ahead) with respect to the NOSE OF THE BOAT (pos_rel[1]), and not with respect to the theoretical target (v_nominal).
                is_ahead = pos_rel[1] > 0
                
                # overtaking rule (I don't know if the ship is behind)
                if not is_ahead:
                    continue 
                
                # If the result is negative, the obstacle is to the left (Port).
                # If the result is positive, the obstacle is to the right (Starboard).
                cross_product = pos_rel[0] * v_nominal[1] - pos_rel[1] * v_nominal[0]
                
                # RULE R6: Stand-on Vessel (verify if the boat coming from left cross_product<0)
                # RULES R3/R4: Give-way Vessel 
                # Eq. 6: a -> b equivale a max(-rho_a, rho_b) >= r
                
                rho_a = cross_product  # >0 the ship is ahead and to the right
                rho_b = -cross_product # We want to bring the obstacle to our left (<0)

                # Calculate the maximum between the two robustness functions (STL logical OR)
                
                # SMOOTH MAXIMUM (Log-Sum-Exp)
                # Replaces the hard boolean if/else max() to regularize the QP optimization
                val_a = -rho_a
                val_b = rho_b
                max_val = max(val_a, val_b)
                
                exp_a = np.exp(self.mu_smooth * (val_a - max_val))
                exp_b = np.exp(self.mu_smooth * (val_b - max_val))
                sum_exp = exp_a + exp_b
                
                h_OR_raw = max_val + (1.0 / self.mu_smooth) * np.log(sum_exp)
                
                # Eq. 5: RELAXATION TERM o_j(t)
                # Prevents over-conservative behaviors if the CBF is already violated upon detection
                if intruder_id not in self.oj_memory:
                    if h_OR_raw < self.r_colreg:
                        self.oj_memory[intruder_id] = abs(h_OR_raw - self.r_colreg) + 0.1
                    else:
                        self.oj_memory[intruder_id] = 0.0
                
                # Exponential decay of the relaxation term
                self.oj_memory[intruder_id] *= np.exp(-self.decay_rate * self.dt)
                o_j_t = self.oj_memory[intruder_id]
                
                # Final h_OR with relaxation term included
                h_OR = h_OR_raw + o_j_t
                
                # Eq. 9: G_OR WITH SOFTMAX WEIGHTED GRADIENT
                # Properly calculates the gradient of the smooth max function
                w_a = exp_a / sum_exp
                w_b = exp_b / sum_exp
                
                grad_a = np.array([-pos_rel[1], pos_rel[0]]) * -1
                grad_b = np.array([-pos_rel[1], pos_rel[0]]) * -1
                
                # Weighted sum of gradients
                grad_OR = w_a * grad_a + w_b * grad_b

                # I halve the allowable margin of maneuver (rilassato/irrigidito ora gestito da Eq.6)
                # relax the bond
                
                # I add the processed constraints to the QP (AND with R1)
                G_colreg = -grad_OR.reshape(1, 2)
                
                # Using the dedicated COLREGs fine-tuned parameters
                h_colreg = np.array([self.gamma_colreg * h_OR - self.r_colreg])
                
                A_list.append(G_colreg)
                b_list.append(h_colreg)

        # Cleanup oj_memory for obstacles that are no longer in range (prevents memory leaks)
        self.oj_memory = {k: v for k, v in self.oj_memory.items() if k in current_intruder_ids}

        G = matrix(np.vstack(A_list)) if A_list else matrix(np.empty((0, 2)))
        h = matrix(np.hstack(b_list)) if b_list else matrix(np.empty((0,)))

        # test fail-safe
        try:
            sol = solvers.qp(P, q, G, h)
            return np.array(sol['x']).flatten()
        except ValueError:
            return np.array([0.0, self.v_max])
