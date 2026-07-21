# equations of CBF and QP matrix
import numpy as np
from cvxopt import matrix, solvers # I use it for convex optimization and find u_opt that is the vector of optimal velocity

solvers.options['show_progress'] = False

class QPColregController:
    # I add the parameter enable_colregs (default at False to start progressively)
    def __init__(self, d_safe=3.0, gamma=1.2, v_max=2.5, r_threshold=0.1, enable_colregs=False):
        self.d_safe = d_safe
        self.gamma = gamma
        self.v_max = v_max
        self.r_threshold = r_threshold
        self.enable_colregs = enable_colregs 

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

        for pos_rel, vel_rel in intruders:
            dist = np.linalg.norm(pos_rel)
            if dist < 0.1:
                continue
                
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
            if abs(pos_rel[0]) < 0.2 and pos_rel[1] > 0:
                grad_h_R1[0] -= 0.5 # Adds a subtle but decisive lateral push

            # I add the processed constraints to the QP (Regola R1 sempre in AND)
            G_cbf = -grad_h_R1.reshape(1, 2)
            h_cbf = np.array([actual_gamma * h_R1 - self.r_threshold])
            
            A_list.append(G_cbf)
            b_list.append(h_cbf)

            # RULES COLREGs (Verify if the obstacle is moving and if it's in front of us )
            if self.enable_colregs:
                is_ahead = np.dot(pos_rel, v_nominal) > 0
                
                # overtaking rule (I don't know if the ship is behind)
                if not is_ahead and is_moving:
                    continue 
                
                if is_moving:
                    # If the result is negative, the obstacle is to the left (Port).
                    # If the result is positive, the obstacle is to the right (Starboard).
                    cross_product = pos_rel[0] * v_nominal[1] - pos_rel[1] * v_nominal[0]
                    
                    # RULE R6: Stand-on Vessel (verify if the boat coming from left cross_product<0)
                    # RULES R3/R4: Give-way Vessel 
                    # Eq. 6: a -> b equivale a max(-rho_a, rho_b) >= r
                    
                    rho_a = cross_product  # >0 the ship is ahead and to the right
                    rho_b = -cross_product # We want to bring the obstacle to our left (<0)

                    # Calculate the maximum between the two robustness functions (STL logical OR)
                    if -rho_a > rho_b:
                        h_OR = -rho_a
                        grad_OR = np.array([-pos_rel[1], pos_rel[0]]) * -1
                    else:
                        h_OR = rho_b
                        grad_OR = np.array([-pos_rel[1], pos_rel[0]]) * -1

                    # I halve the allowable margin of maneuver (rilassato/irrigidito ora gestito da Eq.6)
                    # relax the bond
                    
                    # I add the processed constraints to the QP (AND with R1)
                    G_colreg = -grad_OR.reshape(1, 2)
                    h_colreg = np.array([actual_gamma * h_OR - self.r_threshold])
                    
                    A_list.append(G_colreg)
                    b_list.append(h_colreg)

        G = matrix(np.vstack(A_list)) if A_list else matrix(np.empty((0, 2)))
        h = matrix(np.hstack(b_list)) if b_list else matrix(np.empty((0,)))

        # test fail-safe
        try:
            sol = solvers.qp(P, q, G, h)
            return np.array(sol['x']).flatten()
        except ValueError:
            return np.array([0.0, self.v_max])
