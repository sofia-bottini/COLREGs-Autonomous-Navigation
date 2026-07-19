# equations of CBF and QP matrix
import numpy as np
from cvxopt import matrix, solvers # I use it for convex optimization and find u_opt that is the vector of optimal velocity

solvers.options['show_progress'] = False

class QPColregController:
    # I add the parameter enable_colregs (default at False to start progressively)
    def __init__(self, d_safe=3.0, gamma=1.2, v_max=2.5, enable_colregs=False):
        self.d_safe = d_safe
        self.gamma = gamma
        self.v_max = v_max
        self.enable_colregs = enable_colregs 

    # base of the Quadratic Programming Algorithm to find min(0.5*u^T*P*u+q^T*u)
    def compute_control(self, p_self, v_nominal, intruders):
        n = 2
        P = matrix(np.eye(n))
        q = matrix(-v_nominal.astype(float))

        A_list = []
        b_list = []

        # Maximum speed constraint |v_x|<=v_max & |v_y|<=v_max
        G_speed = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        h_speed = np.array([self.v_max, self.v_max, self.v_max, self.v_max])
        A_list.append(G_speed)
        b_list.append(h_speed)

        for pos_rel, vel_rel in intruders:
            dist = np.linalg.norm(pos_rel)
            if dist < 0.1:
                continue

            # --- RULE R1: SAFE DISTANCE (Obstacle Avoidance Pura) with CBF ---
            h = dist**2 - self.d_safe**2 #h(x)>d_safe
            grad_h = -2 * pos_rel #boat direction
            
            gamma_val = self.gamma
            h_cbf_val = gamma_val * h

            # --- >RULES COLREGs (Verify if the obstacle is moving and if it's in front of us ) ---
            if self.enable_colregs:
                is_moving = np.linalg.norm(vel_rel) > 0.1
                is_ahead = np.dot(pos_rel, v_nominal) > 0
                
                # overtaking rule (I don't know if the ship is behind)
                if not is_ahead and is_moving:
                    continue 
                
                # If the result is negative, the obstacle is to the left (Port).
                # If the result is positive, the obstacle is to the right (Starboard).
                cross_product = pos_rel[0] * v_nominal[1] - pos_rel[1] * v_nominal[0]
                
                # RULE R6: Stand-on Vessel (verify if the boat coming from left cross_product<0)
                is_stand_on = (cross_product < 0) and (dist < self.d_safe * 4)
                if is_stand_on and is_moving:
                    gamma_val = self.gamma * 0.2 # relax the bond
                    h_cbf_val = gamma_val * h
                
                # RULES R3/R4: Give-way Vessel 
                if not is_stand_on and is_moving and dist < (self.d_safe * 3):
                    if cross_product > 0: # the ship is ahead
                        h_cbf_val = h_cbf_val * 0.5 # I halve the allowable margin of maneuver

            # I add the processed constraints to the QP
            G_cbf = -grad_h.reshape(1, 2)
            h_cbf = np.array([h_cbf_val])

            A_list.append(G_cbf)
            b_list.append(h_cbf)

        G = matrix(np.vstack(A_list)) if A_list else matrix(np.empty((0, 2)))
        h = matrix(np.hstack(b_list)) if b_list else matrix(np.empty((0,)))

        # test fail-safe
        try:
            sol = solvers.qp(P, q, G, h)
            return np.array(sol['x']).flatten()
        except ValueError:
            return np.array([0.0, 0.0])