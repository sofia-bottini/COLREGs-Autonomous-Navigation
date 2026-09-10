# COLREGs-Compliant Autonomous Boat Navigation

This repository contains the code for my project on autonomous maritime navigation. The goal was to make a boat navigate safely to a target while following real-world maritime rules (COLREGs). 

To achieve this, I combined Control Barrier Functions (CBFs) for physical safety with Signal Temporal Logic (STL) to encode the navigation rules. Everything is solved in real-time using a Quadratic Programming (QP) solver and simulated in a custom Unity environment via ML-Agents.

## The Problem & The Solution
Initially, mapping Boolean nautical rules (like "pass on the right OR the left") directly into the QP solver caused massive actuator chattering. This happened because the standard `max()` operator used for STL disjunctions creates non-differentiable edges, causing the solver to bounce between opposite gradients.

To fix this, I replaced the standard operator with a **Smooth Maximum (Log-Sum-Exp)** approximation and applied a **Softmax weighted gradient**. This made the optimization problem strictly convex and completely eliminated the steering vibrations.

I also had to implement a few practical workarounds to deal with the physics engine:
* **Relaxation term ($o_j(t)$):** I added a time-decaying memory term to prevent the QP from becoming infeasible if an obstacle spawns too close to the boat.
* **Kinematic Traction Control:** The QP calculates perfect steering angles, but Unity's hydrodynamics made the boat drift at 2.5 m/s. I wrote a simple control law to drop the throttle proportionally to the steering angle, forcing the hull to turn properly during emergency evasions.

## Repository Structure
* `qp_colreg_controller.py`: The math core. It builds the CBF and STL constraints, applies the smooth approximations, and runs the `cvxopt` solver.
* `demo_qp.py`: The bridge script. It handles sensor fusion (merging Unity's GPS data with local odometry for static buoys) and communicates with the ML-Agents environment.

## References
The mathematical framework behind this code is heavily based on:
1. *Charitidou & Dimarogonas (2023)* - For the Log-Sum-Exp smooth approximation (Eq. 6), the Softmax gradient (Eq. 9), and the relaxation term (Eq. 5).
2. *Krasowski & Althoff (2021)* - For translating the COLREGs into geometric constraints (e.g., using the cross-product to detect starboard/port).

## Setup and Usage
You will need Python 3.8+ and the Unity Editor to run the simulation.

1. Clone this repo:
   ```bash
   git clone [https://github.com/sofia-bottini/COLREGs-Autonomous-Navigation.git](https://github.com/sofia-bottini/COLREGs-Autonomous-Navigation.git)
   cd COLREGs-Autonomous-Navigation

Install the dependencies: pip install numpy cvxopt mlagents
Run the controller: python demo_qp.py
When the terminal says Waiting for connection to Unity, press the Play button inside the Unity Editor.

Results
I tested the final algorithm across 9 different stochastic scenarios in Unity, varying the traffic density and intersection angles.

The smooth mathematical approach successfully fixed the chattering issue. Overall, the boat achieved an 85.89% success rate with an average safe distance of 5.70m from obstacles. The remaining constraint violations (around 14%) were mostly due to extreme edge-case respawns in the simulator where evasion was physically impossible.
