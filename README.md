# ctr_policy_ros

This ros package is intended for use with stable-baselines to load and evaluate a RL agent in the
task of concentric tube robot control.

## Installation and use
* Currently, the only way I have been able to get this working is by creating a python 3 virtual 
environment that has the required packages for stable-baselines. You can find these requirements
  in the requirements.txt.
  
* My setup has the stable-baselines cloned, to a directory and install locally from there.

* You will also need the gym-ctm-ros python package installed. You should be able to clone and
install locally as with stable-baselines. Here is the link for that repo 
  [link](https://github.com/keshaviyengar/gym-ctm-ros/tree/master/ctm_envs).

* Sourcing the virtual environment, and running the nodes looks to work, as I'm only using the
loading functionality of stable-baselines.
  
* There are two nodes in two python files in the scripts directory:
    1. ctr_sim.py: This node runs a simple simulator that acts as a simulation interface for now. It will 
    receive actions published to "\delta_joints" and runs through one timestep to determine the
       new state (joint values, goal error).
       
    2. The ctr_sim.py then publishes the state (or obs in the code) as three seperate topics.
    These topics are the joint values "/joint_states", tip position "/tip_pos" and desired tip
       position "/desired_tip_pos".
       
    3. eval_agent.py: This node loads the agent and creates the policy model network. It subscribes to
    the "/joint_states", tip position "/tip_pos" and desired tip position "/desired_tip_pos" and
       creates the state required for the policy network. It then feeds it to the network, gets
       the output actions and publishes these under the topic "\delta_joints"