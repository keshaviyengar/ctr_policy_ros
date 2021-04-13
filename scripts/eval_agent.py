import rospy
import rospkg

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, TriggerRequest

import numpy as np
import gym
import ctm_envs
from stable_baselines import HER
from stable_baselines.common import set_global_seeds
from stable_baselines.her.utils import HERGoalEnvWrapper


# Might need a second hardware env.
# Load up the model end environment.
# During loop, subscribe to current joint position and tip position
# Set to observation of env.
# Run through policy model


class ConcentricTubeRobotPolicyNode(object):
    def __init__(self, env_id, model_path):
        self.env = HERGoalEnvWrapper(gym.make(env_id))
        # Get the policy model
        self.policy_model = HER.load(model_path, env=self.env)
        self.n_tubes = 3
        self.goal_tolerance = 0.001

        # initialise subscribers and publishers
        self.joint_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_callback)
        self.tip_pos_sub = rospy.Subscriber("/tip_pos", Pose, self.tip_pos_callback)
        self.desired_pos_sub = rospy.Subscriber("/desired_pos", Pose, self.desired_pos_callback)

        self.delta_joints_pub = rospy.Publisher("/delta_joints", JointState, queue_size=10)

        self.reset_service = rospy.ServiceProxy("/ctr_reset", Trigger)

        # Initialise joint / tip variables
        self.joint_values = None
        self.tip_pos = None
        self.desired_pos = None

        # Setup a ROS Timer to run policy
        self.policy_timer = rospy.Timer(rospy.Duration(0.5), self.run_policy)

    def joint_state_callback(self, msg):
        betas = msg.position[:self.n_tubes]
        alphas = msg.position[self.n_tubes:]
        joint_value = np.concatenate((betas, alphas))
        self.joint_values = self.joint2rep(joint_value)

    def tip_pos_callback(self, msg):
        self.tip_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

    def desired_pos_callback(self, msg):
        self.desired_pos = np.array([msg.position.x, msg.position.y, msg.position.z])

    def run_policy(self, event):
        if (self.joint_values is not None) and (self.tip_pos is not None) and (self.desired_pos is not None):
            # Get the trig representation and append desired goal
            # Set as state and run through policy
            obs = {"desired_goal": self.desired_pos,
                   "achieved_goal": self.tip_pos,
                   "observation": np.concatenate(
                       (self.joint_values, self.desired_pos - self.tip_pos,
                        np.array([self.goal_tolerance])))}
            # Get action and publish
            action, _ = self.policy_model.predict(obs, deterministic=True)
            delta_joint_msg = JointState()
            delta_joint_msg.position = action
            self.delta_joints_pub.publish(delta_joint_msg)

            # Compute error
            error = np.linalg.norm(self.desired_pos - self.tip_pos)
            if error < self.goal_tolerance:
                print("Reached goal! Resetting...")
                reset_req = TriggerRequest()
                res = self.reset_service(reset_req)
        else:
            if self.joint_values is None:
                rospy.loginfo("waiting for joint state...")
            if self.tip_pos is None:
                rospy.loginfo("waiting for tip pos...")
            if self.desired_pos is None:
                rospy.loginfo("waiting for desired pos...")
            reset_req = TriggerRequest()
            res = self.reset_service(reset_req)

    def rep2joint(self, rep):
        rep = [rep[i:i + 3] for i in range(0, len(rep), 3)]
        beta = np.empty(self.n_tubes)
        alpha = np.empty(self.n_tubes)
        for tube in range(0, self.n_tubes):
            joint = self.single_trig2joint(rep[tube])
            alpha[tube] = joint[0]
            beta[tube] = joint[1]
        return np.concatenate((beta, alpha))

    def joint2rep(self, joint):
        rep = np.array([])
        betas = joint[:self.n_tubes]
        alphas = joint[self.n_tubes:]
        for beta, alpha in zip(betas, alphas):
            trig = self.single_joint2trig(np.array([beta, alpha]))
            rep = np.append(rep, trig)
        return rep

    # Single conversion from a joint to trig representation
    @staticmethod
    def single_joint2trig(joint):
        return np.array([np.cos(joint[1]),
                         np.sin(joint[1]),
                         joint[0]])

    # Single conversion from a trig representation to joint
    @staticmethod
    def single_trig2joint(trig):
        return np.array([np.arctan2(trig[1], trig[0]), trig[2]])


if __name__ == '__main__':
    rospy.init_node("eval_node")
    env_id = "CTR-Reach-v0"
    model_path = "../example_model/cras_exp_6/learned_policy/500000_saved_model.pkl"
    rospy.wait_for_service("/ctr_reset")
    ctr_policy = ConcentricTubeRobotPolicyNode(env_id, model_path)

    rospy.spin()
