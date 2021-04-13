import rospy
import rospkg

from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from std_srvs.srv import Trigger, TriggerResponse

from stable_baselines.her.utils import HERGoalEnvWrapper
import numpy as np
import gym
import ctm_envs


# Hardware simulation class for testing the eval_agent
# Receive delta joints (action) JointState message, step through environment and republish new observation

class ConcentricTubeRobotSimNode(object):
    def __init__(self, env_id):
        self.env = HERGoalEnvWrapper(gym.make(env_id))
        self.n_tubes = 3
        self.goal_tolerance = 0.001

        # Subscribe to actions
        self.delta_joints_pub = rospy.Subscriber("/delta_joints", JointState, self.action_callback)

        # Publish joints, tip pos, desired_pos
        self.joint_sample_pub = rospy.Publisher("/joint_states", JointState, queue_size=10)
        self.tip_pos_pub = rospy.Publisher("/tip_pos", Pose, queue_size=10)
        self.desired_pos_pub = rospy.Publisher("/desired_pos", Pose, queue_size=10)

        # Reset service
        reset_srv = rospy.Service("/ctr_reset", Trigger, self.trigger_reset)

    def trigger_reset(self, request):
        self.reset_robot()
        return TriggerResponse(success=True, message="env has been reset.")

    # Send srv request to reset
    def reset_robot(self):
        obs = self.env.reset()
        self.publish_obs(obs)

    def action_callback(self, msg):
        action = msg.position
        obs, reward, done, info = self.env.step(action)
        print("error: ", info["errors_pos"])
        print("done: ", done)
        self.publish_obs(obs)

    def publish_obs(self, obs):
        obs_dict = self.env.convert_obs_to_dict(obs)

        joint_msg = JointState()
        joint_msg.position = self.rep2joint(obs_dict["observation"][:3 * self.n_tubes])
        self.joint_sample_pub.publish(joint_msg)

        # Tip pos
        achieved_tip_pos = Pose()
        achieved_tip_pos.position.x = obs_dict["achieved_goal"][0]
        achieved_tip_pos.position.y = obs_dict["achieved_goal"][1]
        achieved_tip_pos.position.z = obs_dict["achieved_goal"][2]
        self.tip_pos_pub.publish(achieved_tip_pos)

        desired_tip_pos = Pose()
        desired_tip_pos.position.x = obs_dict["desired_goal"][0]
        desired_tip_pos.position.y = obs_dict["desired_goal"][1]
        desired_tip_pos.position.z = obs_dict["desired_goal"][2]
        self.desired_pos_pub.publish(desired_tip_pos)

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
    rospy.init_node("ctr_sim")
    env_id = "CTR-Reach-v0"
    ctr_sim = ConcentricTubeRobotSimNode(env_id)
    rospy.spin()
