geometry_msgs/PoseStamped grasp_pose  # The grasp pose
string eef_frame_id                   # The coordinate frame of the end-effector used by this grasp
uint32 grasp_id
uint32 successes                      # The number of recorded times this grasp was successful (useful for success rate)
uint32 attempts                       # The number of recorded times this grasp was attempted (useful for success rate)
