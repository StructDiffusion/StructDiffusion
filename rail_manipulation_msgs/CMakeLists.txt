cmake_minimum_required(VERSION 2.8.3)
project(rail_manipulation_msgs)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  geometry_msgs
  message_generation
  sensor_msgs
  visualization_msgs
)

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(FILES
  BoundingVolume.msg
  Grasp.msg
  GraspingState.msg
  GraspFeedback.msg
  SegmentedObject.msg
  SegmentedObjectList.msg
)

## Generate services in the 'srv' folder
add_service_files(FILES
  CallIK.srv
  CartesianPath.srv
  PairwiseRank.srv
  PrepareGrasp.srv
  ProcessSegmentedObjects.srv
  SegmentObjects.srv
  SegmentObjectsFromPointCloud.srv
  SuggestGrasps.srv
  RearrangeObjects.srv
)


## Generate actions in the 'action' folder
add_action_files(FILES
  Arm.action
  Gripper.action
  Lift.action
  MoveToJointPose.action
  MoveToPose.action
  Pickup.action
  Primitive.action
  RecognizeObject.action
  Store.action
  VerifyGrasp.action
)

generate_messages(DEPENDENCIES
  actionlib_msgs
  geometry_msgs
  sensor_msgs
  visualization_msgs
)

###################################################
## Declare things to be passed to other projects ##
###################################################

## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  CATKIN_DEPENDS actionlib_msgs geometry_msgs sensor_msgs visualization_msgs
)
