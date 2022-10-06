^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package rail_manipulation_msgs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.0.14 (2019-06-27)
-------------------
* Added new srv file for segmenting objects, which allows for passing a pointcloud in the request
* Added indices of the object in the organized pc to SegementedObject.msg
* Contributors: Weiyu Liu

0.0.13 (2019-06-07)
-------------------
* New service message, takes a segmented object list in, returns a segmented object list after some processing has been done
* Contributors: David Kent

0.0.12 (2018-09-21)
-------------------
* Added better bounding volumes and some other features to segmented object message, added alternative API for object segmentation that acts more like a service
* Contributors: David Kent

0.0.11 (2017-08-01)
-------------------
* Added new service used in grasp suggestion and ranking
* Contributors: David Kent

0.0.10 (2017-07-18)
-------------------
* Merge pull request `#2 <https://github.com/GT-RAIL/rail_manipulation_msgs/issues/2>`_ from velveteenrobot/develop
  Added new msg and srv to facilitate grasp suggestion
* Added new msg and srv to facilitate grasp suggestion
* Contributors: David Kent, Sarah Elliott

0.0.9 (2017-01-11)
------------------
* Added optional flag for attaching object collision model after pickup
* Updated Cartesian Path service to use stamped poses
* Primitive action for simple Cartesian actions
* added grasp id to grasp messages
* Added joint state difference to optionally prevent large changes in joint configuration
* Contributors: Aaron St. Clair, David Kent, Russell Toris

0.0.8 (2016-02-19)
------------------
* Update README.md
* Update package.xml
* Added grasping state message
* Consolidated some messages with carl_moveit for more general use
* Added (optional) speed and force parameters to the gripper goal
* Contributors: David Kent

0.0.7 (2015-04-27)
------------------
* Merge branch 'develop' of github.com:WPI-RAIL/rail_manipulation_msgs into develop
* Included success rate data with grasps
* Contributors: David Kent

0.0.6 (2015-04-22)
------------------
* cleared flag added
* Contributors: Russell Toris

0.0.5 (2015-04-14)
------------------
* changelog updated
* Added center to segmented object message
* Contributors: David Kent, Russell Toris

0.0.4 (2015-04-10)
------------------
* bounding box info added
* Contributors: Russell Toris

0.0.3 (2015-03-31)
------------------
* orientation added
* removed old recognize actions
* Contributors: Russell Toris

0.0.2 (2015-03-24)
------------------
* Refactored recognize actions
* Merge branch 'develop' of github.com:WPI-RAIL/rail_manipulation_msgs into develop
* Added recognition actions
* Contributors: David Kent

0.0.1 (2015-03-24)
------------------
* added image to segmented
* added marker and centroid to objects
* added readmes and such
* initial messages
* Contributors: Russell Toris
