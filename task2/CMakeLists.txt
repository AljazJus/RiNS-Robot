cmake_minimum_required(VERSION 2.8.3)
project(task2)

## Find system dependencies
find_package(Boost REQUIRED COMPONENTS system filesystem thread)
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)

## Find catkin packages
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  genmsg
  nodelet
  nodelet_topic_tools
  pcl_conversions
  pcl_msgs
  pluginlib
  rosbag
  rosconsole
  roscpp
  roslib
  sensor_msgs
  std_msgs
  rospy
  geometry_msgs
  tf2_ros
  tf2_geometry_msgs
  message_generation
  message_runtime
)

## Add include directories
include_directories(
	${Boost_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Add link directories
link_directories(
  ${Boost_LIBRARY_DIRS}
  ${Eigen_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
  ${catkin_LIBRARY_DIRS}
)

link_libraries(${Boost_LIBRARIES} ${catkin_LIBRARIES} ${PCL_LIBRARIES})

add_message_files(DIRECTORY msg
  FILES
  Cylinder.msg
  Ring.msg
)

add_service_files(
  FILES
  ImageRecognition.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

## Declare the catkin package
catkin_package(
  LIBRARIES
    pcl_ros_filters
    pcl_ros_io
    pcl_ros_tf
  CATKIN_DEPENDS
    dynamic_reconfigure
    pcl_conversions
    pcl_msgs
    roscpp
    rospy
    sensor_msgs
    std_msgs
    message_runtime
    message_generation
  DEPENDS
    Boost
    Eigen
    PCL
)

add_executable (cylinder_segmentation_task2 src/cylinder_segmentation.cpp)

catkin_install_python(PROGRAMS scripts/main.py scripts/detect_rings.py scripts/park_control.py scripts/image_recognition.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
