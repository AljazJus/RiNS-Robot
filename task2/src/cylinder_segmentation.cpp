#include <iostream>
#include <ros/ros.h>
#include <math.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/features/normal_3d.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include "pcl/point_cloud.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.h"
#include "geometry_msgs/PointStamped.h"
#include <pcl/filters/voxel_grid.h>
#include <nav_msgs/Odometry.h>
#include <cmath>
#include <vector>
#include "task2/Cylinder.h"
#include <sound_play/sound_play.h>

ros::Publisher pubx;
ros::Publisher puby;
ros::Publisher pubm;

ros::Publisher cylinders_pub;

tf2_ros::Buffer tf2_buffer;

typedef pcl::PointXYZRGB PointT;
float leaf_size;
float min_dist = 0.5f;
  
std::vector<task2::Cylinder> cylinders;

float actual_colors[4][3] = {
  {1.0f, 0.09411765f, 0.07058824f},   //red
  {0.03137255f, 1.0f, 0.01176471f},   //green
  {0.227451f, 0.5529412f, 1.0f},      //blue
  {1.0f, 0.9764706f, 0.07058824f}     //yellow
};
float normalized_actual_colors[4][3];
/*
float colors[4][3] ={
  {0.409859f, 0.193088f, 0.187731f},   //red
  {0.204892f, 0.414122f, 0.200939f},   //green
  {0.185734f, 0.212379f, 0.248808f},   //blue
  {0.284085f, 0.281274f, 0.170059f}    //yellow
};
*/
float colors[4][3] ={
  {0.559538, 0.229459, 0.211004},   //red
  {0.193382, 0.631492, 0.175126},   //green
  {0.230513, 0.321747, 0.44774},   //blue
  {0.410963, 0.405192, 0.183845}    //yellow
};
float colors_ideal[4][3] = {
  {1.0f, 0.0f, 0.0f},   //red
  {0.0f, 1.0f, 0.0f},   //green
  {0.0f, 0.0f, 1.0f},   //blue
  {1.0f, 1.0f, 0.0f}    //yellow
};
const char* color_names[4] = {"Red", "Green", "Blue", "Yellow"};

int counter = 0;

void getNormalizedColors() {
  for(int i = 0; i < 4; i++) {
    float sum = 0;
    for(int j = 0; j < 3; j++) {
      sum += actual_colors[i][j];
    }
    for(int j = 0; j < 3; j++) {
      normalized_actual_colors[i][j] = actual_colors[i][j] / sum;
    }
  }
}

int find_best_color(double r, double g, double b) {

  double sum = r + g + b;
  double normal_r = r / sum;
  double normal_g = g / sum;
  double normal_b = b / sum;
  std::cout << "Cylinder color: " << normal_r << ", " << normal_g << ", " << normal_b << std::endl;

  int best_i = -1;
  double smallest_distence = 3;
  for(int i = 0; i < 4; i++) {
    double diff_r = normal_r - colors[i][0];
    double diff_g = normal_g - colors[i][1];
    double diff_b = normal_b - colors[i][2];

    double dist = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
    //double dist = abs(diff_r) + abs(diff_g) + abs(diff_b);
    if(dist < smallest_distence) {
      smallest_distence = dist;
      best_i = i;
    }
  }
  return best_i;
}

void 
cloud_cb (const pcl::PCLPointCloud2ConstPtr& cloud_blob)
{
  // All the objects needed
  //std::cout << "Callback!" << std::endl;
  counter++;
  if(counter == 30) {
    counter = 0;
    std::cout << "cylinder print out" << std::endl;
    for(int i = 0; i < cylinders.size(); i++) {
      task2::Cylinder current = cylinders[i];
      std::cout << i << std::endl;
      std::cout << "Position: x=" << current.x << ", y=" << current.y << std::endl;
      std::cout << "Conviction: " << current.conviction << std::endl;
      std::cout << "Color: " << current.color << std::endl;
      std::cout << "---------" << std::endl;
    }

    //normalized colors
    // std::cout << "Normalized colors";
    for(int i = 0; i < 4; i++) {
      // std::cout << color_names[i] << ": ";
      for(int j = 0; j < 3; j++) {
        // std::cout << normalized_actual_colors[i][j] << " ";
      }
      // std::cout << std::endl;
    }
  }
  ros::Time time_rec, time_test;
  time_rec = ros::Time::now();
  
  pcl::PassThrough<PointT> pass;
  pcl::NormalEstimation<PointT, pcl::Normal> ne;
  pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg; 
  pcl::PCDWriter writer;
  pcl::ExtractIndices<PointT> extract;
  pcl::ExtractIndices<pcl::Normal> extract_normals;
  pcl::search::KdTree<PointT>::Ptr tree (new pcl::search::KdTree<PointT> ());
  Eigen::Vector4f centroid;

  // Datasets
  pcl::PointCloud<PointT>::Ptr cloud (new pcl::PointCloud<PointT>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<PointT>::Ptr cloud_filtered2 (new pcl::PointCloud<PointT>);
  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2 (new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_plane (new pcl::ModelCoefficients), coefficients_cylinder (new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr inliers_plane (new pcl::PointIndices), inliers_cylinder (new pcl::PointIndices);
  

  // Read in the cloud data
  pcl::fromPCLPointCloud2 (*cloud_blob, *cloud);
  //std::cerr << "PointCloud has: " << cloud->points.size () << " data points." << std::endl;

  // Downsample the point cloud
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud(cloud);
  sor.setLeafSize(leaf_size, leaf_size, leaf_size);
  sor.filter(*cloud_filtered);
  //std::cerr << "PointCloud after downsampling has: " << cloud_filtered->points.size() << " data points." << std::endl;

  // Build a passthrough filter to remove spurious NaNs
  pass.setInputCloud (cloud_filtered);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0, 1.5);
  pass.filter (*cloud_filtered);
  //std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size () << " data points." << std::endl;

  // Estimate point normals
  ne.setSearchMethod (tree);
  ne.setInputCloud (cloud_filtered);
  ne.setKSearch (50);
  ne.compute (*cloud_normals);

  // Create the segmentation object for the planar model and set all the parameters
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_NORMAL_PLANE);
  seg.setNormalDistanceWeight (0.1);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (100);
  seg.setDistanceThreshold (0.03);
  seg.setInputCloud (cloud_filtered);
  seg.setInputNormals (cloud_normals);
  // Obtain the plane inliers and coefficients
  seg.segment (*inliers_plane, *coefficients_plane);
  //std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;

  // Extract the planar inliers from the input cloud
  extract.setInputCloud (cloud_filtered);
  extract.setIndices (inliers_plane);
  extract.setNegative (false);

  // Write the planar inliers to disk
  pcl::PointCloud<PointT>::Ptr cloud_plane (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_plane);
  //std::cerr << "PointCloud representing the planar component: " << cloud_plane->points.size () << " data points." << std::endl;
  
  pcl::PCLPointCloud2 outcloud_plane;
  pcl::toPCLPointCloud2 (*cloud_plane, outcloud_plane);
  pubx.publish (outcloud_plane);

  // Remove the planar inliers, extract the rest
  extract.setNegative (true);
  extract.filter (*cloud_filtered2);
  extract_normals.setNegative (true);
  extract_normals.setInputCloud (cloud_normals);
  extract_normals.setIndices (inliers_plane);
  extract_normals.filter (*cloud_normals2);

  // Create the segmentation object for cylinder segmentation and set all the parameters
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_CYLINDER);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setNormalDistanceWeight (0.1);
  seg.setMaxIterations (10000);
  seg.setDistanceThreshold (0.02);
  seg.setRadiusLimits (0.11, 0.13);
  seg.setInputCloud (cloud_filtered2);
  seg.setInputNormals (cloud_normals2);
  seg.segment (*inliers_cylinder, *coefficients_cylinder);
  //std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;

  // Write the cylinder inliers to disk
  extract.setInputCloud (cloud_filtered2);
  extract.setIndices (inliers_cylinder);
  extract.setNegative (false);
  pcl::PointCloud<PointT>::Ptr cloud_cylinder (new pcl::PointCloud<PointT> ());
  extract.filter (*cloud_cylinder);
  if (cloud_cylinder->points.empty ()) {
    //std::cerr << "Can't find the cylindrical component." << std::endl;
  }
  else
  {
	  //std::cerr << "PointCloud representing the cylindrical component: " << cloud_cylinder->points.size () << " data points." << std::endl;
    
    if(cloud_cylinder->points.size() < 300) return;

    static sound_play::SoundClient sc;
    pcl::compute3DCentroid (*cloud_cylinder, centroid);
    //std::cerr << "centroid of the cylindrical component: " << centroid[0] << " " <<  centroid[1] << " " <<   centroid[2] << " " <<   centroid[3] << std::endl;

	  //Create a point in the "camera_rgb_optical_frame"
    geometry_msgs::PointStamped point_camera;
    geometry_msgs::PointStamped point_map;
    visualization_msgs::Marker marker;
    geometry_msgs::TransformStamped tss;
          
    point_camera.header.frame_id = "camera_rgb_optical_frame";
    point_camera.header.stamp = ros::Time::now();

    point_map.header.frame_id = "map";
    point_map.header.stamp = ros::Time::now();

    point_camera.point.x = centroid[0];
    point_camera.point.y = centroid[1];
    point_camera.point.z = centroid[2];

	  try{
		  time_test = ros::Time::now();

		  std::cerr << time_rec << std::endl;
		  std::cerr << time_test << std::endl;
  	      tss = tf2_buffer.lookupTransform("map","camera_rgb_optical_frame", time_rec);
          //tf2_buffer.transform(point_camera, point_map, "map", ros::Duration(2));
	  } catch (tf2::TransformException &ex) {
	       ROS_WARN("Transform warning: %s\n", ex.what());
	  }

      //std::cerr << tss ;

      tf2::doTransform(point_camera, point_map, tss);

      //std::cerr << "point_camera: " << point_camera.point.x << " " <<  point_camera.point.y << " " <<  point_camera.point.z << std::endl;

      //std::cerr << "point_map: " << point_map.point.x << " " <<  point_map.point.y << " " <<  point_map.point.z << std::endl;

      auto odom_msg = ros::topic::waitForMessage<nav_msgs::Odometry>("/odom",ros::Duration(0.2f));

      if (odom_msg != nullptr) {
          //ROS_INFO("Received Odometry message");
          
      } else {
          ROS_WARN("Timed out waiting for Odometry message");
      }

      marker.header.frame_id = "map";
      marker.header.stamp = ros::Time::now();

      marker.ns = "cylinder";
      marker.id = cylinders.size();

      marker.type = visualization_msgs::Marker::CYLINDER;
      marker.action = visualization_msgs::Marker::ADD;

      marker.pose.position.x = point_map.point.x;
      marker.pose.position.y = point_map.point.y;
      marker.pose.position.z = 0.22;

      float move_dist = 0.12;
      if (odom_msg != nullptr) {
        float cyl_x = point_map.point.x - odom_msg->pose.pose.position.x;
        float cyl_y = point_map.point.y - odom_msg->pose.pose.position.y;
        float size = sqrt(cyl_x * cyl_x + cyl_y * cyl_y);
        cyl_x = (cyl_x / size) * move_dist;
        cyl_y = (cyl_y / size) * move_dist;
        marker.pose.position.x += cyl_x;
        marker.pose.position.y += cyl_y;
      }



      marker.pose.orientation.x = 0.0;
      marker.pose.orientation.y = 0.0;
      marker.pose.orientation.z = 0.0;
      marker.pose.orientation.w = 1.0;

      marker.scale.x = 0.24;
      marker.scale.y = 0.24;
      marker.scale.z = 0.44;

      int color_r = 0;
      int color_g = 0;
      int color_b = 0;

      int size = cloud_cylinder->points.size();
      int counter = 0;
      //std::cout << "cloud cylinder size: " << s << std::endl;
      for(int i = 0; i < size; i++) {
        pcl::PointXYZRGB point = cloud_cylinder->points[i];
        if((float)point.z >= 0.22) {
          color_r += (int)point.r;
          color_g += (int)point.g;
          color_b += (int)point.b;
          counter++;
        }
      }
      double avg_color_r = ((double)color_r) / (255 * counter);
      double avg_color_g = ((double)color_g) / (255 * counter);
      double avg_color_b = ((double)color_b) / (255 * counter);

      int best_color_index = find_best_color(avg_color_r, avg_color_g, avg_color_b);
      marker.color.r = colors_ideal[best_color_index][0];
      marker.color.g = colors_ideal[best_color_index][1];
      marker.color.b = colors_ideal[best_color_index][2];
      marker.color.a = 1.0f;
      /*
      for(int i = 0; i < cloud_cylinder->points.size(); i++) {
        pcl::PointXYZRGB point = cloud_cylinder->points[i];
        std::cout << "red: " << (uint8_t)point.r << std::endl;
      }
      marker.color.r = 1.0f;
      marker.color.g = 0.0f;
      marker.color.b = 0.0f;
      marker.color.a = 1.0f;
      */
      marker.lifetime = ros::Duration();

      bool new_cyl = true;
      for(int i = 0; i < cylinders.size(); i++) {
        task2::Cylinder& point = cylinders[i];
        float x_cyl = point.x;
        float y_cyl = point.y;
        float x_new = marker.pose.position.x;
        float y_new = marker.pose.position.y;
        if(abs(x_cyl - x_new) < min_dist && abs(y_cyl - y_new) < min_dist) {
          if(size > point.conviction) {
            std::cout << "UPDATED \n UPDTATED \n UPDATED" << std::endl;
            std::cout << size << " " << point.conviction << std::endl; 
            point.x = x_new;
            point.y = y_new;
            point.conviction = size;
            point.color = color_names[best_color_index];

            marker.action = visualization_msgs::Marker::MODIFY;
            marker.id = i;
            cylinders_pub.publish(marker);
          }
          new_cyl = false;
          break;
        }
      }
      if(new_cyl) {
        task2::Cylinder newCylinder;
        newCylinder.x = marker.pose.position.x;
        newCylinder.y = marker.pose.position.y;
        newCylinder.conviction = size;
        
        newCylinder.color = color_names[best_color_index];
        cylinders.push_back(newCylinder);

        cylinders_pub.publish(marker);
        std::cout << "New cylinder" << std::endl;
        
        //say the color of the cylinder
        sc.say(newCylinder.color + " cylinder");
      }


      //pubm.publish (marker);

      pcl::PCLPointCloud2 outcloud_cylinder;
        pcl::toPCLPointCloud2 (*cloud_cylinder, outcloud_cylinder);
        puby.publish (outcloud_cylinder);

  }
  //std::cout << "cylinder size: " << cylinders.size() << std::endl;
  
}

int
main (int argc, char** argv)
{
  //std::cout << "Starting cylinder_segmentation";
  // Initialize ROS
  ros::init (argc, argv, "cylinder_segment");
  ros::NodeHandle nh;

  getNormalizedColors();



  if (!nh.getParam("leaf_size", leaf_size))
  {
    ROS_ERROR("Failed to retrieve parameter 'leaf_size'");
    return 1;
  }

  // For transforming between coordinate frames
  tf2_ros::TransformListener tf2_listener(tf2_buffer);

  // Create a ROS subscriber for the input point cloud
  ros::Subscriber sub = nh.subscribe ("input", 1, cloud_cb);

  // Create a ROS publisher for the output point cloud
  pubx = nh.advertise<pcl::PCLPointCloud2> ("planes", 1);
  puby = nh.advertise<pcl::PCLPointCloud2> ("cylinder", 1);

  //pubm = nh.advertise<visualization_msgs::Marker>("detected_cylinder",1);

  cylinders_pub = nh.advertise<visualization_msgs::Marker>("cyliders", 20);
  // Spin
  ros::spin ();
}
