#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <Eigen/Geometry>
#include <fstream>
#include <iostream>
#include <vector>

#include "nlohmann/json.hpp"

using namespace pcl;
using namespace std;
using json = nlohmann::json;

const string data_root = "/home/kemo/Dataset/sustechscapes-mini-dataset";

int main(int argc, char **argv) {
    const string frame = argv[1];
    const string pcd_filename = data_root + "/lidar/" + frame + ".pcd";
    const string label_filename = data_root + "/label/" + frame + ".json";
    // load point cloud
    pcl::PointCloud<PointXYZI>::Ptr points(new pcl::PointCloud<PointXYZI>);
    if (io::loadPCDFile<PointXYZI>(pcd_filename, *points) == -1) {
        PCL_ERROR("Couldn't read pcd file \n");
        return -1;
    }
    string label_string;
    ifstream label_file(label_filename);
    if (!getline(label_file, label_string)) {
        cout << "Couldn't read label file \n";
        return -1;
    }
    json label_json = json::parse(label_string);

    // pcl visualizer setting
    pcl::visualization::PCLVisualizer *viewer =
        new pcl::visualization::PCLVisualizer("3D Viewer");
    viewer->setBackgroundColor(0, 0, 0);
    // point cloud color
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> color(
        points, 0, 150, 0);
    viewer->addPointCloud<pcl::PointXYZI>(points, color, "point cloud");
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 0.8, "point cloud");
    viewer->initCameraParameters();
    viewer->setShowFPS(false);
    // set camera position
    viewer->setCameraPosition(4.36873, 49.8423, 23.4606, 0.43884, 2.01171,
                              -2.91852, -0.00551891, -0.482579, 0.875835);
    int i = 0;
    for (json label : label_json) {
        if (label["obj_type"] == "Car") {
            json psr = label["psr"];
            json position = psr["position"];
            json scale = psr["scale"];
            json rotation = psr["rotation"];
            Eigen::Quaternionf quaternion =
                Eigen::AngleAxisf(rotation["x"], Eigen::Vector3f::UnitX()) *
                Eigen::AngleAxisf(rotation["y"], Eigen::Vector3f::UnitY()) *
                Eigen::AngleAxisf(rotation["z"], Eigen::Vector3f::UnitZ());
            string box_id = "box" + to_string(i);
            viewer->addCube(
                Eigen::Vector3f(position["x"], position["y"], position["z"]),
                quaternion, scale["x"], scale["y"], scale["z"], box_id);
            viewer->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                box_id);
            viewer->setShapeRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0, 0, box_id);

            ++i;
        }
    }

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
    }
    return 0;
}
