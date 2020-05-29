#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;
namespace fs = boost::filesystem;

fs::path data_root("/home/kemo/Dataset/sustechscapes-mini-dataset");

string convert_to_kitti_calib(string in_filename) {
    ifstream in_file(in_filename);
    string in_content;
    if (in_file.is_open()) {
        in_content.assign(istreambuf_iterator<char>(in_file),
                          istreambuf_iterator<char>());
        in_file.close();
        json calib_json = json::parse(in_content);
        stringstream ss;
        ss << "P0: 0 0 0 0 0 0 0 0 0 0 0 0\n";
        ss << "P1: 0 0 0 0 0 0 0 0 0 0 0 0\n";
        ss << "P2: ";
        json intrinsic = calib_json["intrinsic"];
        ss << intrinsic[0] << " ";
        ss << intrinsic[1] << " ";
        ss << intrinsic[2] << " ";
        ss << "0.0 ";
        ss << intrinsic[3] << " ";
        ss << intrinsic[4] << " ";
        ss << intrinsic[5] << " ";
        ss << "0.0 ";
        ss << intrinsic[6] << " ";
        ss << intrinsic[7] << " ";
        ss << intrinsic[8] << " ";
        ss << "0.0 ";
        ss << endl;
        ss << "P3: 0 0 0 0 0 0 0 0 0 0 0 0\n";
        ss << "R0_rect: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n";
        ss << "Tr_velo_to_cam: ";
        for (auto v : calib_json["extrinsic"]) {
            ss << v << " ";
        }
        ss << endl;
        ss << "Tr_imu_to_velo: 0 0 0 0 0 0 0 0 0 0 0 0\n";
        return ss.str();
    } else {
        return NULL;
    }
}

void pcd2bin(const string in_filename, const string out_file) {
    // Create a PointCloud value
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZI>);

    // Open the PCD file
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(in_filename, *cloud) == -1) {
        PCL_ERROR("Couldn't read in_file\n");
    }
    // Create & write .bin file
    ofstream bin_file(out_file.c_str(), ios::out | ios::binary | ios::app);
    if (!bin_file.good()) cout << "Couldn't open " << out_file << endl;

    // PCD 2 BIN
    for (size_t i = 0; i < cloud->points.size(); ++i) {
        bin_file.write((char*)&cloud->points[i].x, 3 * sizeof(float));
        bin_file.write((char*)&cloud->points[i].intensity, sizeof(float));
    }
    bin_file.close();
}

int main(int argc, char const* argv[]) {
    cout << "Converting sustechscape to kitti format." << endl;

    fs::path image_src_dir = data_root / fs::path("camera/front");
    fs::path lidar_src_dir = data_root / fs::path("lidar");

    fs::path result_root = data_root / fs::path("kitti/training");
    fs::path image_dst_dir = result_root / fs::path("image_2");
    fs::path calib_dst_dir = result_root / fs::path("calib");
    fs::path lidar_dst_dir = result_root / fs::path("velodyne");

    if (!fs::exists(image_dst_dir)) {
        fs::create_directories(image_dst_dir);
    }

    if (!fs::exists(calib_dst_dir)) {
        fs::create_directories(calib_dst_dir);
    }

    if (!fs::exists(lidar_dst_dir)) {
        fs::create_directories(lidar_dst_dir);
    }

    fs::path calib_src = data_root / fs::path("calib/camera/front.json");
    string kitti_calib = convert_to_kitti_calib(calib_src.string());

    for (fs::directory_iterator itr(image_src_dir);
         itr != fs::directory_iterator(); ++itr) {
        string filename = itr->path().filename().string();
        string frame = filename.substr(0, filename.size() - 4);
        fs::path calib_dst(calib_dst_dir / fs::path(frame + ".txt"));
        fs::path image_dst(image_dst_dir / fs::path(frame + ".png"));
        fs::path lidar_dst(lidar_dst_dir / fs::path(frame + ".bin"));

        if (!fs::exists(calib_dst)) {
            ofstream out_file(calib_dst.string());
            out_file << kitti_calib;
            out_file.close();
        }

        if (!fs::exists(image_dst)) {
            cv::imwrite(image_dst.string(), cv::imread(itr->path().string()));
        }

        if (!fs::exists(lidar_dst)) {
            fs::path lidar_src = lidar_src_dir / fs::path(frame + ".pcd");
            pcd2bin(lidar_src.string(), lidar_dst.string());
        }
    }
    return 0;
}
