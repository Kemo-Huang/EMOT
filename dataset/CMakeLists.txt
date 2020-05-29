cmake_minimum_required( VERSION 3.0 )

set (CMKAE_CXX_STANDARD 11)

# pcl 
find_package(PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# boost
find_package(Boost REQUIRED)
include_directories( ${Boost_INCLUDE_DIR})
link_directories(${Boost_LIBRARY_DIR})

# opencv
find_package( OpenCV REQUIRED )

add_executable(sustechscape_to_kitti sustechscape_to_kitti.cpp)

target_link_libraries(sustechscape_to_kitti ${PCL_LIBRARIES})
target_link_libraries(sustechscape_to_kitti ${Boost_LIBRARIES})
target_link_libraries(sustechscape_to_kitti ${OpenCV_LIBS})
