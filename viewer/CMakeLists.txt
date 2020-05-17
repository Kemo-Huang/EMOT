cmake_minimum_required( VERSION 3.0 )

set (CMKAE_CXX_STANDARD 11)

# pcl 
find_package(PCL REQUIRED)
include_directories( ${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(viewer3d viewer3d.cpp)
target_link_libraries( viewer3d ${PCL_LIBRARIES}  )
