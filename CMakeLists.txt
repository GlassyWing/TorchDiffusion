cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(diffusion)
set(CMAKE_CXX_STANDARD 17)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
#find_package(TorchVision REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB ALL_SOURCES "*.cpp" "*.hpp" "./src/*.cpp" "./src/*/*.cpp" "./src/*/*.h" "./src/*.h")
add_executable(diffusion main.cpp ${ALL_SOURCES})
target_compile_features(diffusion PUBLIC cxx_range_for)
target_link_libraries(diffusion ${TORCH_LIBRARIES})
target_link_libraries(diffusion ${OpenCV_LIBS} )
#target_link_libraries(diffusion TorchVision::TorchVision)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET diffusion
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:diffusion>)
endif (MSVC)
