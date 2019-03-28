cmake_minimum_required(VERSION 3.1)

project(TMA4280_Project_2)

enable_language(Fortran)
enable_language(CXX)

#SET(CMAKE_CXX_COMPILER "g++-8")
#add_compile_options(-std=c++11 -O3)

find_package(MPI REQUIRED)
find_package(OpenMP)

set(CMAKE_C_FLAGS "-std=c99 ${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3" ${CMAKE_CXX_FLAGS})

if(MPI_FOUND)
  add_definitions(-DHAVE_MPI=1)
  set(INCLUDES ${INCLUDES} ${MPI_INCLUDE_PATH})
  set(DEPLIBS_C ${DEPLIBS_C} ${MPI_C_LIBRARIES})
  set(DEPLIBS_CXX ${DEPLIBS_CXX} ${MPI_CXX_LIBRARIES})
  set(DEPLIBS_F ${DEPLIBS_F} ${MPI_Fortran_LIBRARIES})
endif()

if(OPENMP_FOUND)
  add_definitions(-DHAVE_OPENMP=1)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -openmp")
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(${INCLUDES})

add_library(common fst.f)

add_executable(poisson main.cpp utils.cpp)
add_executable(poissonC poisson.c)

target_link_libraries(poisson common ${DEPLIBS_CXX})
target_link_libraries(poissonC common ${DEPLIBS_C})
