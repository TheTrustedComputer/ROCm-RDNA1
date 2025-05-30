
cmake_minimum_required( VERSION 3.16.3 )

project( rocm-core LANGUAGES CXX )

set( CMAKE_BUILD_TYPE Release )
set( CMAKE_INSTALL_PREFIX /opt/rocm CACHE STRING "install dir" )

set( PROJECT_VERSION_MAJOR 5 CACHE STRING "major version" )
set( PROJECT_VERSION_MINOR 4 CACHE STRING "minor version" )
set( PROJECT_VERSION_PATCH 3 CACHE STRING "patch version" )
set( ROCM_PATCH_VERSION 50403 CACHE STRING "rocm patch version" )
set( ROCM_BUILD_VERSION 121 CACHE STRING "rocm build version" )
set( CPACK_GENERATOR DEB CACHE STRING "cpack generator" )

configure_file( .info/version.in .info/version )
configure_file( include/rocm_version.h.in include/rocm_version.h)

install(FILES "${PROJECT_BINARY_DIR}/include/rocm_version.h"
  DESTINATION include
  )

install(FILES "${PROJECT_BINARY_DIR}/.info/version"
  DESTINATION .info
  )

set( CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH} )

set( CPACK_PACKAGE_NAME ${PROJECT_NAME} )
set( CPACK_PACKAGE_VENDOR "Advanced Micro Devices, Inc" )
set( CPACK_DEBIAN_PACKAGE_MAINTAINER amd )
set( CPACK_DEBIAN_PACKAGE_SECTION "devel" )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "Radeon Open Compute (ROCm) Runtime software stack" )
set( CPACK_PACKAGE_VERSION ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}.${ROCM_PATCH_VERSION}-${ROCM_BUILD_VERSION} )
set( CPACK_GENERATOR ${CPACK_GENERATOR} )
set( CPACK_DEBIAN_PACKAGE_ARCHITECTURE amd64 )
set( CPACK_DEBIAN_PACKAGE_DEPENDS "" )
set( CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} )
set( CPACK_DEBIAN_FILE_NAME "DEB-DEFAULT" )

include( CPack )
