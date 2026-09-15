cmake_minimum_required(VERSION 3.18)
project(sage_native_fft_baseline LANGUAGES CXX)
if(NOT ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_CANN_PACKAGE_PATH}")
endif()
find_path(CANN_INCLUDE acl/acl.h PATHS
  "${ASCEND_CANN_PACKAGE_PATH}/include"
  "${ASCEND_CANN_PACKAGE_PATH}/runtime/include" REQUIRED NO_DEFAULT_PATH)
foreach(component ascendcl acl_op_compiler)
  find_library(CANN_${component} ${component} PATHS
    "${ASCEND_CANN_PACKAGE_PATH}/lib64"
    "${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64" REQUIRED NO_DEFAULT_PATH)
endforeach()
foreach(program native_fft_pipeline native_fft_builtin_norm)
  add_executable(${program} ${program}.cpp)
  target_compile_features(${program} PRIVATE cxx_std_17)
  target_compile_options(${program} PRIVATE -O2)
  target_include_directories(${program} PRIVATE "${CANN_INCLUDE}")
  target_link_libraries(${program} PRIVATE ${CANN_ascendcl} ${CANN_acl_op_compiler} ${CMAKE_DL_LIBS})
endforeach()
