cmake_minimum_required(VERSION 3.9)
project(DVDeep)

#build configuration
add_compile_options(-W -Wall)

#config for debug
set(CMAKE_CXX_FLAGS_DEBUG "-g")

#config for release
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

#C++ standard
set(CMAKE_CXX_STANDARD 17)

#common
set(common_headers include/Common.h)

#math
set(math_sources src/math/Matrix.cpp)
set(math_headers include/math/Matrix.h)

#layer
set(layer_sources src/layer/Layer.cpp
        src/layer/Affine.cpp
        src/layer/Activation.cpp
        src/layer/Output.cpp)
set(layer_headers include/layer/Layer.h
        include/layer/Affine.h
        include/layer/Activation.h
        include/layer/Output.h)

#network
set(network_sources src/network/FFN.cpp)
set(network_headers include/network/FFN.h)

#summarize
set(HEADER ${common_headers}
        ${math_headers}
        ${layer_headers}
        ${network_headers})
set(SOURCE ${math_sources}
        ${layer_sources}
        ${network_sources})
set(SRC ${SOURCE} ${HEADER})

#make executable and static library
add_library(dvdeeplib ${SRC})
add_executable(dvdeep main.cpp)
target_link_libraries(dvdeep dvdeeplib)
set_target_properties(dvdeeplib PROPERTIES OUTPUT_NAME dvdeep)