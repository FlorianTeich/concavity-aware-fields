cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project(ConcavityAwareFields)

set(ENV{CC} gcc-7)
set(ENV{CXX} g++-7)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif(CCACHE_FOUND)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set( CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake )

set(BUILD_TESTING OFF CACHE BOOL "Build the testing tree.")

SET(CMAKE_CXX_FLAGS -pthread)

option(LIBIGL_USE_STATIC_LIBRARY     "Use libigl as static library" OFF)
option(LIBIGL_WITH_ANTTWEAKBAR       "Use AntTweakBar"    OFF)
option(LIBIGL_WITH_CGAL              "Use CGAL"           ON)
option(LIBIGL_WITH_COMISO            "Use CoMiso"         OFF)
option(LIBIGL_WITH_CORK              "Use Cork"           OFF)
option(LIBIGL_WITH_EMBREE            "Use Embree"         ON)
option(LIBIGL_WITH_LIM               "Use LIM"            OFF)
option(LIBIGL_WITH_MATLAB            "Use Matlab"         OFF)
option(LIBIGL_WITH_MOSEK             "Use MOSEK"          OFF)
option(LIBIGL_WITH_OPENGL            "Use OpenGL"         ON)
option(LIBIGL_WITH_OPENGL_GLFW       "Use GLFW"           ON)
option(LIBIGL_WITH_OPENGL_GLFW_IMGUI "Use IMGUI"          ON)
option(LIBIGL_WITH_PNG               "Use PNG"            ON)
option(LIBIGL_WITH_PYTHON            "Use Python"         OFF)
option(LIBIGL_WITH_TETGEN            "Use Tetgen"         OFF)
option(LIBIGL_WITH_TRIANGLE          "Use Triangle"       OFF)
option(LIBIGL_WITH_VIEWER            "Use OpenGL viewer"  ON)
option(LIBIGL_WITH_IMGUI             "Use imgui menu"     ON)
option(LIBIGL_WITH_XML               "Use XML"            OFF)

option(BUILD_DOC "Build documentation" ON)

set(OpenGL_GL_PREFERENCE "LEGACY")

set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Boost COMPONENTS program_options REQUIRED)
find_package(PCL 1.10 EXACT REQUIRED)
find_package(VTK REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(X11 REQUIRED)
find_package(SPQR REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(GTK2 2.6 REQUIRED gtk)
find_package(CGAL REQUIRED)
find_package(CGAL REQUIRED COMPONENTS Core)
find_package(LIBIGL REQUIRED)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data DESTINATION ${CMAKE_BINARY_DIR})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/docs)
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)

#add_custom_target(Sphinx ALL
#        COMMAND
#        ${SPHINX_EXECUTABLE} -b html
#        ${SPHINX_SOURCE} ${SPHINX_BUILD}
#        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
#        COMMENT "Generating documentation with Sphinx")
#set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
#set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/docs/sphinx)


option(BUILD_TESTS "Build Tests" OFF)
if (BUILD_TESTS)
    # Download and unpack googletest at configure time
    configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
    execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "CMake step for googletest failed: ${result}")
    endif()
    execute_process(COMMAND ${CMAKE_COMMAND} --build .
            RESULT_VARIABLE result
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
    if(result)
        message(FATAL_ERROR "Build step for googletest failed: ${result}")
    endif()

    # Prevent overriding the parent project's compiler/linker
    # settings on Windows
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(gtest_color "1")

    # Add googletest directly to our build. This defines
    # the gtest and gtest_main targets.
    add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
            ${CMAKE_BINARY_DIR}/googletest-build
            EXCLUDE_FROM_ALL)

    # The gtest/gtest_main targets carry header search path
    # dependencies automatically when using CMake 2.8.11 or
    # later. Otherwise we have to add them here ourselves.
    if (CMAKE_VERSION VERSION_LESS 2.8.11)
        include_directories("${gtest_SOURCE_DIR}/include")
    endif()

    # Now simply link against gtest or gtest_main as needed. Eg
    add_executable(unit_test src/tests/common_tests.cpp)
    target_link_libraries(unit_test ${Boost_SYSTEM_LIBRARY} gtest_main igl::core)
    add_test(NAME test_all COMMAND unit_test)

endif(BUILD_TESTS)

if(False)
if (DOXYGEN_FOUND)
    find_program(LATEX pdflatex)

    # set input and output files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)
    message("Doxygen build started")

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( doc_doxygen ALL
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Generating API documentation with Doxygen"
            VERBATIM )


    # Find all the public headers
    #get_target_property(CAT_CUTIFIER_PUBLIC_HEADER_DIR CatCutifier INTERFACE_INCLUDE_DIRECTORIES)
    #file(GLOB_RECURSE CAT_CUTIFIER_PUBLIC_HEADERS ${CAT_CUTIFIER_PUBLIC_HEADER_DIR}/*.h)

    set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
    set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    #Replace variables inside @@ with the current values
    configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
    #        DEPENDS ${CAT_CUTIFIER_PUBLIC_HEADERS}
            COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
            MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
            COMMENT "Generating docs")

    #add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

    if(LATEX)
        # run latex twice to fix links
        add_custom_command(
                OUTPUT ${PROJECT_BINARY_DIR}/doc_doxygen/index.pdf
                DEPENDS ${PROJECT_BINARY_DIR}/doc_doxygen/latex/refman.tex
                WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/doc_doxygen/latex/
                COMMAND ${LATEX}
                ARGS ${PROJECT_BINARY_DIR}/doc_doxygen/latex/refman.tex
                COMMAND ${LATEX}
                ARGS ${PROJECT_BINARY_DIR}/doc_doxygen/latex/refman.tex
                COMMAND ${CMAKE_COMMAND}
                ARGS -E copy ${PROJECT_BINARY_DIR}/doc_doxygen/latex/refman.pdf ${PROJECT_BINARY_DIR}/doc_doxygen/index.pdf
        )

        add_custom_target(pdf
                COMMENT "Generating PDF documentation with latex" VERBATIM
                DEPENDS ${PROJECT_BINARY_DIR}/doc_doxygen/index.pdf
                )

        add_custom_target(doc
                DEPENDS doc_doxygen pdf
                COMMENT "Generating API documentation with Doxygen" VERBATIM
                )
    endif()


else (DOXYGEN_FOUND)
    message("Doxygen need to be installed to generate the doxygen documentation")
endif (DOXYGEN_FOUND)
endif()

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE TRUE)

message(STATUS "LIBIGL INCLUDE DIR: ${LIBIGL_INCLUDE_DIR}")
message(STATUS "LIBIGL EXTRA DIR: ${LIBIGL_EXTRA_LIBRARIES}")

include(${VTK_USE_FILE})
include( ${CGAL_USE_FILE} )
include_directories(
        ${INCLUDE_DIRECTORIES}
        ${SPQR_INCLUDES}
        ${PYTHON_INCLUDE_DIRS}
        ${LIBIGL_INCLUDE_DIR}
        ${LIBIGL_INCLUDE_DIR}/../external/glad/include
        ${LIBIGL_EXTRA_LIBRARIES}
        ${PCL_INCLUDE_DIRS}
        ${CGAL_INCLUDE_DIR}
        ${OPENGL_INCLUDE_DIRS}
        ${GLUT_INCLUDE_DIRS}
        ${GLEW_INCLUDE_DIRS}
        ${GLFW3_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/include
        ${LIBIGL_EXTERNAL}
        ${GTK2_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/superquadric_fitting/include
        ${LIBIGL_EXTERNAL}
        ${CERES_INCLUDES})

link_directories(${PCL_LIBRARY_DIRS} /usr/local/lib)
add_definitions(${PCL_DEFINITIONS})

if ( CMAKE_COMPILER_IS_GNUCC )
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -std=gnu++1z -w -DNDEBUG -lstdc++ \
    -pthread -lXcursor -lGLEW -lGL \
    -lGLU -lX11 -lXxf86vm -lXrandr \
    -lpthread -fPIC -lXi -lm -lstdc++fs \
    -fopenmp -lboost_program_options \
    -lboost_system -lboost_filesystem")

###############
# LIBRARIES
###############

add_library(cpc_lib src/include/cpc.cpp)
add_library(merge src/include/merge2.cpp)
add_library(sdf src/include/sdf.cpp)
add_library(custom_isoline src/include/custom_isoline.cpp)
add_library(split_mesh SHARED src/include/split_mesh.cpp)
add_library(basic_mesh_functions src/include/basic_mesh_functions.cpp)
add_library(get_separate_lines src/include/get_separate_lines.cpp)
add_library(create_laplacian src/include/create_laplacian.cpp)
add_library(submesh src/include/submesh.cpp)
add_library(mesh_utils src/include/mesh_utils.cpp)
add_library(tinyxml_lib src/include/tinyxml2.cpp)

target_link_libraries(cpc_lib ${PCL_LIBRARIES} merge)
target_include_directories(basic_mesh_functions PUBLIC "${LIBIGL_INCLUDE_DIR}/../external/eigen")
target_link_libraries(basic_mesh_functions ${SPQR_LIBRARIES} igl::core custom_isoline)
target_include_directories(create_laplacian PUBLIC "${LIBIGL_INCLUDE_DIR}/../external/eigen")
target_link_libraries(create_laplacian igl::core basic_mesh_functions get_separate_lines)
target_link_libraries(mesh_utils ${PCL_LIBRARIES})
target_link_libraries(merge sdf)

###############
# EXECUTABLES
###############

# Laplacian Segmentation
add_executable(caf_segmentation src/conc_aware/conc_aware.cpp)
target_include_directories(caf_segmentation PUBLIC "${LIBIGL_INCLUDE_DIR}/../external/eigen")
target_link_libraries(caf_segmentation ${Boost_LIBRARIES}
        merge
        Boost::program_options
        tinyxml_lib
        igl::core
        igl::opengl
        igl::opengl_glfw
        igl::opengl_glfw_imgui
        create_laplacian
        basic_mesh_functions
        get_separate_lines
        split_mesh
        cpc_lib
        submesh)

#add_subdirectory ("src")
#add_subdirectory ("docs")

# Find all the public headers
#get_target_property(PUBLIC_HEADER_DIR src INTERFACE_INCLUDE_DIRECTORIES)
#file(GLOB_RECURSE PUBLIC_HEADERS ${PUBLIC_HEADER_DIR}/*.hpp)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/docs/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#Replace variables inside @@ with the current values
configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
        #DEPENDS ${PUBLIC_HEADERS}
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
        MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
        COMMENT "Generating docs")

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})


add_custom_target(Sphinx ALL
        COMMAND ${SPHINX_EXECUTABLE} -b html
        # Tell Breathe where to find the Doxygen output
        -Dbreathe_projects.concavity-aware-fields=${DOXYGEN_OUTPUT_DIR}/xml
        ${SPHINX_SOURCE} ${SPHINX_BUILD}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating documentation with Sphinx")

