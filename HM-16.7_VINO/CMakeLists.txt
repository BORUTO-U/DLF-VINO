project(TAppEncoder)

SET(CMAKE_BUILD_TYPE "Release")

include_directories( ./App )
include_directories( ./Lib )

#aux_source_directory(./App/TAppDecoder A)  
aux_source_directory(./App/TAppEncoder B)  
#aux_source_directory(./App/utils C) 
aux_source_directory(./Lib/libmd5 D)          

aux_source_directory(./Lib/TAppCommon E)
aux_source_directory(./Lib/TLibCommon F)
aux_source_directory(./Lib/TLibDecoder G)
aux_source_directory(./Lib/TLibEncoder H)
aux_source_directory(./Lib/TLibVideoIO I)
#aux_source_directory(./Lib/TLibTorch J)
aux_source_directory(./Lib/TLibVino K)

##########################################################################3

set (TARGET_NAME "TAppEncoder")

#file (GLOB SRC
#        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
#        )

# Find OpenCV components if exist
find_package(OpenCV COMPONENTS imgcodecs QUIET)
if(NOT(OpenCV_FOUND))
    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
    return()
endif()

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${SRC})

link_directories(${LIB_FOLDER})

# Create library file from sources.
#add_executable(${TARGET_NAME} ${SRC})
add_executable(${TARGET_NAME} "./App/TAppEncoder/encmain.cpp" ${B} ${D} ${E} ${F} ${G} ${H} ${I} ${K})

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
COMPILE_PDB_NAME ${TARGET_NAME})


target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} ${OpenCV_LIBRARIES})


if(UNIX)
    target_link_libraries(${TARGET_NAME} ${LIB_DL} pthread)
endif()

#########################################################################3

#target_compile_features( TAppEncoder PUBLIC cxx_range_for)

