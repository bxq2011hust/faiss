
include_directories(${CMAKE_SOURCE_DIR}/.. ${CMAKE_SOURCE_DIR})
file(GLOB testSrcs ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
    
foreach(source ${testSrcs})
    get_filename_component(name ${source} NAME_WE)

    # target
    add_executable(${name} ${source})
    target_link_libraries(${name} faiss blas lapack)

    # Install
    # install(TARGETS ${name} DESTINATION bin)
endforeach(source)

