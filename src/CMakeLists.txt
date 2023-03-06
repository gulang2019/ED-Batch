set (SRC
    utils.cc
    pattern_cache.cc
    scheduler.cc
    static_scheduler.cc
    OoC.cc 
)

ADD_LIBRARY(OoC SHARED ${SRC})

target_include_directories(OoC PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/pq-trees)
add_subdirectory(pq-trees)
target_link_libraries(OoC PRIVATE pqtree)

if(CMAKE_SYSTEM MATCHES "Windows")
INSTALL(TARGETS OoC RUNTIME DESTINATION "lib")
else(CMAKE_SYSTEM MATCHES "Windows")
INSTALL(TARGETS OoC LIBRARY DESTINATION "lib")
endif(CMAKE_SYSTEM MATCHES "Windows")
INSTALL(FILES OoC.h utils.h DESTINATION "include")
