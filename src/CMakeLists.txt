target_sources(${SRC_LIBRARY} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/simulator.cpp
                                        ${CMAKE_CURRENT_LIST_DIR}/MPIHandler.cpp
                                        ${CMAKE_CURRENT_LIST_DIR}/MPILaplace.cpp)
target_include_directories(${SRC_LIBRARY} PUBLIC include)

target_link_libraries(${MAIN_BINARY} PRIVATE ${SRC_LIBRARY})