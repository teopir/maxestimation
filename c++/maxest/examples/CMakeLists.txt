add_executable(simpletest main_simpletest.cpp)
target_link_libraries(simpletest maxest)

add_executable(maxestfromgp main_maxestfromgp.cpp)
target_link_libraries(maxestfromgp maxest)
