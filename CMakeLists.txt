cmake_minimum_required(VERSION 3.14)
project(faiss_rest_api LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optional: try for a more static binary
option(BUILD_STATIC "Link statically where possible (experimental)" OFF)
if(BUILD_STATIC)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
endif()

include(FetchContent)

FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.18.3
)
FetchContent_MakeAvailable(httplib)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ====================== FAISS detection (flexible) ======================
# Mode 1 (preferred): FAISS was installed with `cmake --install`
# Mode 2: Point to the FAISS *source* tree you have (no need to install)
#
# Example for Mode 2:
#   cmake .. -DFAISS_SOURCE_DIR=/full/path/to/your/faiss-clone

set(FAISS_SOURCE_DIR "" CACHE PATH "Path to FAISS source root (use this if you did NOT run 'cmake --install' on FAISS)")

find_package(faiss CONFIG QUIET)
if(NOT faiss_FOUND)
    find_package(Faiss QUIET)
endif()

if(FAISS_SOURCE_DIR)
    message(STATUS "Using FAISS from source tree: ${FAISS_SOURCE_DIR}")

    # Headers live in faiss/faiss/ inside the source tree.
    # Adding ${FAISS_SOURCE_DIR}/faiss makes #include <faiss/Index.h> work.
    set(FAISS_INCLUDE_DIR "${FAISS_SOURCE_DIR}/faiss")

    # Try to find the built library (static or shared)
    find_library(FAISS_LIBRARY
        NAMES faiss libfaiss
        PATHS
            "${FAISS_SOURCE_DIR}/build"
            "${FAISS_SOURCE_DIR}/build/faiss"
            "${FAISS_SOURCE_DIR}/build/lib"
        NO_DEFAULT_PATH
    )

    if(NOT FAISS_LIBRARY)
        message(FATAL_ERROR
            "Could not find libfaiss in ${FAISS_SOURCE_DIR}/build\n"
            "Please build FAISS first (cd faiss && cmake -B build ... && cmake --build build)")
    endif()

    add_library(faiss::faiss UNKNOWN IMPORTED)
    set_target_properties(faiss::faiss PROPERTIES
        IMPORTED_LOCATION "${FAISS_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${FAISS_INCLUDE_DIR}"
    )

elseif(TARGET faiss::faiss OR TARGET faiss)
    message(STATUS "Using system-installed FAISS via find_package")
else()
    message(FATAL_ERROR
        "FAISS not found.\n"
        "Either:\n"
        "  1. Install it: cd faiss && cmake --install build   (then re-run this cmake)\n"
        "  2. Or pass -DFAISS_SOURCE_DIR=/path/to/faiss-clone   (see README)")
endif()

find_package(Threads REQUIRED)

# FAISS was built with OpenMP support, so we must satisfy its imported target
find_package(OpenMP REQUIRED)

add_executable(faiss_rest_api src/main.cpp)

target_link_libraries(faiss_rest_api PRIVATE
    httplib::httplib
    nlohmann_json::nlohmann_json
    Threads::Threads
    OpenMP::OpenMP_CXX
)

if(TARGET faiss::faiss)
    target_link_libraries(faiss_rest_api PRIVATE faiss::faiss)
else()
    target_link_libraries(faiss_rest_api PRIVATE faiss)
endif()

install(TARGETS faiss_rest_api DESTINATION bin)
