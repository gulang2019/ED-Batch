find_path(OoC_INCLUDE_DIR OoC.h ${OoC_ROOT_DIR}/include)
find_library(OoC_LIBRARY NAMES OoC PATHS ${OoC_ROOT_DIR}/lib)

if (OoC_INCLUDE_DIR AND OoC_LIBRARY) 
    set (OoC_FOUND TRUE)
    message("-- Found OoC")
    message("  * include: ${OoC_INCLUDE_DIR}")
    message("  * lib: ${OoC_LIBRARY}")
endif (OoC_INCLUDE_DIR AND OoC_LIBRARY)