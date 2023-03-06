ADD_LIBRARY(pqtree OBJECT ext-pqtree.cc pqnode.cc pqtree.cc ext-pqtree.h pqtree.h pqnode.h)
set_property(TARGET pqtree PROPERTY POSITION_INDEPENDENT_CODE ON)
INSTALL(FILES ext-pqtree.h pqtree.h pqnode.h set_methods.h DESTINATION "include")
