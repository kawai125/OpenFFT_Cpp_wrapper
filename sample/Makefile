#====================================================================
#  Makefile for test code of openfft.hpp
#====================================================================

#--- compiler settings
#------ for GCC + OpenMPI
CXX = mpicxx

CPPFLAGS = -std=c++11 -MMD -fopenmp

#CPPFLAGS += -O3 -ffast-math -funroll-loops
CPPFLAGS += -Og -Wall -g3
#CPPFLAGS += -O0 -Wall -Wextra -g3 -fsanitize=address,leak,undefined

#--- macro settings
#CPPFLAGS += -DNDEBUG

#--- C++ compile target
SRC_DIR = .
SRC_CPP = $(SRC_DIR)/check_c2c_3d.cpp $(SRC_DIR)/check_r2c_3d.cpp $(SRC_DIR)/check_c2c_4d.cpp


FFTW_DIR    = $(FFTW_ROOT)
OPENFFT_DIR = $(OPENFFT_ROOT)
WRAPPER_DIR = -I../include/


#--- preparing for PATH
INCLUDE_FFTW  = -I$(FFTW_DIR)/include/
LIB_FFTW      = -L$(FFTW_DIR)/lib/ -lfftw3 -lfftw3_omp

OPENFFT_INCLUDE = -I$(OPENFFT_DIR)/include/
OPENFFT_LIB     =   $(OPENFFT_DIR)/lib/libopenfft.a

INCLUDE = $(INCLUDE_FFTW) $(OPENFFT_INCLUDE) $(WRAPPER_DIR)
LIBS    = $(OPENFFT_LIB) $(LIB_FFTW) -lm


#--- preparing for main code
OBJ_DIR = obj
ifeq "$(strip $(OBJ_DIR))" ""
  OBJ_DIR = .
endif

OBJS  = $(addprefix $(OBJ_DIR)/, $(notdir $(SRC_CPP:.cpp=.o) ) )
DEPS  = $(OBJS:.o=.d)
TGT   = $(notdir $(SRC_CPP:.cpp=.x) )


#=======================
#  test target
#=======================
all: $(TGT)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@[ -d $(OBJ_DIR) ] || mkdir -p $(OBJ_DIR)
	$(CXX) $(CPPFLAGS) $(INCLUDE) -o $@ -c $<

%.x: $(OBJ_DIR)/%.o
	$(CXX) $(CPPFLAGS) -o $@ $< $(LIBS)

#=======================
#  actions
#=======================
.PHONY: clean
clean:
	rm -rf ./*.x $(OBJ_DIR)

-include $(DEPS)
