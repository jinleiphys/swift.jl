# Makefile for Nuclear Potentials Library
# Creates a dynamic library from Fortran potential models that can be called from Julia

# Compiler settings
FC = gfortran
FFLAGS = -O2 -fPIC -Wall -Wextra
FFLAGS_F90 = $(FFLAGS) -ffree-form -ffree-line-length-none
FFLAGS_F77 = $(FFLAGS) -std=legacy -ffixed-form

# Platform detection
UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
  # macOS
  SHARED_EXT = dylib
  SHARED_FLAGS = -dynamiclib -single_module -undefined dynamic_lookup
else ifeq ($(UNAME), Linux)
  # Linux
  SHARED_EXT = so
  SHARED_FLAGS = -shared
else
  # Windows (assuming MinGW)
  SHARED_EXT = dll
  SHARED_FLAGS = -shared -Wl,--export-all-symbols
endif

# Library name and install path
LIBNAME = libpotentials
LIBFILE = $(LIBNAME).$(SHARED_EXT)
INSTALL_DIR = $(HOME)/lib

# Source files
F77_SOURCES = av18pot.f 
F90_SOURCES = 

# Object files
F77_OBJECTS = $(F77_SOURCES:.f=.o)
F90_OBJECTS = $(F90_SOURCES:.f90=.o)
OBJECTS = $(F77_OBJECTS) $(F90_OBJECTS)

# Default target
all: $(LIBFILE)

# Rule for .f (Fortran 77) files
%.o: %.f
	$(FC) $(FFLAGS_F77) -c $< -o $@

# Rule for .f90 (Fortran 90) files
%.o: %.f90
	$(FC) $(FFLAGS_F90) -c $< -o $@

# Create the shared library
$(LIBFILE): $(OBJECTS)
	$(FC) $(SHARED_FLAGS) -o $@ $(OBJECTS) $(LDFLAGS)
	@echo "Library $(LIBFILE) created successfully"



# Clean intermediate files
clean:
	rm -f $(OBJECTS) $(LIBFILE) *.mod 

# Deep clean (includes installed library)
distclean: clean
	rm -f $(INSTALL_DIR)/$(LIBFILE)

.PHONY: all install clean distclean julia_module