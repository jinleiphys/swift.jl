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
F90_SOURCES = nijmengen.f90

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


# Create a module file for Julia
julia_module: $(LIBFILE)
	@echo 'module NuclearPotentials' > nuclear_potentials.jl
	@echo '    # Auto-generated Julia module for nuclear potentials library' >> nuclear_potentials.jl
	@echo '    const libpath = "$(LIBFILE)"' >> nuclear_potentials.jl
	@echo '' >> nuclear_potentials.jl
	@echo '    # AV18 Potential functions' >> nuclear_potentials.jl 
	@echo '    function av18pw(lpot, l, s, j, t, t1z, t2z, r)' >> nuclear_potentials.jl
	@echo '        result = zeros(Float64, 2, 2)' >> nuclear_potentials.jl
	@echo '        ccall((:av18pw_, libpath), Nothing,' >> nuclear_potentials.jl
	@echo '              (Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Int32}, Ref{Float64}, Ref{Float64}),' >> nuclear_potentials.jl
	@echo '              lpot, l, s, j, t, t1z, t2z, r, result)' >> nuclear_potentials.jl
	@echo '        return result' >> nuclear_potentials.jl
	@echo '    end' >> nuclear_potentials.jl
	@echo '' >> nuclear_potentials.jl
	@echo '    # Nijmegen Potential functions' >> nuclear_potentials.jl
	@echo '    function potn(l, s, j, type_str, r)' >> nuclear_potentials.jl
	@echo '        return ccall((:potn_, libpath), Float64,' >> nuclear_potentials.jl
	@echo '                    (Ref{Int32}, Ref{Int32}, Ref{Int32}, Ptr{UInt8}, Ref{Float64}, Csize_t),' >> nuclear_potentials.jl
	@echo '                    l, s, j, type_str, r, length(type_str))' >> nuclear_potentials.jl
	@echo '    end' >> nuclear_potentials.jl
	@echo '' >> nuclear_potentials.jl
	@echo '    # Add more function wrappers as needed...' >> nuclear_potentials.jl
	@echo 'end' >> nuclear_potentials.jl
	@echo "Julia module file created: nuclear_potentials.jl"

# Clean intermediate files
clean:
	rm -f $(OBJECTS) $(LIBFILE) *.mod nuclear_potentials.jl

# Deep clean (includes installed library)
distclean: clean
	rm -f $(INSTALL_DIR)/$(LIBFILE)
	rm -f nuclear_potentials.jl

.PHONY: all install clean distclean julia_module