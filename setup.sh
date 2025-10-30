#!/bin/bash

# Swift.jl Project Setup Script
# This script sets up the complete development environment for the nuclear physics framework

set -e  # Exit on error

echo "=========================================="
echo "Swift.jl Project Setup"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Minimum required Julia version
MIN_JULIA_VERSION="1.9.0"

# Function to compare version numbers
version_ge() {
    # Returns 0 (true) if $1 >= $2
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Function to get Julia version
get_julia_version() {
    if command -v julia &> /dev/null; then
        julia -e 'print(VERSION)' 2>/dev/null || echo "0.0.0"
    else
        echo "0.0.0"
    fi
}

# Step 1: Check and install Julia
echo "Step 1: Checking Julia installation..."
echo "=========================================="

CURRENT_JULIA_VERSION=$(get_julia_version)

if [ "$CURRENT_JULIA_VERSION" = "0.0.0" ]; then
    echo -e "${YELLOW}Julia is not installed.${NC}"
    NEED_INSTALL=true
else
    echo "Current Julia version: $CURRENT_JULIA_VERSION"
    if version_ge "$CURRENT_JULIA_VERSION" "$MIN_JULIA_VERSION"; then
        echo -e "${GREEN}Julia version is sufficient (>= $MIN_JULIA_VERSION)${NC}"
        NEED_INSTALL=false
    else
        echo -e "${YELLOW}Julia version is outdated (required >= $MIN_JULIA_VERSION)${NC}"
        NEED_INSTALL=true
    fi
fi

if [ "$NEED_INSTALL" = true ]; then
    echo ""
    echo "Installing Julia..."

    # Detect platform
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            echo "Using Homebrew to install Julia..."
            brew install julia
        else
            echo -e "${RED}Error: Homebrew is not installed.${NC}"
            echo "Please install Homebrew from https://brew.sh/ or install Julia manually from https://julialang.org/downloads/"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Downloading Julia for Linux..."
        JULIA_URL="https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.0-linux-x86_64.tar.gz"
        wget -O julia.tar.gz "$JULIA_URL"
        tar -xzf julia.tar.gz
        sudo mv julia-*/ /opt/julia
        sudo ln -sf /opt/julia/bin/julia /usr/local/bin/julia
        rm julia.tar.gz
    else
        echo -e "${RED}Error: Unsupported operating system: $OSTYPE${NC}"
        echo "Please install Julia manually from https://julialang.org/downloads/"
        exit 1
    fi

    # Verify installation
    INSTALLED_VERSION=$(get_julia_version)
    if [ "$INSTALLED_VERSION" = "0.0.0" ]; then
        echo -e "${RED}Error: Julia installation failed${NC}"
        exit 1
    else
        echo -e "${GREEN}Julia $INSTALLED_VERSION installed successfully${NC}"
    fi
fi

echo ""

# Get the absolute path to the project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Step 2: Install Julia packages
echo "Step 2: Installing Julia packages..."
echo "=========================================="

cd "$SCRIPT_DIR"

if [ ! -f "setup.jl" ]; then
    echo -e "${RED}Error: setup.jl not found in project root${NC}"
    exit 1
fi

echo "Running julia setup.jl..."
julia setup.jl

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Julia packages installed successfully${NC}"
else
    echo -e "${RED}Error: Julia package installation failed${NC}"
    exit 1
fi

echo ""

# Step 3: Compile Fortran libraries
echo "Step 3: Compiling Fortran libraries..."
echo "=========================================="

cd "$SCRIPT_DIR/NNpot"

# Check for gfortran
if ! command -v gfortran &> /dev/null; then
    echo -e "${RED}Error: gfortran compiler not found${NC}"
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "On macOS, install with: brew install gcc"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "On Linux, install with: sudo apt-get install gfortran  (Debian/Ubuntu)"
        echo "                    or: sudo yum install gcc-gfortran  (RedHat/CentOS)"
    fi
    exit 1
fi

echo "Found gfortran: $(gfortran --version | head -n 1)"
echo ""

# Clean previous builds
echo "Cleaning previous builds..."
make clean

# Compile libraries
echo "Compiling Fortran libraries..."
make

if [ $? -eq 0 ]; then
    echo -e "${GREEN}Fortran libraries compiled successfully${NC}"

    # List generated libraries
    echo ""
    echo "Generated libraries:"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        ls -lh *.dylib 2>/dev/null || echo "No .dylib files found"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        ls -lh *.so 2>/dev/null || echo "No .so files found"
    else
        ls -lh *.dll 2>/dev/null || echo "No .dll files found"
    fi
else
    echo -e "${RED}Error: Fortran library compilation failed${NC}"
    exit 1
fi

echo ""
echo "=========================================="
echo -e "${GREEN}Setup completed successfully!${NC}"
echo "=========================================="
echo ""
echo "You can now:"
echo "  - Start development: cd swift && julia"
echo "  - Run main calculation: cd swift && julia swift_3H.jl"
echo ""
