#!/bin/bash

# SWIFT GUI Launcher Script
# This script helps launch the GUI with proper settings and error checking

set -e  # Exit on error

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "    SWIFT GUI Launcher"
echo "    Modern Web UI with Multi-Monitor Support"
echo "========================================================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Check if we're in a graphical environment
if [ -z "$DISPLAY" ] && [ "$(uname)" = "Darwin" ]; then
    echo -e "${BLUE}Setting up macOS display environment...${NC}"
    export DISPLAY=:0
fi

# Step 1: Check Julia installation
echo -e "${BLUE}Checking dependencies...${NC}"

if ! command -v julia &> /dev/null; then
    echo -e "${RED}Error: Julia is not installed or not in PATH${NC}"
    echo "Please run ./setup.sh first to install all dependencies"
    exit 1
fi

echo -e "${GREEN}✓ Julia found: $(julia -e 'print(VERSION)')${NC}"

# Step 2: Check if swift.jl exists
if [ ! -f "swift.jl" ]; then
    echo -e "${RED}Error: swift.jl not found in current directory${NC}"
    echo "Please make sure you're running this script from the project root"
    exit 1
fi

echo -e "${GREEN}✓ GUI launcher found${NC}"

# Step 3: Check Plotly.js
if [ ! -f "general_modules/plotly-2.33.0.min.js" ]; then
    echo -e "${YELLOW}Warning: Plotly.js not found${NC}"
    echo "Visualizations may not work. Please download Plotly.js to general_modules/"
fi

# Step 4: Check Fortran libraries
LIBS_OK=true
if [[ "$(uname)" == "Darwin" ]]; then
    if [ ! -f "NNpot/libpotentials.dylib" ]; then
        echo -e "${YELLOW}Warning: libpotentials.dylib not found${NC}"
        LIBS_OK=false
    fi
elif [[ "$(uname)" == "Linux" ]]; then
    if [ ! -f "NNpot/libpotentials.so" ]; then
        echo -e "${YELLOW}Warning: libpotentials.so not found${NC}"
        LIBS_OK=false
    fi
fi

if [ "$LIBS_OK" = false ]; then
    echo -e "${YELLOW}Fortran libraries not compiled. Run ./setup.sh to compile them.${NC}"
fi

echo ""
echo -e "${GREEN}Launching SWIFT GUI...${NC}"
echo ""
echo -e "${BLUE}Features:${NC}"
echo "  • Multi-monitor support - centers on your current screen"
echo "  • Real-time calculation progress"
echo "  • Interactive parameter adjustment"
echo "  • Live wavefunction visualization"
echo ""
echo -e "${YELLOW}Note: Close the window or press Ctrl+C here to exit${NC}"
echo ""

# Run Julia in foreground with error handling
if julia swift.jl; then
    echo ""
    echo -e "${GREEN}GUI closed successfully.${NC}"
else
    EXIT_CODE=$?
    echo ""
    echo -e "${RED}GUI exited with error code: $EXIT_CODE${NC}"
    echo "Check the output above for error details."
    exit $EXIT_CODE
fi
