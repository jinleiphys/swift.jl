#!/bin/bash

# SWIFT GUI Launcher Script
# This script helps launch the GUI with proper settings for macOS

echo "========================================================================"
echo "    SWIFT GUI Launcher"
echo "========================================================================"
echo ""

# Check if we're in a graphical environment
if [ -z "$DISPLAY" ] && [ "$(uname)" = "Darwin" ]; then
    echo "Setting up macOS display environment..."
    export DISPLAY=:0
fi

# Launch Julia with GUI
echo "Launching SWIFT GUI..."
echo "If the window doesn't appear, try running in a local terminal (not SSH)"
echo ""

# Run Julia in foreground
julia swift.jl

echo ""
echo "GUI closed."
