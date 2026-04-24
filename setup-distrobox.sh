#!/usr/bin/env bash
#
# Broxy distrobox environment setup script (Ubuntu 22.04)
#
# Usage:
#   Step 1: Run this script on the HOST to create the distrobox
#       ./setup-distrobox.sh create
#
#   Step 2: Enter the distrobox and run the install step
#       distrobox enter broxy
#       cd ~/Broxy
#       ./setup-distrobox.sh install
#
#   Step 3 (still inside distrobox): Build the project
#       ./setup-distrobox.sh build
#
#   Step 4 (still inside distrobox): Run
#       ./setup-distrobox.sh run
#

set -o errexit
set -o nounset
set -o pipefail

# Resolve the repo's absolute path on the host, regardless of where it was cloned
# (so the distrobox mount doesn't assume $HOME/broxy-home/Broxy).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACTION="${1:-help}"

case "$ACTION" in

#=============================================================================
# STEP 1: Create distrobox (run on HOST)
#=============================================================================
create)
    echo "=== Creating distrobox 'broxy' (Ubuntu 22.04 + GPU) ==="

    # Remove existing if present
    distrobox rm -f broxy 2>/dev/null || true

    distrobox create \
        --name broxy \
        --image docker.io/library/ubuntu:22.04 \
        --home "$HOME/broxy-home" \
        --volume "$SCRIPT_DIR:$HOME/Broxy:rw" \
        --additional-flags "--device nvidia.com/gpu=all"

    echo ""
    echo "=== Distrobox created! Next steps: ==="
    echo "  1. distrobox enter broxy"
    echo "  2. cd ~/Broxy"
    echo "  3. ./setup-distrobox.sh install"
    ;;

#=============================================================================
# STEP 2: Install dependencies (run INSIDE distrobox)
#=============================================================================
install)
    echo "=== Installing dependencies inside distrobox ==="

    sudo apt update

    echo "--- Installing build tools and libraries ---"
    sudo DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        build-essential \
        gfortran \
        qtbase5-dev \
        libglew-dev \
        libgl1-mesa-dev \
        libeigen3-dev \
        libqglviewer-dev-qt5 \
        git \
        wget \
        mesa-utils \
        gpg

    # gcc-8 is required by the project (.pro files hardcode gcc-8/g++-8).
    # It's not in Ubuntu 22.04 repos, so we grab it from focal (20.04).
    if ! command -v gcc-8 &>/dev/null; then
        echo "--- Installing gcc-8 from Ubuntu 20.04 (focal) repo ---"
        echo "deb http://archive.ubuntu.com/ubuntu focal main universe" \
            | sudo tee /etc/apt/sources.list.d/focal-toolchain.list
        sudo apt update
        sudo DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
            gcc-8 g++-8
        # Remove focal source to avoid pulling other focal packages
        sudo rm /etc/apt/sources.list.d/focal-toolchain.list
        sudo apt update
    else
        echo "--- gcc-8 already installed, skipping ---"
    fi

    # Install CUDA toolkit 10.2 via NVIDIA apt repo (project needs legacy cuSPARSE APIs).
    # We use the ubuntu1804 repo since CUDA 10.2 packages aren't available for 22.04.
    echo "--- Installing CUDA toolkit 10.2 ---"
    if [ ! -f /etc/apt/keyrings/cuda-archive-keyring.gpg ]; then
        wget -qO /tmp/cuda-repo-key.pub \
            https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
        sudo mkdir -p /etc/apt/keyrings
        sudo gpg --dearmor -o /etc/apt/keyrings/cuda-archive-keyring.gpg /tmp/cuda-repo-key.pub
        rm /tmp/cuda-repo-key.pub
        echo "deb [signed-by=/etc/apt/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" \
            | sudo tee /etc/apt/sources.list.d/cuda-10-2.list
        sudo apt update
    fi
    sudo DEBIAN_FRONTEND=noninteractive apt install -y --no-install-recommends \
        cuda-nvcc-10-2 \
        cuda-cudart-dev-10-2 \
        cuda-libraries-dev-10-2

    # Create /usr/local/cuda symlink (cuda.prf looks for it there)
    sudo ln -sf /usr/local/cuda-10.2 /usr/local/cuda

    # Verify CUDA is accessible
    echo "--- CUDA nvcc verified: ---"
    /usr/local/cuda-10.2/bin/nvcc --version

    # Clone vcglib if not present
    if [ ! -d "vcglib/.git" ]; then
        echo "--- Cloning vcglib v1.0.1 ---"
        rm -rf vcglib
        git clone --depth 1 --branch v1.0.1 https://github.com/cnr-isti-vclab/vcglib.git
    else
        echo "--- vcglib already present, skipping ---"
    fi

    echo ""
    echo "=== Dependencies installed! Next: ==="
    echo "  ./setup-distrobox.sh build"
    ;;

#=============================================================================
# STEP 3: Build (run INSIDE distrobox)
#=============================================================================
build)
    echo "=== Building Broxy ==="

    # Make sure CUDA is on PATH
    export PATH=/usr/local/cuda-10.2/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH:-}

    # Clean old build artifacts
    echo "--- Cleaning old builds ---"
    rm -f Bin/libGMorpho.so*

    # Build GMorpho library
    echo "--- Building GMorpho library ---"
    cd GMorpho
    qmake CONFIG+=debug_and_release
    # Fix qmake-generated Makefile issue (: \ entries)
    sed -i 's/\\: \\/\/ \\/g' Makefile
    sed -i 's/\\: \\/\/ \\/g' Makefile.Release 2>/dev/null || true
    sed -i 's/\\: \\/\/ \\/g' Makefile.Debug 2>/dev/null || true
    make debug
    cd ..

    # Build Broxy application
    echo "--- Building Broxy application ---"
    qmake CONFIG+=debug_and_release
    make debug

    echo ""
    echo "=== Build complete! ==="
    echo "  ./setup-distrobox.sh run"
    ;;

#=============================================================================
# STEP 4: Run (run INSIDE distrobox)
#=============================================================================
run)
    echo "=== Running Broxy ==="
    export LD_LIBRARY_PATH=.:/usr/local/cuda-10.2/lib64:${LD_LIBRARY_PATH:-}
    cd Bin
    ./Broxy
    ;;

#=============================================================================
# Help
#=============================================================================
*)
    echo "Broxy distrobox environment setup (Ubuntu 22.04)"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands (run in order):"
    echo "  create   - Create distrobox container (run on HOST)"
    echo "  install  - Install dependencies (run INSIDE distrobox)"
    echo "  build    - Build the project (run INSIDE distrobox)"
    echo "  run      - Run Broxy (run INSIDE distrobox)"
    ;;

esac
