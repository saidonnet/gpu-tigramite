#!/bin/bash
#==============================================================================
# GPU-Tigramite Standalone Installer
# Professional installation script with full system dependency management
#==============================================================================

set -euo pipefail

#==============================================================================
# Configuration
#==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="gpu-tigramite-install.log"
REQUIRED_CUDA_VERSION="11.8"
REQUIRED_GCC_VERSION="10"
MIN_DISK_SPACE_GB=5

#==============================================================================
# Colors and Formatting
#==============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $*" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*" | tee -a "$LOG_FILE"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE"; }

#==============================================================================
# Error Handling
#==============================================================================

cleanup() {
    if [ $? -ne 0 ]; then
        log_error "Installation failed! Check log: $LOG_FILE"
        log_error "Last 20 lines of log:"
        tail -20 "$LOG_FILE"
    fi
}
trap cleanup EXIT

#==============================================================================
# System Checks
#==============================================================================

check_os() {
    log_info "Checking operating system..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        OS_NAME=$ID
        OS_VERSION=$VERSION_ID
        log_success "Detected: $NAME $VERSION"
        
        if [[ "$OS_NAME" != "ubuntu" && "$OS_NAME" != "debian" ]]; then
            log_warning "This script is optimized for Ubuntu/Debian"
            log_warning "Some features may not work on $NAME"
        fi
    else
        log_error "Cannot detect OS"
        exit 1
    fi
}

check_root() {
    log_info "Checking sudo access..."
    
    if ! sudo -v; then
        log_error "This script requires sudo privileges"
        exit 1
    fi
    log_success "Sudo access confirmed"
}

check_disk_space() {
    log_info "Checking disk space..."
    
    available_space=$(df -BG "$SCRIPT_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$available_space" -lt "$MIN_DISK_SPACE_GB" ]; then
        log_error "Insufficient disk space. Need ${MIN_DISK_SPACE_GB}GB, have ${available_space}GB"
        exit 1
    fi
    log_success "Disk space: ${available_space}GB available"
}

check_python() {
    log_info "Checking Python installation..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 not found. Installing..."
        sudo apt-get update
        sudo apt-get install -y python3 python3-pip python3-dev
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    log_success "Python $PYTHON_VERSION found"
    
    # Ensure pip is installed
    if ! python3 -m pip --version &> /dev/null; then
        log_info "Installing pip for Python 3..."
        sudo apt-get install -y python3-pip python3-venv
    fi
    
    log_success "pip $(python3 -m pip --version | awk '{print $2}') found"
}

#==============================================================================
# System Dependencies Installation
#==============================================================================

install_build_tools() {
    log_info "Installing build tools..."
    
    sudo apt-get update
    sudo apt-get install -y \
        build-essential \
        cmake \
        git \
        wget \
        curl \
        linux-headers-$(uname -r)
    
    log_success "Build tools installed"
}

install_gcc10() {
    log_info "Installing GCC 10 (required for CUDA 11.8)..."
    
    if command -v gcc-10 &> /dev/null; then
        log_success "GCC 10 already installed"
        return
    fi
    
    sudo apt-get install -y gcc-10 g++-10
    
    # Verify installation
    if ! command -v gcc-10 &> /dev/null; then
        log_error "GCC 10 installation failed"
        exit 1
    fi
    
    GCC10_VERSION=$(gcc-10 --version | head -1)
    log_success "Installed: $GCC10_VERSION"
}

check_gpu_available() {
    # Check if NVIDIA GPU is accessible (drivers loaded)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        return 0  # GPU available
    else
        return 1  # GPU not available
    fi
}

install_cuda() {
    log_info "Checking CUDA installation..."
    
    # Check if CUDA already installed
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
        log_success "CUDA $CUDA_VERSION already installed"
        
        # Check if version is compatible
        if [[ "$CUDA_VERSION" < "11.8" ]]; then
            log_warning "CUDA $CUDA_VERSION may not be compatible. Recommended: 11.8+"
        fi
        
        # Check if GPU is actually accessible
        if check_gpu_available; then
            log_success "GPU is accessible (drivers loaded)"
            return
        else
            log_warning "CUDA installed but GPU not accessible"
            log_warning "You may need to reboot to load GPU drivers"
            return
        fi
    fi
    
    log_info "Installing CUDA Toolkit $REQUIRED_CUDA_VERSION..."
    
    # Detect Ubuntu version
    . /etc/os-release
    UBUNTU_VERSION=$VERSION_ID
    
    # Download NVIDIA keyring
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    
    # Update package lists
    sudo apt-get update
    
    # Install CUDA drivers
    sudo apt-get install -y cuda-drivers
    
    # Install only essential CUDA components (no nsight-systems bloat)
    log_info "Installing essential CUDA components only..."
    sudo apt-get install -y \
        cuda-compiler-11-8 \
        cuda-libraries-dev-11-8 \
        cuda-cudart-dev-11-8 \
        cuda-nvcc-11-8
    
    # Create symlink for /usr/local/cuda
    if [ ! -L /usr/local/cuda ]; then
        sudo ln -sf /usr/local/cuda-11.8 /usr/local/cuda
    fi
    
    # Setup environment
    setup_cuda_environment
    
    log_success "CUDA Toolkit $REQUIRED_CUDA_VERSION installed"
    
    # Check if GPU is accessible
    if ! check_gpu_available; then
        log_warning "GPU drivers installed but not loaded yet"
        log_warning "REBOOT REQUIRED to activate GPU"
    else
        log_success "GPU is accessible"
    fi
}

setup_cuda_environment() {
    log_info "Setting up CUDA environment..."
    
    # Create system-wide CUDA environment
    sudo tee /etc/profile.d/cuda.sh > /dev/null << 'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64
EOF
    
    # Source for current session (use parameter expansion to handle unset variables)
    export CUDA_HOME=/usr/local/cuda
    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:}/usr/local/cuda/lib64
    
    log_success "CUDA environment configured"
}

#==============================================================================
# Python Dependencies
#==============================================================================

install_python_deps() {
    log_info "Installing Python dependencies..."
    
    # Ubuntu 24.04 uses PEP 668 externally-managed-environment
    # We need --break-system-packages for system-wide installs
    PIP_FLAGS="--break-system-packages"
    
    # Skip pip upgrade (Debian-installed pip can't be upgraded via pip itself)
    log_info "Using system pip $(python3 -m pip --version | awk '{print $2}')"
    
    # Install build dependencies
    python3 -m pip install $PIP_FLAGS \
        setuptools>=68.0 \
        wheel \
        cmake>=3.18 \
        pybind11>=2.11.0 \
        numpy>=1.24.0
    
    # Install PyTorch with CUDA support
    log_info "Installing PyTorch with CUDA 11.8 support..."
    python3 -m pip install $PIP_FLAGS torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install Tigramite
    python3 -m pip install $PIP_FLAGS tigramite>=5.0.0
    
    log_success "Python dependencies installed"
}

#==============================================================================
# GPU-Tigramite Build and Installation
#==============================================================================

build_gpu_tigramite() {
    log_info "Building GPU-Tigramite..."
    
    cd "$SCRIPT_DIR"
    
    # Set compiler environment for CUDA
    export CC=/usr/bin/gcc-10
    export CXX=/usr/bin/g++-10
    export CUDAHOSTCXX=/usr/bin/g++-10
    
    # Ensure CUDA is in PATH for CMake
    export CUDA_HOME=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}${LD_LIBRARY_PATH:+:}/usr/local/cuda/lib64
    
    log_info "Using compilers: CC=$CC, CXX=$CXX"
    log_info "CUDA_HOME=$CUDA_HOME"
    
    # Verify nvcc is accessible
    if ! command -v nvcc &> /dev/null; then
        log_error "nvcc not found in PATH. CUDA installation may be incomplete."
        return 1
    fi
    log_info "nvcc version: $(nvcc --version | grep release | awk '{print $5}')"
    
    # Clean previous builds
    rm -rf build dist *.egg-info
    
    # Build and install (with --break-system-packages for Ubuntu 24.04)
    log_info "This may take 5-10 minutes..."
    log_info "Building with CUDA support..."
    
    # CRITICAL: COMPLETELY UNINSTALL any existing gpu-tigramite from ANY location
    log_info "Uninstalling any existing gpu-tigramite installations..."
    sudo python3 -m pip uninstall -y gpu-tigramite --break-system-packages 2>&1 | tee -a "$LOG_FILE" || true
    
    # CRITICAL: Force reinstall to ensure modified source code is recompiled
    PYTHONPATH="" sudo python3 -m pip install --break-system-packages --no-cache-dir --upgrade --force-reinstall . --no-build-isolation --verbose >> "$LOG_FILE" 2>&1
    
    if [ $? -ne 0 ]; then
        log_error "Build failed. Check log: $LOG_FILE"
        tail -50 "$LOG_FILE"
        return 1
    fi
    
    log_success "GPU-Tigramite built and installed"
    
    # CRITICAL FIX: Copy .so file to source directory for development mode
    log_info "Copying CUDA module to source directory..."
    SO_FILE=$(find /usr/local/lib/python*/dist-packages/gpu_tigramite/cuda -name "gpucmiknn*.so" 2>/dev/null | head -1)
    
    if [ -n "$SO_FILE" ] && [ -f "$SO_FILE" ]; then
        cp "$SO_FILE" "$SCRIPT_DIR/src/gpu_tigramite/cuda/"
        log_success "CUDA module copied to source directory"
    else
        log_warning "CUDA .so file not found in site-packages, checking build directory..."
        SO_FILE=$(find "$SCRIPT_DIR/build" -name "gpucmiknn*.so" 2>/dev/null | head -1)
        if [ -n "$SO_FILE" ] && [ -f "$SO_FILE" ]; then
            cp "$SO_FILE" "$SCRIPT_DIR/src/gpu_tigramite/cuda/"
            log_success "CUDA module copied from build directory"
        else
            log_error "CUDA .so file not found. Build may have failed."
            return 1
        fi
    fi
}

verify_installation() {
    log_info "Verifying installation..."
    
    # Test import
    if ! python3 -c "from gpu_tigramite import GPUCMIknn" 2>&1 | tee -a "$LOG_FILE"; then
        log_error "Import test failed"
        return 1
    fi
    
    # Test CUDA availability
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        log_success "CUDA is available"
        
        # Test GPU module - if kernels not available, reboot IS needed
        if python3 -c "from gpu_tigramite import GPUCMIknn; test = GPUCMIknn()" 2>&1 | grep -q "GPU CMIknn CUDA module not found"; then
            log_warning "CUDA module imported but GPU kernels not accessible"
            return 1  # Signal that reboot is needed
        else
            log_success "GPU-Tigramite fully operational"
        fi
    else
        log_warning "CUDA not available"
        return 1  # Signal that reboot is needed
    fi
}

#==============================================================================
# Main Installation Flow
#==============================================================================

main() {
    echo "=============================================================================="
    echo "GPU-Tigramite Standalone Installer"
    echo "=============================================================================="
    echo ""
    echo "This script will:"
    echo "  1. Check system requirements"
    echo "  2. Install system dependencies (GCC 10, CUDA 11.8)"
    echo "  3. Install Python dependencies (PyTorch, Tigramite)"
    echo "  4. Build and install GPU-Tigramite with CUDA support"
    echo ""
    echo "Log file: $LOG_FILE"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled"
        exit 0
    fi
    
    # Clear log file
    > "$LOG_FILE"
    
    echo ""
    echo "=============================================================================="
    echo "Phase 1: System Checks"
    echo "=============================================================================="
    check_os
    check_root
    check_disk_space
    check_python
    
    echo ""
    echo "=============================================================================="
    echo "Phase 2: System Dependencies"
    echo "=============================================================================="
    install_build_tools
    install_gcc10
    install_cuda
    
    echo ""
    echo "=============================================================================="
    echo "Phase 3: Python Dependencies"
    echo "=============================================================================="
    install_python_deps
    
    echo ""
    echo "=============================================================================="
    echo "Phase 4: Build GPU-Tigramite"
    echo "=============================================================================="
    build_gpu_tigramite
    
    echo ""
    echo "=============================================================================="
    echo "Phase 5: Verification"
    echo "=============================================================================="
    
    # Run verification once and capture result
    VERIFICATION_RESULT=0
    verify_installation || VERIFICATION_RESULT=$?
    
    echo ""
    echo "=============================================================================="
    echo "Installation Complete!"
    echo "=============================================================================="
    echo ""
    log_success "GPU-Tigramite successfully installed"
    echo ""
    
    # Use captured verification result (don't run again)
    if [ $VERIFICATION_RESULT -eq 0 ]; then
        echo "  ✓ No reboot needed - GPU is ready!"
        echo ""
        echo "  Test GPU-Tigramite:"
        echo "     python3 -c 'from gpu_tigramite import GPUCMIknn; print(\"✓ Ready\")'"
    else
        echo ""
        echo "  ⚠️  REBOOT REQUIRED"
        echo "  GPU drivers installed but not fully loaded."
        echo ""
        echo "  Please:"
        echo "  1. Reboot now: sudo reboot"
        echo "  2. After reboot, verify GPU: nvidia-smi"
        echo "  3. GPU-Tigramite will be ready to use"
        echo ""
        exit 2  # Exit with special code to signal reboot needed
    fi
    echo ""
    echo "Documentation: $SCRIPT_DIR/README.md"
    echo "Examples: $SCRIPT_DIR/examples/"
    echo ""
}

main "$@"