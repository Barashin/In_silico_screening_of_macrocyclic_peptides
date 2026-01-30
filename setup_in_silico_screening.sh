#!/bin/bash
# Setup script for in_silico_screening environment
# Compatible with both local (micromamba) and Google Colab (conda)
# Last update: 2026-01-26

ENV_NAME='in_silico_screening'
PYTHON_VERSION='3.11'

echo "========================================"
echo "In Silico Screening Environment Setup"
echo "========================================"

# Detect environment manager
if command -v micromamba &> /dev/null; then
    CONDA_CMD="micromamba"
    echo "Using micromamba"
elif command -v conda &> /dev/null; then
    CONDA_CMD="conda"
    echo "Using conda"
else
    echo "ERROR: Neither micromamba nor conda found!"
    echo "Please install micromamba or conda first."
    exit 1
fi

# Deactivate all environments
if [ -n "${CONDA_SHLVL}" ]; then
    for i in $(seq ${CONDA_SHLVL}); do
        $CONDA_CMD deactivate 2>/dev/null || true
    done
fi

# Check if environment already exists
if $CONDA_CMD env list | grep -q " $ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        $CONDA_CMD env remove -n $ENV_NAME -y
    else
        echo "Installation cancelled."
        exit 0
    fi
fi

# Create new environment
echo "Creating '$ENV_NAME' environment with Python $PYTHON_VERSION..."
$CONDA_CMD create -n $ENV_NAME python=$PYTHON_VERSION -y -c conda-forge

# Activate environment
echo "Activating '$ENV_NAME' environment..."
if [ "$CONDA_CMD" = "micromamba" ]; then
    eval "$(micromamba shell hook --shell=bash)"
fi
$CONDA_CMD activate $ENV_NAME

# Install PyTorch (CPU version for compatibility)
echo ""
echo "Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install PyTorch Geometric
echo ""
echo "Installing PyTorch Geometric..."
pip install torch-geometric

# Install GPyTorch for Gaussian Process
echo ""
echo "Installing GPyTorch..."
pip install gpytorch

# Install RDKit
echo ""
echo "Installing RDKit..."
$CONDA_CMD install -y rdkit -c conda-forge

# Install scientific computing packages
echo ""
echo "Installing scientific packages..."
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Install Jupyter for notebook support
echo ""
echo "Installing Jupyter..."
pip install jupyter ipykernel ipywidgets

# Register kernel for Jupyter
echo ""
echo "Registering Jupyter kernel..."
python -m ipykernel install --user --name $ENV_NAME --display-name "Python ($ENV_NAME)"

# Install additional utilities
echo ""
echo "Installing additional utilities..."
pip install tqdm

echo ""
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo ""
echo "Environment: $ENV_NAME"
echo "Python version: $PYTHON_VERSION"
echo ""
echo "Installed packages:"
echo "  - PyTorch (CPU)"
echo "  - PyTorch Geometric"
echo "  - GPyTorch"
echo "  - RDKit"
echo "  - NumPy, Pandas, SciPy"
echo "  - scikit-learn"
echo "  - Matplotlib, Seaborn"
echo "  - Jupyter"
echo ""
echo "To activate this environment, run:"
if [ "$CONDA_CMD" = "micromamba" ]; then
    echo "  micromamba activate $ENV_NAME"
else
    echo "  conda activate $ENV_NAME"
fi
echo ""
echo "To verify installation, run:"
echo "  python -c 'import torch; import torch_geometric; import gpytorch; import rdkit; print(\"All packages loaded successfully!\")'"
echo ""
