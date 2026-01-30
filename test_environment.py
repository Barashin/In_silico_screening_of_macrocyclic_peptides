#!/usr/bin/env python
"""
Environment Test Script for in_silico_screening

Tests all required packages and functionality
"""

import sys

def test_imports():
    """Test all required package imports"""
    print("="*50)
    print("Testing Package Imports")
    print("="*50)

    tests = []

    # PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        tests.append(("PyTorch", True))
    except ImportError as e:
        print(f"✗ PyTorch: {e}")
        tests.append(("PyTorch", False))

    # PyTorch Geometric
    try:
        import torch_geometric
        from torch_geometric.data import Data, Batch
        print(f"✓ PyTorch Geometric {torch_geometric.__version__}")
        tests.append(("PyTorch Geometric", True))
    except ImportError as e:
        print(f"✗ PyTorch Geometric: {e}")
        tests.append(("PyTorch Geometric", False))

    # GPyTorch
    try:
        import gpytorch
        print(f"✓ GPyTorch {gpytorch.__version__}")
        tests.append(("GPyTorch", True))
    except ImportError as e:
        print(f"✗ GPyTorch: {e}")
        tests.append(("GPyTorch", False))

    # RDKit
    try:
        from rdkit import Chem
        from rdkit import __version__ as rdkit_version
        print(f"✓ RDKit {rdkit_version}")
        tests.append(("RDKit", True))
    except ImportError as e:
        print(f"✗ RDKit: {e}")
        tests.append(("RDKit", False))

    # NumPy
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
        tests.append(("NumPy", True))
    except ImportError as e:
        print(f"✗ NumPy: {e}")
        tests.append(("NumPy", False))

    # Pandas
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
        tests.append(("Pandas", True))
    except ImportError as e:
        print(f"✗ Pandas: {e}")
        tests.append(("Pandas", False))

    # SciPy
    try:
        import scipy
        print(f"✓ SciPy {scipy.__version__}")
        tests.append(("SciPy", True))
    except ImportError as e:
        print(f"✗ SciPy: {e}")
        tests.append(("SciPy", False))

    # Matplotlib
    try:
        import matplotlib
        print(f"✓ Matplotlib {matplotlib.__version__}")
        tests.append(("Matplotlib", True))
    except ImportError as e:
        print(f"✗ Matplotlib: {e}")
        tests.append(("Matplotlib", False))

    return tests

def test_molecule_creation():
    """Test RDKit molecule creation from peptide sequence"""
    print("\n" + "="*50)
    print("Testing Molecule Creation")
    print("="*50)

    try:
        from rdkit import Chem

        test_seq = "ACDEFGHIKLM"
        mol = Chem.MolFromSequence(test_seq)

        if mol:
            print(f"✓ Created molecule from sequence: {test_seq}")
            print(f"  Atoms: {mol.GetNumAtoms()}")
            print(f"  Bonds: {mol.GetNumBonds()}")
            return True
        else:
            print("✗ Failed to create molecule")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_graph_conversion():
    """Test peptide to graph conversion"""
    print("\n" + "="*50)
    print("Testing Graph Conversion")
    print("="*50)

    try:
        import torch
        from torch_geometric.data import Data
        from rdkit import Chem

        test_seq = "ACDEFGHIKLM"

        # Create molecule
        mol = Chem.MolFromSequence(test_seq)
        if not mol:
            print("✗ Failed to create molecule")
            return False

        # Extract node features
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum() / 20.0,
                atom.GetDegree() / 4.0,
                atom.GetFormalCharge() / 2.0,
                int(atom.GetHybridization()) / 6.0,
                float(atom.GetIsAromatic()),
                atom.GetNumRadicalElectrons() / 2.0,
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float32)

        # Extract edges
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Create graph
        graph = Data(x=x, edge_index=edge_index)

        print(f"✓ Created graph from sequence: {test_seq}")
        print(f"  Nodes: {graph.x.shape[0]}")
        print(f"  Node features: {graph.x.shape[1]}")
        print(f"  Edges: {graph.edge_index.shape[1]}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gnn_model():
    """Test custom GNN model"""
    print("\n" + "="*50)
    print("Testing Custom GNN Model")
    print("="*50)

    try:
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data, Batch

        # Simple GNN layer test
        class SimpleGNN(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                self.linear = nn.Linear(in_channels, out_channels)

            def forward(self, x, edge_index):
                return self.linear(x)

        # Create test data
        x = torch.randn(10, 6)  # 10 nodes, 6 features
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

        # Create model
        model = SimpleGNN(6, 32)

        # Forward pass
        out = model(x, edge_index)

        print(f"✓ GNN model test passed")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {out.shape}")

        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("In Silico Screening Environment Test")
    print("="*50)
    print(f"Python version: {sys.version}")
    print()

    # Test imports
    import_tests = test_imports()

    # Test molecule creation
    mol_test = test_molecule_creation()

    # Test graph conversion
    graph_test = test_graph_conversion()

    # Test GNN model
    gnn_test = test_gnn_model()

    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)

    all_passed = all([result for _, result in import_tests])
    all_passed = all_passed and mol_test and graph_test and gnn_test

    for name, result in import_tests:
        status = "✓" if result else "✗"
        print(f"{status} {name}")

    print(f"{'✓' if mol_test else '✗'} Molecule Creation")
    print(f"{'✓' if graph_test else '✗'} Graph Conversion")
    print(f"{'✓' if gnn_test else '✗'} GNN Model")

    print()
    if all_passed:
        print("🎉 All tests passed! Environment is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
