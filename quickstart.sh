#!/bin/bash

echo "🚀 DFAL Quantum Circuit Discovery Engine - Quick Start"
echo "======================================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "✓ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Activate virtual environment
source venv/bin/activate

# Install core dependencies (minimal for demo)
echo ""
echo "Installing core dependencies..."
pip install -q numpy scipy aiohttp python-dotenv

echo "✓ Core dependencies installed"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✓ Created .env file (add your API keys if using hardware)"
else
    echo "✓ .env file exists"
fi

# Create discoveries directory
mkdir -p discoveries
echo "✓ Output directory ready"

echo ""
echo "======================================================="
echo "Setup complete! You can now run:"
echo ""
echo "  python main.py --generations 5"
echo ""
echo "Or for a longer run:"
echo ""
echo "  python main.py --generations 20"
echo ""
echo "To use IonQ hardware (requires API key in .env):"
echo ""
echo "  python main.py --generations 10 --hardware"
echo ""
echo "======================================================="
echo ""

# Run a quick demo
read -p "Run a quick 5-generation demo now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo ""
    python main.py --generations 5
fi