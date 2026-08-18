#!/bin/bash

echo "🔧 Installing Python dependencies..."
pip install --upgrade pip

# Install pydantic-core with pre-built wheel
pip install pydantic-core==2.14.1 --no-build-isolation

# Install remaining requirements
pip install -r requirements.txt --no-cache-dir

echo "✅ Build complete!"
