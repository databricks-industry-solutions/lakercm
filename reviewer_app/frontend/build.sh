#!/bin/bash
# =============================================================================
# LakeRCM Frontend Build Script
# =============================================================================
# Builds the React app into static files served by the FastAPI backend.
# Usage: ./build.sh
# =============================================================================

set -e

echo "Building LakeRCM Frontend..."

if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    exit 1
fi

cd "$(dirname "$0")"

if [ -d "dist" ]; then
    echo "Cleaning old build..."
    rm -rf dist
fi

echo "Installing dependencies..."
npm install

echo "Building production bundle..."
npm run build

if [ -d "dist" ]; then
    echo "Frontend built successfully!"
    echo "Build output: $(pwd)/dist"
else
    echo "Build failed - dist directory not created"
    exit 1
fi
