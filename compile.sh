#!/bin/bash

cd submodules/diff-triangle2-rasterization/

# # Delete the build, diff_textured_triangle_rasterization.egg-info, and dist folders if they exist
rm -rf build
rm -rf dist
rm -rf diff_textured_triangle_rasterization.egg-info

pip install .

cd ../..
