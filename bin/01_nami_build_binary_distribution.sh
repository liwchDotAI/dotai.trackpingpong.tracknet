# Build binary distribution
rm -rf ./build
rm -rf ./dist
python setup.py bdist_wheel
echo whl file in dist/ folder.
