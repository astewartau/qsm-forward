# build package
echo "Buildilng package..."
rm -rf dist/ qsm_forward.egg-info/
python -m build .

# documentation
echo "Building documentation..."
rm -rf docs-build/_build/
cd docs-build/ && make markdown && cd ../
mv docs-build/_build/markdown/* docs/
rm -rf docs-build/_build/

# upload
twine upload dist/*

