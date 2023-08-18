pip uninstall -y qstorch
rm -rf build dist qstorch.egg-info
python setup.py bdist_wheel
cd dist
pip install qstorch-0.4-py3-none-any.whl
