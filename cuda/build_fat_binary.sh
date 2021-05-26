export TORCH_CUDA_ARCH_LIST="3.5+PTX;3.7+PTX;5.0+PTX;5.2+PTX;5.3+PTX;6.0+PTX;6.1+PTX;6.2+PTX;7.0+PTX;7.2+PTX;7.5+PTX;8.0+PTX;8.6+PTX"

#python setup.py install
python setup.py bdist_wheel
