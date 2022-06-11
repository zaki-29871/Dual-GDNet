# export CUDA_HOME=/usr/local/cuda-10.0
# export CUDA_HOME=/usr/local/cuda-10.1
cd GDNet/extensions
rm -r build
conda activate py38
# /opt/conda/bin/python ./setup.py install --user
python ./setup.py install --user
read -p "Press any key to continue..."