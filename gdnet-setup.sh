# export CUDA_HOME=/usr/local/cuda-10.0
# export CUDA_HOME=/usr/local/cuda-10.1
. /c/Users/p6607/anaconda3/etc/profile.d/conda.sh
conda activate py38
cd GDNet/extensions
rm -r build
# /opt/conda/bin/python ./setup.py install --user
python ./setup.py install --user
read -p "Press any key to continue..."
