# Dual-GDNet: Dual Guided-diffusion Network for Stereo Image Dense Matching
Neural Network Project for Ruei-Ping's Master Thesis

*Author: Ruei-Ping Wang (Jacky)*  
*Organizer: Yi-Chen Chen (Jack) and Ruei-Ping Wang (Jacky)*

Size of parameter: 11,593,136

## Prerequisite
* Windows 10 x64
* RTX 2080 Ti
* **Python 3.6**
* **PyTorch v1.6**
* **CUDA Toolkit v10.2**
* **cuDNN v8**
* **VS2019 (msvc)**

## Dependency Compilation in Windows (gdnet-lib)
    cd ./GANet/extensions/
    rm -r build
replace `CppExtension` with `CUDAExtension` in line 5 of setup.py    
    python ./setup.py install

## Installation
    .env/Scripts/activate.bat
    python -m pip install --upgrade pip
    pip install wheel
    pip install -r requirements.txt
    pip3 install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio===0.10.1+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

## Issues

1. https://github.com/pytorch/pytorch/issues/42467
2. https://github.com/pytorch/pytorch/issues/11004#issuecomment-506721668
3. https://blog.csdn.net/tanmx219/article/details/100829920

## Modify pytorch package
修改 C:\Python37\Lib\site-packages\torch\include\c10\macros\Macros.h
CONSTEXPR_EXCEPT_WIN_CUDA -> CONSTEXPR_EXCEPT_WIN_CUDA const

修改 C:\Python37\Lib\site-packages\torch\include\pybind11\cast.h 1449行
explicit operator type&() { return *(this->value); ->
explicit operator type&() { *((type *)(this->value)); }

setup.py
使用CUDAExtension

## Original KITTI 2015's height and width
    height: 1242 = 2*3^3*23
    width: 375 = 3*5^3

## Efficiency
Dual-GANet: 13 seconds per KITTI 2015 image with size 384, 1248

## Size of input image for models

The size of input images are required to be multiple for specific model.

| Name           | Height | Width  | Max disparity |
| -------------- | ------ | ------ | ------------- |
| GDNet_sdc6     | mod 64 | mod 64 | mod 32        |
| GDNet_sdc6f    | mod 64 | mod 64 | mod 32        |
| GDNet_mdc6     | mod 32 | mod 32 | mod 16        |
| GDNet_mdc6f    | mod 32 | mod 32 | mod 16        |
| GDNet_fdc6     | mod 32 | mod 32 | mod 8         |
| GDNet_fdc6f    | mod 32 | mod 32 | mod 8         |
| LEAStereo_fdcf | mod 24 | mod 24 | mod 8         |
| GDNet_sd9d6    | mod 64 | mod 64 | mod 32        |

### Evaluation
| Name           | Height | Width | Max disparity |
| -------------- | ------ | ----- | ------------- |
| Flyingthings3D | 384    | 960   | 144           |
| KITTI 2015     | 352    | 1216  | 144           |

## Other
- Disparity with float32 must multiply 256 to be uint16 format
