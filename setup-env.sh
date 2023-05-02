
# get miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh
rm miniconda.sh

# remove previous environment
conda activate base
conda env remove --name qsm_forward

# setup conda environment
conda create --name qsm_forward python=3.8
conda activate qsm_forward
pip install nibabel nilearn numpy ipykernel matplotlib scikit-image dipy nilearn torch

