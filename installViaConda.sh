#!/bin/bash


###########################################
# ##### Check if conda is installed ##### #
###########################################
flagConda=false

if ! command -v conda &> /dev/null
then
    echo "It appears that CONDA is not installed"
    echo "Run the following commands to install it"
    echo ""
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    chmod +x Miniconda3-latest-Linux-x86_64.sh"
    echo "    ./Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "once done, restart the terminal"
    echo ""
else
    flagConda=true
fi


#########################################
# ##### Conda creation enviroment ##### #
#########################################
if $flagConda
then
    # Installing enviroment via CONDA
    source /home/$USER/miniconda3/etc/profile.d/conda.sh
    conda create --name csp python=3.12.3
    conda activate csp

    # Installing package via CONDA
    conda install pip
    conda install conda-forge::libffi==3.4.6

    # Installing GeNN
    export CUDA_PATH=/usr/local/cuda
    pip install pybind11==2.13.6
    pip install psutil==7.0.0
    pip install numpy==2.2.4
    pip install pkgconfig==1.6.0
    git clone --branch 5.4.0 https://github.com/genn-team/genn.git
    cd genn
    python setup.py install

    # Installing package via PIP
    pip install matplotlib==3.10.1
    pip install pyvis==0.3.1
    pip install sPyNNaker==1!7.3.0
    pip install jupyterlab==4.4.0
    pip install pandas==2.2.3
    python -m ipykernel install --user --name=cspPy
fi
