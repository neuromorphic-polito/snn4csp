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
    conda create --name csp python=3.13.2
    conda activate csp

    # Installing GeNN
    export CUDA_PATH=/usr/local/cuda
    pip install pybind11 psutil numpy
    tar -xzf genn-5.1.0.tar.gz
    cd genn-5.1.0
    python setup.py install

    # Installing package via PIP
    pip install matplotlib
    pip install pyvis==0.3.1
    pip install sPyNNaker
    pip install jupyterlab
    pip install pandas
    python -m ipykernel install --user --name=cspPy313
fi
