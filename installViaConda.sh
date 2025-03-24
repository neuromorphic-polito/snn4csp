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
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh"
    echo "    chmod +x Miniconda3-py38_4.10.3-Linux-x86_64.sh"
    echo "    ./Miniconda3-py38_4.10.3-Linux-x86_64.sh"
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
    echo "conda activate csp"
    echo "sudo apt update"
    echo "sudo apt install g++"
    echo "https://developer.nvidia.com/cuda-downloads"
    echo "sudo apt install libffi-dev"
    export CUDA_PATH=/usr/local/cuda
    pip install pybind11 psutil numpy
    tar -xzf genn-5.1.0.tar.gz
    cd genn-5.1.0
    python setup.py install

    # Installing package via PIP
    pip install matplotlib
    pip install pyvis==0.3.1
    pip install sPyNNaker8 --ignore-installed certifi
    pip install notebook
    python -m ipykernel install --user --name=cspPy312
    pip install nni

    # Setting SpiNNaker 8
    echo ""
    echo "python"
    echo "import spynnaker8 as sim"
    echo "sim.setup()"
    echo "sim.end()"

    echo "cd /home/<USER>"
    echo "nano .spynnaker.cfg"

    echo "[Machine]"
    echo "machineName = <NAME>"
    echo "version = 5"
fi
