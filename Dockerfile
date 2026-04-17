ARG BASE=13.0.2-devel-ubuntu24.04
FROM nvidia/cuda:${BASE}

ENV DEBIAN_FRONTEND=noninteractive
ENV MAIN_USER=innuce
ENV CUDA_PATH=/usr/local/cuda
ENV GENN_PATH=/opt/genn
ENV HOME=/home/${MAIN_USER}

# -------------------------
# System dependencies (NO sudo)
# -------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        wget \
        git \
        nano \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        swig \
        libffi-dev \
        pkg-config \
        ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# -------------------------
# Install code-server
# -------------------------
RUN curl -fsSL https://code-server.dev/install.sh | sh

# -------------------------
# Create non-root user
# -------------------------
RUN useradd -m -s /bin/bash ${MAIN_USER}

# -------------------------
# Python packages (system-wide)
# -------------------------
RUN python3 -m pip install --no-cache-dir --break-system-packages \
    numpy==2.2.4 \
    matplotlib==3.10.1 \
    jupyterlab==4.4.0 \
    pandas==2.2.3 \
    psutil==7.0.0 \
    pybind11==2.13.6 \
    pkgconfig==1.6.0 \
    pyvis==0.3.1 \
    sPyNNaker==1!7.3.0

# -------------------------
# Install GeNN
# -------------------------
WORKDIR /opt
RUN git clone --branch 5.4.0 https://github.com/genn-team/genn.git ${GENN_PATH}

WORKDIR ${GENN_PATH}
RUN python3 setup.py install

# -------------------------
# Workspace + demo
# -------------------------
WORKDIR ${HOME}
RUN git clone --branch v1.0.0 https://github.com/neuromorphic-polito/snn4csp.git

# -------------------------
# Fix permissions
# -------------------------
RUN chown -R ${MAIN_USER}:${MAIN_USER} ${HOME} ${GENN_PATH}

USER ${MAIN_USER}

# -------------------------
# Jupyter kernel (user-level)
# -------------------------
RUN python3 -m ipykernel install --user --name=cspPy

# -------------------------
# Expose code-server
# -------------------------
EXPOSE 8080

# -------------------------
# Start code-server
# -------------------------
CMD ["code-server", "--bind-addr", "0.0.0.0:8080", "--auth", "none", "/home/innuce/snn4csp"]
