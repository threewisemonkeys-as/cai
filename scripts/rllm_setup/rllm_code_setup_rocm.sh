export PATH="$HOME/.local/bin:$PATH"

conda create -n ray_env python=3.11 -y
conda init
source ~/.bashrc
conda activate ray_env
pip install "ray[data,train,tune,serve] @ https://github.com/ROCm/ray/releases/download/v3.0.0-dev0%2Brocm/ray-3.0.0.dev0-cp311-cp311-manylinux2014_x86_64.whl"
git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout rllm2
pip install --user -e .
cd ..
git clone --recurse-submodules https://github.com/threewisemonkeys-as/rllm.git
cd rllm
git checkout rllm2
pip install --user -e ./verl[vllm]
pip install --user -e .
cd ..
pip install --upgrade --user antlr4-python3-runtime
conda install -c conda-forge libstdcxx-ng>=12.2
conda update -c conda-forge libstdcxx-ng gcc_linux-64