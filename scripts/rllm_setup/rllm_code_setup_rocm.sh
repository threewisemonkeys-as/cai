export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout rllm2_rocm --recurse-submodules
pip install --user -e .
cd ..
git clone --recurse-submodules https://github.com/threewisemonkeys-as/rllm.git
cd rllm
git checkout rllm2_rocm --recurse-submodules
pip install --user -e ./verl[vllm]
pip install --user -e .
cd ..

export RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES=1
