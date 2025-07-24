git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout rllm2
pip install -e .
cd ..
git clone --recurse-submodules https://github.com/threewisemonkeys-as/rllm.git
cd rllm
git checkout rllm2
pip install -e ./verl[vllm]
pip install -e .
python examples/swe/prepare_swe_data.py
cd ..
