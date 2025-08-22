export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout fixed-data-alta-cluster
git submodule update --init --recursive
pip install --user -e .
cd ..
git clone https://github.com/threewisemonkeys-as/rllm.git
cd rllm
git checkout rllm2 
git submodule update --init --recursive
pip install --user -e ./verl[vllm]
pip install --user -e .
cd ..
pip install -U --user openai==1.99.5