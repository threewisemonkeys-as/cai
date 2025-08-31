export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout fixed-data-alta-cluster
git submodule update --init --recursive
pip install --user -e .
cd ..
git clone 
cd rllm
git remote set-url origin https://github.com/threewisemonkeys-as/rllm.git
git checkout rllm2 
rm -r verl 
git clone https://github.com/threewisemonkeys-as/verl.git
cd verl
git fetch 
git checkout main 
cd ..
pip install --user -e ./verl[vllm]
pip install --user -e .
cd ..
pip install -U --user openai==1.99.5