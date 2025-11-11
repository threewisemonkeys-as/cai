export PATH="$HOME/.local/bin:$PATH"

git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
cd R2E-Gym
git checkout mn-k8s
git submodule update --init --recursive
pip install --user -e .
cd ..

git clone https://github.com/threewisemonkeys-as/SWE-smith.git
cd SWE-smith
git checkout cai
pip install -e .
# python swesmith/build_repo/download_images.py
cd ..
