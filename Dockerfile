FROM isadoracw/rllm

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

RUN git clone https://github.com/threewisemonkeys-as/R2E-Gym.git
RUN cd R2E-Gym && \
    git checkout fixed-data-alta-cluster && \
    git submodule update --init --recursive && \
    pip install --user -e .
RUN cd .. && \
    git clone https://github.com/threewisemonkeys-as/rllm.git && \
    cd rllm && \
    git remote set-url origin https://github.com/threewisemonkeys-as/rllm.git && \
    rm -r verl && \
    git clone https://github.com/threewisemonkeys-as/verl.git && \
    cd verl && \
    git fetch && \
    git checkout main && \
    cd .. && \
    pip install --user -e ./verl[vllm] && \
    pip install --user -e . && \
    cd .. && \
    pip install -U --user openai==1.99.5 && \
    pip install protobuf==4.21.12

CMD ["/bin/bash"]
