#!/bin/bash

# Parse command line arguments for NODES and GPUS_PER_NODE
if [ $# -lt 2 ]; then
  echo "Usage: $0 <NODES> <GPUS_PER_NODE>"
  exit 1
fi

NODES=$1
GPUS_PER_NODE=$2

# Start Ray cluster manually
ray_init_timeout=300  # Default timeout for Ray initialization in seconds.
ray_port=6379  # Port used by the Ray head node.
HEAD_NODE_ADDRESS="${MASTER_ADDR}:${ray_port}"

if [ "$NODE_RANK" -eq 0 ]; then
  # Head node
  ray start --head --port=${ray_port}
  ray status

  # Poll Ray until every worker node is active.
  for (( i=0; i < $ray_init_timeout; i+=5 )); do
      active_nodes=`python3 -c 'import ray; ray.init(); print(sum(node["Alive"] for node in ray.nodes()))'`
      if [ $active_nodes -eq $NODES ]; then
        echo "All ray workers are active and the ray cluster is initialized successfully."
        exit 0
      fi
      echo "Wait for all ray workers to be active. $active_nodes/$NODES is active"
      sleep 5s;
  done

  echo "Ray cluster initialized."
  ray status
else
  # Worker node - retry until connection succeeds or timeout expires
  for (( i=0; i < $ray_init_timeout; i+=5 )); do
    ray start --address="${HEAD_NODE_ADDRESS}" --block
    if [ $? -eq 0 ]; then
      echo "Worker: Ray runtime started with head address ${HEAD_NODE_ADDRESS}"
      ray status
      exit 0
    fi
    echo "Waiting until the ray worker is active..."
    sleep 5s;
  done

  echo "Ray worker start timeout, head address: ${HEAD_NODE_ADDRESS}"
  exit 1
fi