from kubernetes import client, config
config.load_kube_config()        # uses $KUBECONFIG
v1 = client.CoreV1Api()
print([n.metadata.name for n in v1.list_node().items])