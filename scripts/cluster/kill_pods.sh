kubectl get pods -n default -o json \
| jq -r '
    .items[]
    | select(   # keep only the pods we want to delete …
        (.metadata.ownerReferences // [])     # if absent, treat as empty list
        | all(.kind != "DaemonSet")           # …and be sure none of the owners are DaemonSets
      )
    | .metadata.name                          # output just the pod name
' \
| xargs -r kubectl delete pod -n default