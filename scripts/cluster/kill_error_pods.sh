kubectl get pods -n default -o json \
| jq -r '
    .items[]
    # 1) keep everything that is **not** owned by a DaemonSet …
    | select((.metadata.ownerReferences // []) | all(.kind != "DaemonSet"))
    # 2) … **and** is in an error-like state
    | select(
        # a) the whole Pod already moved to the Failed phase
        .status.phase == "Failed"
        # b) -- or at least one container ended with reason "Error"
        or ((.status.containerStatuses // [])
            | any(.state.terminated.reason? == "Error"))
        # (you can add more reasons here, e.g. CrashLoopBackOff, ImagePullBackOff, etc.)
      )
    | .metadata.name
' \
| xargs -r kubectl delete pod -n default