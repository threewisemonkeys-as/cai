# do not actually run this file, it is just a collection of commands
exit 0

train.sh \
    -m "agentica-org/DeepSWE-Preview" \
    -d "d3" \
    -b 16 \
    -r 4 \
    -n 2 \
    -g 8 \
    $1 $2


train.sh \
    -m "Qwen/Qwen3-8B" \
    -d "d3" \
    -b 16 \
    -r 4 \
    -n 2 \
    -g 8 \
    $1 $2