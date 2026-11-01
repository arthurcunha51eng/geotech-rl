#!/bin/bash
# Sends the repository to the pod and starts a command there, in the background. Runs on the local machine.
#   pod_push.sh <ip> <port> <remote command>
set -euo pipefail

IP=$1
PORT=$2
shift 2
KEY=$HOME/.ssh/geotech_pod
KNOWN=$HOME/geotech-rl/.pod_known_hosts
OPTIONS=(-i "$KEY" -o UserKnownHostsFile="$KNOWN" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20)
REPO=$(cd "$(dirname "$0")/.." && pwd)

tar -C "$(dirname "$REPO")" --exclude='__pycache__' --exclude='.venv' --exclude='.git' -czf /tmp/geotech-rl.tgz "$(basename "$REPO")"
ssh "${OPTIONS[@]}" -p "$PORT" "root@$IP" 'mkdir -p /root/work'
scp -q "${OPTIONS[@]}" -P "$PORT" /tmp/geotech-rl.tgz "root@$IP:/root/work/"
ssh "${OPTIONS[@]}" -p "$PORT" "root@$IP" 'cd /root/work && rm -rf geotech-rl && tar -xzf geotech-rl.tgz'
ssh "${OPTIONS[@]}" -p "$PORT" "root@$IP" "cd /root/work && nohup bash -c '$*' > /root/work/job.log 2>&1 &"
echo "started: $*"
