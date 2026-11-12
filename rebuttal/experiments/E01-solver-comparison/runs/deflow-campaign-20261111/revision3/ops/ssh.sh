#!/bin/sh
export SSH_ASKPASS_REQUIRE=force
export SSH_ASKPASS=/Users/ruslanageev/dev/mbrl-world-models/output/deflow-campaign-20261111/revision3/ops/ssh-askpass.sh
exec /usr/bin/ssh -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p 18290 "$@"
