#!/usr/bin/env bash

set -euo pipefail

CAN="${CAN:-can0}"
BITRATE="${BITRATE:-1000000}"

echo "Bringing up ${CAN} at ${BITRATE} bps..."
sudo ip link set "${CAN}" down 2>/dev/null || true
sudo ip link set "${CAN}" type can bitrate "${BITRATE}"
sudo ip link set "${CAN}" up
ip -details link show "${CAN}"

echo
echo "Optional traffic check:"
echo "  candump ${CAN}"
