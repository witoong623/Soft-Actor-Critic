#!/usr/bin/env bash

python -m tools.manual_control --filter vehicle.evt.echo_4s3 --sync --fps 10 --collect-trajectory
