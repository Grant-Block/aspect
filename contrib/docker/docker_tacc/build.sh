#!/bin/bash

# This script generates a docker image from the latest aspect development version.
# It requires a docker installation on the local machine, and the ability to
# communicate with the docker daemon without root user privileges (see the docker
# webpage for an explanation).
# Note: This container is build from the developer version on Github, it does not use
# the local ASPECT folder. Therefore local changes are not included in the container.
VERSION=`cat ../../../VERSION`
docker build -t ghcr.io/geodynamics/aspect:${VERSION}-tacc .
