#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# args

args="release"
[ $# -gt 0 ] && args=$@

# func

ECHO="echo -e"

_echo() {
  text="$1"; shift; options="$1"; shift;
  [ -z "$options" ] && options="1;33";
  $ECHO "\033[${options}m${text}\033[0m"
}

# make

TAG="$(basename "$0"): "
_echo "${TAG}args=\"$args\"" "1;35"

cd $BASE_DIR
find . -mindepth 2 -type f -name "Makefile" | while read -r mkfile; do
  _echo "${TAG}make -f $mkfile -j$(nproc) args=\"$args\"" "1;35"
  make -f $mkfile -j$(nproc) args="$args"
done
