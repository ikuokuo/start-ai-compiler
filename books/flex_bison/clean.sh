#!/bin/bash

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

cd $BASE_DIR
find . -mindepth 2 -type f -name "Makefile" | while read -r mkfile; do
  cd $(dirname $mkfile)
  make clean
  cd $BASE_DIR
done
