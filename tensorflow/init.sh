#!/bin/bash

cd ops/cpp_wrappers
sh compile_wrappers.sh
cd ../tf_custom_ops
sh compile_op.sh
cd ../..

