#!/bin/bash
dir=$(pwd)
if [ ! -d $dir/dataset ]
then
  mkdir -p $dir/dataset
fi

if [ ! -d $dir/model ]
then
  mkdir -p $dir/model
fi

# Modelnet dataset
wget http://maxwell.cs.umass.edu/mvcnn-data/modelnet40v1.tar
tar -xvf modelnet40v1.tar --directory ./dataset
python preprocess_modelnet.py
