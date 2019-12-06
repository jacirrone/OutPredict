#!/bin/bash

eval "$(conda shell.bash hook)"

if [ -d "./scikit-learn" ]
then
    echo "Scikit-learn file already unzipped..."
else
    echo "Unzipping Scikit-learn file..."
    unzip scikit-learn.zip
fi

conda env create -f op3.yml

conda activate op3

#openMP env vars
export CC=/usr/bin/clang
export CXX=/usr/bin/clang++
export CPPFLAGS="$CPPFLAGS -Xpreprocessor -fopenmp"
export CFLAGS="$CFLAGS -I/usr/local/opt/libomp/include"
export CXXFLAGS="$CXXFLAGS -I/usr/local/opt/libomp/include"
export LDFLAGS="$LDFLAGS -L/usr/local/opt/libomp/lib -lomp"
export DYLD_LIBRARY_PATH=/usr/local/opt/libomp/lib

cd ./scikit-learn
python setup.py build_ext --inplace
python setup.py install --user
conda deactivate





