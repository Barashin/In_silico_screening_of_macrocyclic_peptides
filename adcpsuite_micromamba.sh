#!/bin/bash
# last update 10/16/24
env_name='adcpsuite'

echo "Running adcpsuite installation script v1.1"
if [ -z "${MAMBA_ROOT_PREFIX}" ]
then
    echo ERROR: please install micromamba first
    exit
else
    eval "$(${MAMBA_EXE} shell hook --shell=bash --root-prefix=${MAMBA_ROOT_PREFIX})"
    echo "Using micromamba at '${MAMBA_EXE}'."
fi

# First deactivate all micromamba/conda environments
# reference:https://stackoverflow.com/questions/68441393/deactivate-all-conda-environments
for i in $(seq ${CONDA_SHLVL}); do
    micromamba deactivate
done

if ! micromamba env list | grep " $env_name " >/dev/null 2>&1;
then
    echo "Creating '$env_name' environment."
    yes | micromamba create -n $env_name python=3.7 -c conda-forge --no-rc > /dev/null
else
    echo "Environment '$env_name' detected."
fi

micromamba activate $env_name
echo "Environment '$env_name' activated."
echo "Installing required dependencies from conda-forge..."
yes|micromamba install setuptools=59.8.0 numpy=1.21.6 openbabel=2.4.1 openmm=7.6.0 parmed=3.4.3 mmtf-python=1.1.3 pdbfixer=1.8.1  -c conda-forge --no-rc > /dev/null
echo "Installing other required dependencies from pypi.org(pip)..."
python -m pip install  Pillow==9.5.0 pyside2==5.15.2.1 pybel==0.15.5 biopython==1.81 pypdb==2.3 wget | grep -v "Requirement already satisfied"

echo "Installing Adsuite Packages from ccsb.scripps.edu..."
# 
# #Download required packages
# #Download required packages
ADCP_TMP_DIR=~/AdcpTmpDir
mkdir $ADCP_TMP_DIR

for pkglnk in \
https://ccsb.scripps.edu/mamba/pip/py37/adcp/adcp-0.0.25-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/adcp/adcp-0.0.25.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/adfr/adfr-0.0.22-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/adfr/adfr-0.0.22.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/adfrcc/adfrcc-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/adfrcc/adfrcc-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/appframework/appframework-0.0.3-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/appframework/appframework-0.0.3.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/autosite/autosite-0.0.4-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/autosite/autosite-0.0.4.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/bhtree/bhtree-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/bhtree/bhtree-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/dejavu2/dejavu2-0.0.4-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/dejavu2/dejavu2-0.0.4.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/geomutils/geomutils-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/geomutils/geomutils-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/gle/gle-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/gle/gle-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/mglkey/mglkey-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/mglkey/mglkey-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/mglutil/mglutil-0.0.4-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/mglutil/mglutil-0.0.4.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/molkit2/molkit2-0.0.5-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/molkit2/molkit2-0.0.5.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/mslib/mslib-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/mslib/mslib-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/mslibcom/mslibcom-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/mslibcom/mslibcom-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/opengltk/opengltk-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/opengltk/opengltk-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/pmvapp/pmvapp-0.0.6-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/pmvapp/pmvapp-0.0.6.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/prody/prody-0.0.4-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/prody/prody-0.0.4.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/pyglf/pyglf-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/pyglf/pyglf-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/support/support-0.0.2-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/support/support-0.0.2.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/utpackages/utpackages-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/utpackages/utpackages-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/volume/volume-0.0.2-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/volume/volume-0.0.2.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/legacy/autodocktools/autodocktools-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/legacy/autodocktools/autodocktools-0.0.1.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/legacy/molkit/molkit-0.0.2-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/legacy/molkit/molkit-0.0.2.tar.gz \
https://ccsb.scripps.edu/mamba/pip/py37/legacy/pybabel/pybabel-0.0.1-py3-none-any.whl \
https://ccsb.scripps.edu/mamba/pip/py37/legacy/pybabel/pybabel-0.0.1.tar.gz 
do
     #new_dir_name=`echo $pkglnk | sed "s|/| |g" | awk '{ print $(NF-1) }'`
#     if [ ! -d $ADCP_TMP_DIR/$new_dir_name ] ; 
#     then 
#         mkdir  $ADCP_TMP_DIR/$new_dir_name; 
#     fi
#     wget $pkglnk -P $ADCP_TMP_DIR/$new_dir_name
    nm=`echo $pkglnk | sed "s|/| |g" | awk '{ print $NF }'`
    if [ ! -f $ADCP_TMP_DIR/$nm ]; then
        echo "Downloading $nm"
        wget -q $pkglnk -P $ADCP_TMP_DIR 
    fi
    
done

python -m pip install --no-index --find-links=$ADCP_TMP_DIR  $ADCP_TMP_DIR/prody-0.0.4-py3-none-any.whl
python -m pip install --no-index --find-links=$ADCP_TMP_DIR  $ADCP_TMP_DIR/autosite-0.0.4-py3-none-any.whl
python -m pip install --no-index --find-links=$ADCP_TMP_DIR  $ADCP_TMP_DIR/mslib-0.0.1-py3-none-any.whl
python -m pip install --no-cache-dir --upgrade --no-dependencies adcp adfrcc  dejavu2 gle mglutil opengltk support volume adfr appframework bhtree geomutils mglkey molkit2 mslibcom pmvapp pyglf utpackages autodocktools  molkit  pybabel  --find-links=$ADCP_TMP_DIR | grep -v "Requirement already satisfied"

echo "YOU CAN DELETE TEMP DIRECTORY '$ADCP_TMP_DIR' MANUALLY!"

#python -m pip install --no-cache-dir  --upgrade adcp   --find-links=$ADCP_TMP_DIR 

#python -m pip install --no-cache-dir  --upgrade adcp adfrcc autosite dejavu2 gle mglutil mslib opengltk prody support volume adfr appframework bhtree geomutils mglkey molkit2 mslibcom pmvapp pyglf utpackages  --index-url https://ccsb.scripps.edu/mamba/pip/py37/ | grep -v "Requirement already satisfied"
#python -m pip install --no-cache-dir --upgrade autodocktools  molkit  pybabel --index-url https://ccsb.scripps.edu/mamba/pip/py37/legacy/ | grep -v "Requirement already satisfied"

#echo "Installation complete!"
