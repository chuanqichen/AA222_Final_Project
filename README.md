# AA222 Final Project

## Setup Tricks and Tips on Ubuntu 20.04 LTS
```
Install this might fails 

 - jaxlib==0.3.10+cuda11.cudnn82

Instead, choose following: 
 - jaxlib==0.3.10

For GPU setup, this is critial 
pip install --upgrade jax==0.3.10 jaxlib==0.3.10+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html
Download and install cudnn: 
tar -xvf cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz 
cp -r cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive /usr/local/
in .bashrc add: 
export PATH=/usr/local/cudnn-8.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cudnn-8.4/lib:$LD_LIBRARY_PATH

pip install hydra-core
pip install hydra_colorlog --upgrade
pip install wandb

git clone -b wdb https://github.com/yutaizhou/evojax.git
cd evojax/
python setup.py develop
```

## Execution 
python run.py +run=mnist +mode=debug

