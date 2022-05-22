# AA222 Final Project

## Setup Tricks and Tips on Ubuntu 20.04 LTS
```
jax Setup 
Install this might fails if you don't have gpu
 - jaxlib==0.3.10+cuda11.cudnn82
Instead, choose following ro run on cpu: 
 - jaxlib==0.3.10

To setup jax on GPU:
 1. pip install --upgrade jax==0.3.10 jaxlib==0.3.10+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_releases.html
2. NVIDIA-SMI 515.43.04    Driver Version: 515.43.04    CUDA Version: 11.7 
3. Install cudnn 
Download and install cudnn: 
tar -xvf cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive.tar.xz 
cp -r cudnn-linux-x86_64-8.4.0.27_cuda11.6-archive /usr/local/
in .bashrc add: 
export PATH=/usr/local/cudnn-8.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cudnn-8.4/lib:$LD_LIBRARY_PATH

pip install hydra-core
pip install hydra_colorlog --upgrade
pip install wandb
Run "wandb login" at shell and copy and paste the API key of team project on wandb

evojax and wdb integration setup: 
git clone -b wdb https://github.com/yutaizhou/evojax.git
cd evojax/
python setup.py develop
```

## Execution 
```
python run.py +run=mnist +mode=debug
python run.py +run=math +mode=debug
python run.py +run=cartpole +mode=debug
```
