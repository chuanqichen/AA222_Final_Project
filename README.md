# AA222 Final Project

## Setup Tricks and Tips on Ubuntu 20.04 LTS
```
Install this might fails 

 - jaxlib==0.3.10+cuda11.cudnn82

Instead, choose following: 
 - jaxlib==0.3.10

pip install hydra-core
pip install hydra_colorlog --upgrade
pip install wandb

git clone -b wdb https://github.com/yutaizhou/evojax.git
cd evojax/
python setup.py develop
```

## Execution 
python run.py +run=mnist +mode=debug

