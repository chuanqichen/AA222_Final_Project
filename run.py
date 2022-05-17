import os; os.environ['JAX_PLATFORMS'] = "GPU,CPU"
import shutil

from evojax import util
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb
OmegaConf.register_new_resolver("get_class_name", lambda x: x._target_.split(".")[-1])
OmegaConf.register_new_resolver("get_original_cwd", lambda _: hydra.utils.get_original_cwd())

wandb.init(project="aa222_final", mode="disabled")
@hydra.main(config_path="configs", config_name="config")
def main(cfg):
    if cfg.gpu_id is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

    # log_dir = './output/mnist'
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir, exist_ok=True)
    # logger = util.create_logger(
    #     name='MNIST', log_dir=log_dir, debug=cfg.debug)
    logger = None

    policy = instantiate(cfg.policy, logger=logger)
    train_task = instantiate(cfg.train_task)
    test_task = instantiate(cfg.test_task)
    solver = instantiate(
        cfg.solver,
        param_size = policy.num_params,
        logger = logger
    )
    trainer = instantiate(
        cfg.trainer,
        policy = policy,
        solver = solver,
        train_task = train_task,
        test_task = test_task,
        logger = logger,
        log_dir = "output"
    )

    # Train the model
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)


if __name__ == '__main__':
    main()
