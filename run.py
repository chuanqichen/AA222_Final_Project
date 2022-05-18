import os; os.environ['JAX_PLATFORMS'] = "GPU,CPU"
import shutil

from evojax import util
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf
import wandb

OmegaConf.register_new_resolver("get_class_name", lambda x: x._target_.split(".")[-1])

@hydra.main(config_path="configs", config_name="config", version_base=None) # version base since I am using hydra 1.2
def main(cfg):
    wandb.init(**cfg.wandb)    
    if cfg.gpu_id is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)

    train_task = instantiate(cfg.task, test = False)
    test_task = instantiate(cfg.task, test = True)

    #* policy is handled differently depending on task
    policy_name = HydraConfig.get().runtime.choices.policy
    if policy_name == "mlp_pi":
        policy = instantiate(
            cfg.policy,
            act_dim = train_task.act_shape[0]
        )
    elif policy_name == "mlp":
        policy = instantiate(
            cfg.policy,
            input_dim = train_task.obs_shape[0],
            output_dim = train_task.act_shape[0]
        )
    else:
        policy = instantiate(cfg.policy)

    solver = instantiate(
        cfg.solver,
        param_size = policy.num_params,
    )
    trainer = instantiate(
        cfg.trainer,
        policy = policy,
        solver = solver,
        train_task = train_task,
        test_task = test_task,
    )

    # Train the model
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(cfg.log_dir, 'best.npz')
    tar_file = os.path.join(cfg.log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = cfg.log_dir
    trainer.run(demo_mode=True)


if __name__ == '__main__':
    main()
