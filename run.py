import os; os.environ['JAX_PLATFORMS'] = "GPU,CPU"
import shutil

from evojax import util
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import jax
from omegaconf import OmegaConf
import wandb

OmegaConf.register_new_resolver("get_class_name", lambda x: x._target_.split(".")[-1])

id = wandb.util.generate_id()

@hydra.main(config_path="configs", config_name="config", version_base=None) # version base since I am using hydra 1.2
def main(cfg):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    cfg_hydra = HydraConfig.get()
    log_dir = cfg_hydra.runtime.output_dir

    group = f"{cfg_hydra.job.name}-{id}" if "MULTIRUN" in str(cfg_hydra.mode) else None
    name  = f"{cfg_hydra.job.override_dirname}" if "MULTIRUN" in str(cfg_hydra.mode) else None
    run = wandb.init(**cfg.wandb, config= cfg_dict, reinit=True, group=group, name=name)    

    if cfg.gpu_id is not None: 
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.gpu_id)
    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg_hydra.job.env_set.CUDA_VISIBLE_DEVICES

    train_task = instantiate(cfg.task, test = False)
    test_task = instantiate(cfg.task, test = True)

    #* policy is handled differently depending on task
    policy_name = cfg_hydra.runtime.choices.policy
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
        log_dir = log_dir
    )

    # Train the model
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # * Save out a gif if doing control tasks like cartpole
    if cfg_hydra.runtime.choices.task == "cartpole":
        task_reset_fn = jax.jit(test_task.reset)
        policy_reset_fn = jax.jit(policy.reset)
        step_fn = jax.jit(test_task.step)
        act_fn = jax.jit(policy.get_actions)
        rollout_key = jax.random.PRNGKey(seed=0)[None, :]

        best_params = trainer.solver.best_params[None, :]
        images = []
        task_s = task_reset_fn(rollout_key)
        policy_s = policy_reset_fn(task_s)
        images.append(test_task.render(task_s, 0))
        done = False
        step = 0
        while not done:
            act, policy_s = act_fn(task_s, best_params, policy_s)
            task_s, r, d = step_fn(task_s, act)
            step += 1
            done = bool(d[0])
            if step % 5 == 0:
                images.append(test_task.render(task_s, 0))

        gif_file = os.path.join(
            log_dir,
            'cartpole_{}.gif'.format('hard' if cfg.task.harder else 'easy')
        )

        images[0].save(
            gif_file, save_all=True,
            append_images=images[1:], duration=40, loop=0
        )
        # logger.info('GIF saved to {}'.format(gif_file))
        wandb.log(
            {"video": wandb.Video(gif_file, fps=4, format="gif")}
        )

    run.finish()

if __name__ == '__main__':
    main()