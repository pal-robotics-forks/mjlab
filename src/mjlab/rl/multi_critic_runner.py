import os
from pathlib import Path

import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners import MultiCriticOnPolicyRunner

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabMultiCriticOnPolicyRunner(MultiCriticOnPolicyRunner):
  """Multi-critic on-policy runner with mjlab checkpoint persistence."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    """Construct the mjlab multi-critic runner.

    The MultiCriticPPO configuration uses one critic observation group
    per critic:

        critic_0
        critic_1
        ...
        critic_N

    The model configuration itself can either use:
      - a shared ``critic`` config, which is copied by the runner/model
        construction logic for all critics, or
      - explicit ``critic_0``, ``critic_1``, ... configs.

    This method strips optional ``None`` values from whichever critic
    configuration format is present.
    """

    # -------------------------------------------------------------
    # Clean actor configuration.
    # -------------------------------------------------------------
    for key in ("actor",):
      if key in train_cfg:
        self._clean_model_cfg(train_cfg[key])

    # -------------------------------------------------------------
    # Clean critic configuration.
    #
    # Support both:
    #
    #   "critic": {...}
    #
    # and:
    #
    #   "critic_0": {...}
    #   "critic_1": {...}
    #   ...
    # -------------------------------------------------------------
    for key, cfg in train_cfg.items():
      if key.startswith("critic"):
          self._clean_model_cfg(cfg)
  
    super().__init__(
      env=env,
      train_cfg=train_cfg,
      log_dir=log_dir,
      device=device,
    )

  @staticmethod
  def _clean_model_cfg(model_cfg: dict) -> None:
    """Remove optional None-valued model configuration entries."""

    # MLPModel/CNNModel should not receive ``None`` configs.
    for opt in ("cnn_cfg", "distribution_cfg"):
      if model_cfg.get(opt) is None:
        model_cfg.pop(opt, None)

    # RNN parameters should only be present for RNN models.
    if model_cfg.get("rnn_type") is None:
      for opt in (
        "rnn_type",
        "rnn_hidden_dim",
        "rnn_num_layers",
      ):
        model_cfg.pop(opt, None)

  def export_policy_to_onnx(
    self,
    path: str,
    filename: str = "policy.onnx",
    verbose: bool = False,
  ) -> None:
    """Export the actor policy to ONNX using the legacy exporter."""

    onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)

    onnx_model.to("cpu")
    onnx_model.eval()

    os.makedirs(path, exist_ok=True)

    torch.onnx.export(
      onnx_model,
      onnx_model.get_dummy_inputs(),  # type: ignore[operator]
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
      verbose=verbose,
      input_names=onnx_model.input_names,  # type: ignore[arg-type]
      output_names=onnx_model.output_names,  # type: ignore[arg-type]
      dynamic_axes={},
      dynamo=False,
    )

  @staticmethod
  def _get_export_paths(
    checkpoint_path: str,
  ) -> tuple[Path, str, Path]:
    """Resolve ONNX export paths from a checkpoint path."""

    export_dir = Path(checkpoint_path).parent
    filename = f"{export_dir.name}.onnx"

    return (
      export_dir,
      filename,
      export_dir / filename,
    )

  def save(
    self,
    path: str,
    infos=None,
  ) -> None:
    """Save MultiCriticPPO checkpoint and mjlab environment state."""

    # -------------------------------------------------------------
    # Persist environment state.
    # -------------------------------------------------------------
    env_state = {
      "common_step_counter": (
        self.env.unwrapped.common_step_counter
      ),
    }

    infos = {
      **(infos or {}),
      "env_state": env_state,
    }

    # -------------------------------------------------------------
    # Save algorithm state.
    #
    # MultiCriticPPO.alg.save() should contain the actor and all
    # critic states managed by the algorithm.
    # -------------------------------------------------------------
    saved_dict = self.alg.save()

    saved_dict["iter"] = (
      self.current_learning_iteration
    )

    saved_dict["infos"] = infos

    torch.save(
      saved_dict,
      path,
    )

    # -------------------------------------------------------------
    # Optional W&B upload.
    # -------------------------------------------------------------
    if self.cfg["upload_model"]:
      self.logger.save_model(
        path,
        self.current_learning_iteration,
      )

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    """Load MultiCriticPPO checkpoint and restore mjlab state."""

    loaded_dict = torch.load(
      path,
      map_location=map_location,
      weights_only=False,
    )

    # -------------------------------------------------------------
    # Legacy single-critic checkpoint migration.
    #
    # This is retained for compatibility with old rsl_rl checkpoints.
    #
    # NOTE:
    # This migration only understands the old single-critic format.
    # A true multi-critic checkpoint should already contain the
    # MultiCriticPPO state produced by self.alg.save().
    # -------------------------------------------------------------
    if "model_state_dict" in loaded_dict:
      print(
        f"Detected legacy checkpoint at {path}. "
        "Migrating to new format..."
      )

      model_state_dict = loaded_dict.pop(
        "model_state_dict"
      )

      actor_state_dict = {}
      critic_state_dict = {}

      for key, value in model_state_dict.items():

        # Actor.
        if key.startswith("actor."):
          new_key = key.replace(
            "actor.",
            "mlp.",
          )
          actor_state_dict[new_key] = value

        elif key.startswith("actor_obs_normalizer."):
          new_key = key.replace(
            "actor_obs_normalizer.",
            "obs_normalizer.",
          )
          actor_state_dict[new_key] = value

        elif key in ("std", "log_std"):
          actor_state_dict[key] = value

        # Critic.
        if key.startswith("critic."):
          new_key = key.replace(
            "critic.",
            "mlp.",
          )
          critic_state_dict[new_key] = value

        elif key.startswith("critic_obs_normalizer."):
          new_key = key.replace(
            "critic_obs_normalizer.",
            "obs_normalizer.",
          )
          critic_state_dict[new_key] = value

      loaded_dict["actor_state_dict"] = (
        actor_state_dict
      )

      loaded_dict["critic_state_dict"] = (
        critic_state_dict
      )

    # -------------------------------------------------------------
    # Migrate rsl_rl 4.x actor distribution keys.
    # -------------------------------------------------------------
    actor_sd = loaded_dict.get(
      "actor_state_dict",
      {},
    )

    if "std" in actor_sd:
      actor_sd[
        "distribution.std_param"
      ] = actor_sd.pop("std")

    if "log_std" in actor_sd:
      actor_sd[
        "distribution.log_std_param"
      ] = actor_sd.pop("log_std")

    # -------------------------------------------------------------
    # Load algorithm.
    #
    # For MultiCriticPPO, self.alg.load() is responsible for restoring
    # the actor and all critic networks.
    # -------------------------------------------------------------
    load_iteration = self.alg.load(
      loaded_dict,
      load_cfg,
      strict,
    )

    if load_iteration:
      self.current_learning_iteration = (
        loaded_dict["iter"]
      )

    # -------------------------------------------------------------
    # Restore environment state.
    # -------------------------------------------------------------
    infos = loaded_dict.get(
      "infos",
      {},
    )

    if (
      infos
      and "env_state" in infos
      and "common_step_counter" in infos["env_state"]
    ):
      self.env.unwrapped.common_step_counter = (
        infos["env_state"]["common_step_counter"]
      )

    return infos