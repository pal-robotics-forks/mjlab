"""Evaluation manager for evaluating metrics in the environment, during evaluation"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence
from typing import Callable, TypeVar
import matplotlib.pyplot as plt

import torch
from prettytable import PrettyTable

from mjlab.managers.manager_base import ManagerBase, ManagerTermBase
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

@dataclass(kw_only=True)
class EvaluationTermCfg(abc.ABC):
  """Configuration for an evaluation term.

  Action terms evaluate chosen metrics on a given environment. Can be used on parallelised envs, returning mean evaluations"""

  weight: float
  """ Weight for this term, in case user want to do weighted combination of terms"""

  method: Callable[..., torch.Tensor]
  """ Mathematical method bound to this evaluation term """

  reference_value: float
  """ Reference value that will be used during evaluation computation """

  values: str
  """ Tensor containing the value(s) that will be compared to reference value """

  def build(self, env: ManagerBasedRlEnv) -> EvaluationTerm:
    return EvaluationTerm(self, env)


class EvaluationTerm(ManagerTermBase):
  """Base class for evaluation terms.

  The evaluation term is responsible for evaluating a chosen metric on a given
  environment. Can be used on parallelised envs, returning mean evaluations.
  """

  def __init__(self, cfg: EvaluationTermCfg, env: ManagerBasedRlEnv):
    self.cfg = cfg
    self.eval_result = 0.0
    super().__init__(env)
    self._entity = self._env.scene[_DEFAULT_ASSET_CFG.name]

  def get_attribute_from_string(self, obj, path: str):
    for key in path.split("."):
      obj = getattr(obj, key)
    return obj

  def process_evaluation(self) -> None:
    # val is of shape [num_envs, 1]
    val = self.get_attribute_from_string(self._entity, self.cfg.values)

    # Store meaned results of evaluating method
    self.eval_result = (torch.mean(self.cfg.method(val, self.cfg.reference_value))).item()


class EvaluationManager(ManagerBase):
  """Manages evaluation processing for the environment.

  The evaluation manager aggregates multiple evaluation terms, each evaluating a different
  metric of the simulation.
  """

  def __init__(self, cfg: dict[str, EvaluationTermCfg], env: ManagerBasedRlEnv):
    self.cfg = cfg
    super().__init__(env=env)

    self._step_counter = 0
    self._update_every = 100

    n = len(self._terms)
    window, axes = plt.subplots(n, 2, figsize=(12, 4 * n), gridspec_kw={'width_ratios': [3, 1]})
    self._evaluation_window = window

    self._lines = {}
    self._line_plots = {}
    self._bar_plots = {}
    self._history = {name: [] for name in self._terms.keys()}
    self._x_history = []
    self._graph_terms = []
    self._eval_results = []

    for i, name in enumerate(self._terms.keys()):
      row = axes[i] if n > 1 else axes
      self._line_plots[name] = row[0]
      self._bar_plots[name] = row[1]

      line, = self._line_plots[name].plot([], [], label=name)
      self._lines[name] = line
      self._line_plots[name].set_ylim(0, 1)
      self._line_plots[name].set_title(name)
      self._bar_plots[name].set_ylim(0, 1)

    plt.tight_layout()
    plt.show(block=False)

  def __str__(self) -> str:
    msg = f"<EvaluationManager> contains {len(self._term_names)} active terms.\n"
    table = PrettyTable()
    table.title = f"Active Evaluation Terms (shape: {self.total_evaluation_dim})"
    table.field_names = ["Index", "Name", "Dimension"]
    table.align["Name"] = "l"
    table.align["Weight"] = "r"
    for index, (name, term) in enumerate(self._terms.items()):
      table.add_row([index, name, term.cfg.weight])
    msg += table.get_string()
    msg += "\n"
    return msg

  # Properties.

  @property
  def total_evaluation_dim(self) -> int:
    return sum(self.evaluation_term_dim)

  @property
  def evaluation_term_dim(self) -> list[int]:
    return [1 for term in self._terms.values()]

  @property
  def evaluation_window(self):
    return self._evaluation_window

  @property
  def active_terms(self) -> list[str]:
    return self._term_names

  # Methods.

  def get_term(self, name: str) -> EvaluationTerm:
    return self._terms[name]

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> dict[str, float]:
    if env_ids is None:
      env_ids = slice(None)

    for term in self._terms.values():
      term.reset(env_ids=env_ids)
    return {}

  def refresh_mainplot(self) -> None:
    self._step_counter += 1
    if self._step_counter % self._update_every != 0:
      return

    self._x_history.append(self._step_counter)
    for name, result in zip(self._graph_terms, self._eval_results):
      self._history[name].append(result)
      self._lines[name].set_data(self._x_history, self._history[name])
      self._line_plots[name].set_xlim(0, self._step_counter)

      self._bar_plots[name].clear()
      self._bar_plots[name].set_ylim(0, 1)
      mean = sum(self._history[name]) / len(self._history[name])
      self._bar_plots[name].bar([name], [mean])

    plt.pause(0.001)

  def process_eval(self) -> None:
    self._graph_terms = []
    self._eval_results = []
    for term_name in self._terms.keys():
      self._terms[term_name].process_evaluation()
      self._graph_terms.append(term_name)
      self._eval_results.append(self._terms[term_name].eval_result)
    self.refresh_mainplot()

  def _prepare_terms(self):
    self._term_names: list[str] = list()
    self._terms: dict[str, EvaluationTerm] = dict()

    for term_name, term_cfg in self.cfg.items():
      term_cfg: EvaluationTermCfg | None
      if term_cfg is None:
        print(f"term: {term_name} set to None, skipping...")
        continue
      term = term_cfg.build(self._env)
      self._term_names.append(term_name)
      self._terms[term_name] = term


class NullEvaluationManager:
  """Placeholder for absent evaluation manager that safely no-ops all operations."""

  def __init__(self):
    self.active_terms: list[str] = []
    self.cfg = None

  def __str__(self) -> str:
    return "<NullEvaluationManager> (inactive)"

  def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, float]:
    return {}

  def process_eval(self) -> None:
    return None
