import re
import wandb
import random
import argparse
import requests
import numpy as np
from typing import Optional, Union, List
from dataclasses import dataclass, field

import nmmo
from nmmo.core.config import Config as GameConfig
from nmmo.task.task_api import Task
from nmmo.task.scenario import Scenario

from openelm import ELM
from openelm.configs import ELMConfig, PromptModelConfig, EnvConfig, MAPElitesConfig
from openelm.environments import BaseEnvironment, Genotype, ENVS_DICT
from openelm.mutation_model import DiffModel
from openelm.utils.code_eval import pool_exec_processes
from openelm.sandbox.server.sandbox_codex_execute import ExecResult

from transformers import AutoTokenizer

from sample_tasks import uniq_predicates, tasks, import_str, task_import_str, task_prompt_str, task_instruction, task_preamble

from scripted import baselines

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
Phenotype = Optional[np.ndarray]

@dataclass
class NMMOConfig(EnvConfig):
    """Config for the NMMO environment."""
    env_name:str = "NMMO"
    behavior_space: list[list[float]] = field(
        default_factory=lambda: [
            # Unique_predicates, length of the task, difficulty
            [0, 20],
            [0, 20],
            [0, 20],
        ]
    )
    init_prompt: str = ""
    impr: str = ""
    starting_seeds: list[str] = field(default_factory=lambda: ["square"])
    instruction: int = 1
    crossover: bool = False
    batch_size: int = 1
    mutate: bool = False
    game_config: GameConfig = None

class NMMOTask(Genotype):
    """A task in the NMMO environment."""
    def __init__(self, program_str: str, result_obj: List[Task]):

        self.program_str = program_str
        self.result_obj = result_obj

    def evaluate(self) -> float:
        # how to evaluate the fitness of a task? (time taken for the baseline RL algo to solve?)
        # for now, just the length of the task
        self._fitness = len(self.program_str)/10

        return self._fitness
    
    def __str__(self) -> str:
        return self.program_str
    
    def _count_predicates(self, task_str):
        predicates = set()
        for i in uniq_predicates:
            if i in task_str:
                predicates.add(i)
        return len(predicates)
        
    def to_phenotype(self) -> Optional[Phenotype]:
        # phenotypes of the task?
        # creating a dummy version, string length of the task, unique predicates, difficulty of the task,
        # if self.valid:
        return np.array(
            [
                self.morphology["predicates"],
                self.morphology["length"],
                self.morphology["lines"]
            ]
        )
        # else: 
        #     return None

    @property
    def fitness(self) -> Optional[float]:
        return self._fitness


class NMMO(BaseEnvironment[NMMOTask]):
    """The NMMO environment."""

    def __init__(
        self,
        config: NMMOConfig,
        mutation_model: DiffModel,
    ) -> None:

        self.config: NMMOConfig = config
        self.batch_size = self.config.batch_size
        self.mutation_model: DiffModel = mutation_model
        self.genotype_space = np.array(self.config.behavior_space).T
        self.genotype_ndim = self.genotype_space.shape[1]
        self.impr = config.impr
        self.init_prompt = config.init_prompt

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        return None
    
    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        pass

    def construct_prompt(
        self, code_batch: Optional[Union[list[str], str]] = None
    ) -> dict[str, str]:
        
        prompt_str = task_import_str

        # if self.config.mutate and code_batch is not None:
        #     # add prev task to the prompt if mutate is true
        #     if isinstance(code_batch, list):
        #         prompt_str += "\n"+code_batch[0]
        #     elif isinstance(code_batch, str):
        #         prompt_str += "\n"+code_batch 
        # else:
        #     prompt_str += "\n"+self.init_prompt

        prompt_str += f'\n\n{task_instruction}\n\n{task_prompt_str}\n\n{task_preamble}'
            
        # instruction postpended to the prompt
        # inst = "\n# use the predicates listed in the imports and complete the task using diverse predicates than before\ndef task_"
        # import_s += inst
        # prompt_str += inst

        print('PROOOOMPT', prompt_str)
        return {"prompt": prompt_str, "template": f'{task_import_str}\n\n{task_preamble}'}

    def generate_programs(self, code_batch: list[dict[str, str]]) -> list[NMMOTask]:
        
        local_scope_exec: bool = False
        
        generated_programs = self.mutation_model.generate_programs(
            code_batch, local_scope_exec, do_trunc = False
        )

        # Truncate
        for i, gp in enumerate(generated_programs):
            ret_line = 'return scenario.tasks'
            i_return = gp.find(ret_line)
            generated_programs[i] = gp[:i_return+len(ret_line)].strip()

        for gp in generated_programs:
            print('COOOOODE', gp)
        
        # TODO: add sandbox option
        results = pool_exec_processes(
            generated_programs,
            func_name="new_task_4",
            args={
                'scenario': Scenario(self.config.game_config)
            },
            timeout=self.config.timeout,
            processes=self.config.processes,
            debug=self.config.debug,
        )

        results = [{'program_str': gen_prog, 'result_obj': res_obj}
                    for (gen_prog, res_obj) in zip(generated_programs, results)]
        return [NMMOTask(**p) for p in results]

    def random(self) -> list[NMMOTask]:
        program_list = [self.construct_prompt() for _ in range(self.batch_size)]
        new_tasks = self.generate_programs(program_list)
        for t in new_tasks:
            print("###########NEW TASKSSS", t.result_obj)
            print(t.program_str)
        return new_tasks

    def mutate(self, x: list[NMMOTask]) -> list[NMMOTask]:
        task_list = [sr.program_str for sr in x]
        program_list = list(map(self.construct_prompt, task_list))
        new_tasks = self.generate_programs(program_list)
        return new_tasks

    def fitness(self, task: NMMOTask) -> float:
        raise Exception
        if isinstance(task.result_obj, ExecResult):
            print('BAAAAAAAD!!!!!!!!!!!')
            return -np.inf

        print('GOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOD')

        return 0

# tmp test config
class ScriptedAgentTestConfig(nmmo.config.Small, nmmo.config.AllGameSystems):
  __test__ = False

  LOG_ENV = True

  LOG_MILESTONES = True
  LOG_EVENTS = False
  LOG_VERBOSE = False

  SPECIALIZE = True
  PLAYERS = [
    baselines.Fisher, baselines.Herbalist,
    baselines.Prospector,baselines.Carver, baselines.Alchemist,
    baselines.Melee, baselines.Range, baselines.Mage]
  
def main(temperature, imports, n_tasks, model, mutate, batch_size):

    impr = import_str[imports]
    prompt_tasks = "\n" + "\n".join(random.sample(tasks, n_tasks))

    config = ELMConfig()
    config.env = NMMOConfig()
    config.env.impr = impr
    config.env.init_prompt = prompt_tasks
    config.env.mutate = mutate
    config.env.batch_size = batch_size
    config.env.game_config = ScriptedAgentTestConfig()
    config.qd = MAPElitesConfig()
    # config.qd.map_grid_size = (20)
    config.model = PromptModelConfig()
    config.model.temp = temperature
    config.model.batch_size = batch_size
    config.batch_size = batch_size
    config.model.model_path = f"Salesforce/codegen-{model}-mono"

    ENVS_DICT["NMMO"] = NMMO

    elm = ELM(config)

    return elm.run(init_steps = 2, total_steps = 100)


if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument("--temperature", type=float, default=0.9, help="temperature for the LLM for sampling")
    args.add_argument("--mutate", action='store_true', default=False, help="include the task output from the LLM in the next gens of prompt or not")
    args.add_argument("--imports", type=str, default="short_import", help="Use a smaller import statement or a larger well-defined one")
    args.add_argument("--n_tasks", type=int, default=2, help="number of sample tasks to use as prompt")
    args.add_argument("--model", type=str, default='2B', help="model size to use, 2B/6B")
    args.add_argument("--batch_size", type=int, default=8, help="batch size")
    # Deep Speed
    # args.add_argument("--local_rank", type=int, default=0)
    args = args.parse_args()

    # wandb.init(
    #     project="NMMO-ELM",
    #     config=vars(args)
    # )

    # max_fitness, niches, qd, fitnesses = main(args.temperature, args.imports, args.n_tasks, args.model, args.mutate, args.batch_size)
    main(args.temperature, args.imports, args.n_tasks, args.model, args.mutate, args.batch_size)

    # write max fitness niches and qd to file with n_tasks, temperature, model, imports as the file name
    # with open(f"tasks_{args.n_tasks}_{args.temperature}_{args.imports}.txt", "w", encoding="utf-8") as f:
    #     f.write(f"max fitness: {max_fitness}\n")
    #     f.write(f"niches: {niches}\n")
    #     f.write(f"qd: {qd}\n")
    # print(f"Niches filled: {niches}")
    # print(f"QD: {qd}")