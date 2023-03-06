from typing import Any, List, Tuple
from enum import Enum
from importlib import import_module
import inspect

import numpy as np
import gymnasium as gym
import torch
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class ConfigHandler:
    def __init__(self):
        self.object_registry = {}

    def save_config(self, stacierl_object: Any, file: str) -> None:
        obj_dict = self.object_to_config_dict(stacierl_object)
        self.save_config_dict(obj_dict, file)

    def load_config(self, file: str) -> Any:
        obj_dict = self.load_config_dict(file)
        obj = self.config_dict_to_object(obj_dict)
        return obj

    def object_to_config_dict(self, stacierl_object: Any) -> dict:
        self.object_registry = {}
        config_dict = self._obj_to_dict(stacierl_object)
        self.object_registry = {}
        return config_dict

    def config_dict_to_object(
        self,
        config_dict: dict,
        object_registry: dict = None,
        class_str_replace: List[Tuple[str, str]] = None,
    ) -> Any:
        class_str_replace = class_str_replace or []
        self.object_registry = object_registry or {}
        obj = self._dict_to_obj(config_dict, class_str_replace)
        self.object_registry = {}
        return obj

    def load_config_dict(self, file: str) -> dict:

        with open(file, "r", encoding="utf-8") as config:
            config_dict = yaml.load(config, Loader=Loader)
        return config_dict

    def save_config_dict(self, config_dict: dict, file: str) -> None:
        if not file.endswith(".yml"):
            file += ".yml"
        with open(file, "w", encoding="utf-8") as dumpfile:
            yaml.dump(config_dict, dumpfile, default_flow_style=False, sort_keys=False)

    def _obj_to_dict(self, stacierl_object) -> dict:
        attributes_dict = {}
        attributes_dict[
            "class"
        ] = f"{stacierl_object.__module__}.{stacierl_object.__class__.__name__}"
        attributes_dict["_id"] = id(stacierl_object)
        if id(stacierl_object) in self.object_registry:
            return attributes_dict
        init_attributes = self._get_init_attributes(stacierl_object.__init__)

        if "args" in init_attributes:
            init_attributes.remove("args")

        if "kwargs" in init_attributes:
            init_attributes.remove("kwargs")

        if isinstance(stacierl_object, torch.optim.Optimizer):
            init_attributes.remove("params")
            defaults = stacierl_object.defaults
            for key in list(defaults.keys()):
                if key in init_attributes:
                    init_attributes.remove(key)
                else:
                    defaults.pop(key)
            attributes_dict.update(defaults)

        for attribute in init_attributes:
            value = getattr(stacierl_object, attribute)

            if isinstance(value, np.integer):
                dict_value = int(value)

            elif isinstance(value, torch.device):
                dict_value = str(value)

            elif isinstance(value, Enum):
                dict_value = value.value

            elif isinstance(value, np.ndarray):
                dict_value = value.tolist()

            elif isinstance(value, gym.Env):
                dict_value = f"{value.__module__}.{value.__class__.__name__}"

            elif isinstance(value, list):
                dict_value = []
                for v in value:
                    if hasattr(v, "__module__"):
                        search_string = v.__module__ + str(type(v).__bases__)
                        if "stacierl." in search_string:
                            dict_value.append(self._obj_to_dict(v))
                        continue

                    dict_value.append(v)

            else:
                if hasattr(value, "__module__"):
                    search_string = value.__module__ + str(type(value).__bases__)
                    if "stacierl.optimizer" in value.__module__:
                        dict_value = self._obj_to_dict(value)

                    elif "stacierl." in search_string:
                        dict_value = self._obj_to_dict(value)

                    elif "torch.optim.lr_scheduler" in value.__module__:
                        dict_value = self._obj_to_dict(value)

                    elif isinstance(value, torch.optim.Optimizer):
                        dict_value = self._obj_to_dict(value)

                    else:
                        raise NotImplementedError(
                            f"Handling this class {value.__class__} in not implemented "
                        )

                else:
                    dict_value = value

            attributes_dict[attribute] = dict_value
        self.object_registry[id(stacierl_object)] = attributes_dict
        return attributes_dict

    @staticmethod
    def _get_init_attributes(init_function):

        init_attributes = []
        kwargs = inspect.signature(init_function)
        for param in kwargs.parameters.values():
            init_attributes.append(param.name)

        return init_attributes

    def _dict_to_obj(
        self, obj_config_dict: dict, class_str_replace: List[Tuple[str, str]]
    ):
        if not ("class" in obj_config_dict.keys() and "_id" in obj_config_dict.keys()):
            return obj_config_dict

        obj_id = obj_config_dict.pop("_id")
        if obj_id in self.object_registry.keys():
            return self.object_registry[obj_id]

        class_str: str = obj_config_dict.pop("class")
        for str_replace in class_str_replace:
            class_str.replace(str_replace[0], str_replace[1])
        for attribute_name, value in obj_config_dict.items():
            if isinstance(value, dict):
                obj_config_dict[attribute_name] = self._dict_to_obj(
                    value, class_str_replace
                )
            if isinstance(value, (list, tuple)):
                for i, list_entry in enumerate(value):
                    if isinstance(list_entry, dict):
                        obj_config_dict[attribute_name][i] = self._dict_to_obj(
                            list_entry, class_str_replace
                        )

        constructor = self._get_class_constructor(class_str)
        obj = constructor(**obj_config_dict)
        self.object_registry[obj_id] = obj
        return obj

    def _get_class_constructor(self, class_str: str):
        module_path, class_name = class_str.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
