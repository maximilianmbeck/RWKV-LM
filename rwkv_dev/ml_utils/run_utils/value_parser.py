from abc import ABC, abstractmethod
from ast import arg
from typing import Callable, List, Union
import numpy as np

class ListStringParser(ABC):

    def parse_to_list(self, list_str: str) -> List[Union[float, int]]:
        """Returns None if pattern does not match."""
        if self._match(list_str):
            args = list_str.replace(self.pattern, '').replace('(','').replace(')','').replace(' ', '').split(',')
            return self._generate_value_list(args)
        else:
            return None

    def _match(self, list_str: str) -> bool:
        pos = list_str.find(self.pattern)
        return pos == 0


    @abstractmethod
    def _generate_value_list(self, args: List[str]) -> List[Union[float, int]]:
        pass
    
    @property
    @abstractmethod
    def pattern(self) -> str:
        pass

class Arange(ListStringParser):
    def _generate_value_list(self, args: List[str], conv_fun: Callable) -> List[Union[float, int]]:
        assert 0 < len(args) <= 3
        return np.arange(*[conv_fun(arg) for arg in args]).tolist()

class ArangeInt(Arange):
    @property
    def pattern(self):
        return 'arange_int'
    
    def _generate_value_list(self, args: List[str]) -> List[Union[float, int]]:
        return super()._generate_value_list(args, int)

class ArangeFloat(Arange):
    @property
    def pattern(self):
        return 'arange_float'
    
    def _generate_value_list(self, args: List[str]) -> List[Union[float, int]]:
        return super()._generate_value_list(args, float)

class Linspace(ListStringParser):
    @property
    def pattern(self):
        return 'linspace'

    def _generate_value_list(self, args: List[str]) -> List[Union[float, int]]:
        assert 2 < len(args) <= 4
        endpoint = False
        if len(args) == 4:
            endp_str = args[-1]
            bool_str = endp_str.replace('endpoint', '').replace('=', '')
            if not ('True' in bool_str or 'False' in bool_str):
                raise ValueError('No bool variable found in endpoint argument.')
            endpoint = 'True' in bool_str
        return np.linspace(*[float(arg) for arg in args[0:2]], int(args[2]), endpoint=endpoint).tolist()

def parse_list_str(list_str: str) -> List[Union[float, int]]:
    list_parsers = [ArangeInt(), ArangeFloat(), Linspace()]
    for parser in list_parsers:
        parsed_list = parser.parse_to_list(list_str)
        if not parsed_list is None:
            return parsed_list
    
    raise ValueError(f'`{list_str}` could not be parsed!"')
