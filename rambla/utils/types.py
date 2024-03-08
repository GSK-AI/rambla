from typing import Callable, Dict, List, Literal, Tuple, Type, Union

import numpy as np

ExceptionType = Type[Exception]
FlexibleExceptionType = Union[ExceptionType, Tuple[ExceptionType, ...]]
DecoratorType = Callable[[Callable], Callable]


# TODO: include torch.tensor
ArrayLike = Union[List[float], np.ndarray]


LabelDictType = Dict[str, float]
BinaryInt = Literal[0, 1]
BinaryIntString = Literal["0", "1"]
