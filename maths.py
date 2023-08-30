from enum import Enum, auto
from Typing import Union, Type

import tensorflow as tf

class DistributionType(Enum):
    CONSTANT = auto()
    UNIFORM = auto()
    NORMAL = auto()

@dataclass
class Distribution:
    value : Union[int, float] = None
    dtype : Type = float
    min_ : Union[int, float] = None
    max_ : Union[int, float] = None
    mean : float = None
    std : float = None
    distibution : DistributionType = DistributionType.CONSTANT
    num_samples : int = 1

    def sample(self) -> Union[List[Union[int, float]], Union[int, float]]:
        
        match self.distibution:
            
            case DistributionType.CONSTANT:

                if self.value is None:
                    raise ValueError(
                        "No constant value given in constant distribution."
                    )
                else:
                    samples = [self.value] * self.num_samples
            
            case DistributionType.UNIFORM:
                
                if self.min_ is None:
                     raise ValueError(
                        "No minumum value given in uniform distribution."
                    )
                elif self.max_ is None:
                    raise ValueError(
                        "No maximum value given in uniform distribution."
                    )
                else:                
                    samples = \
                        np.random.uniform(
                            min_, 
                            max_, 
                            self.num_samples
                        )
                    
            case DistributionType.NORMAL:
                
                if self.mean is None:
                    raise ValueError(
                        "No mean value given in normal distribution."
                    )
                elif self.std is None:
                    raise ValueError(
                        "No std value given in normal distribution."
                else:
                        
                    if self.min_ is None:
                        self.min_ = float("-inf")
                    elif self.max_ is None:
                        self.max_ = float("inf")
                    
                    samples = \
                        truncnorm.rvs(
                            (self.min_ - self.mean) / self.std,
                            (self.max_ - self.mean) / self.std,
                            loc=self.mean_value,
                            scale=self.std,
                            size=self.num_samples
                        )
            
            case _:
                raise ValueError('Unsupported distribution type')

        if self.dtype == int:
            samples = [int(sample) for sample in samples]
        
        samples = samples if self.num_samples > 1 else samples[0]
        return samples
    
def randomise_arguments(input_dict, func):
    output_dict = {}
    for key, value in input_dict.items():
        output_dict[key] = randomise_dict(value)

    return func(**output_dict), output_dict


@tf.function
def replace_nan_and_inf_with_zero(tensor):
    tensor = tf.where(tf.math.is_nan(tensor), tf.zeros_like(tensor), tensor)
    tensor = tf.where(tf.math.is_inf(tensor), tf.zeros_like(tensor), tensor)
    return tensor    