from maths import Distribution, DistibutionType
from numpy import np

@dataclass
class WNBConfig:
    nothing_yet : int = 1.0
    
@dataclass
class cuPhenomDConfig:
    mass_1_msun : Distribution = \
        Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM)
    mass_2_msun : Distribution = \
        Distribution(min_=5.0, max_=95.0, type_=DistributionType.UNIFORM)
    inclination_radians : Distribution = \
        Distribution(min_=0.0, max_=np.pi, type_=DistributionType.UNIFORM)
    distance_mpc : Distribution = \
        Distribution(value=1000.0, type_=DistributionType.CONSTANT)
    reference_orbital_phase_in : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    ascending_node_longitude : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    eccentricity : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    mean_periastron_anomaly : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    spin_1_in : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)
    spin_2_in : Distribution = \
        Distribution(value=0.0, type_=DistributionType.CONSTANT)

@dataclass
class InjectionGenerator:
    sample_rate_hertz : float
    duration_seconds : float
    config : Union[cuPhenomDConfig, WNBConfig]
    snr : Union[np.ndarray, Distribution] = \
        Distribution(value=50.0, type_=DistributionType.CONSTANT)
    front_padding_duration_seconds : float = 0.3
    back_padding_duration_seconds : float = 0.0
