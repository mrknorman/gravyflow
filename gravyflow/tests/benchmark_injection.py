
import time
import numpy as np
import gravyflow as gf
from gravyflow.src.dataset.features.injection import InjectionGenerator, RippleGenerator, ScalingMethod, ScalingTypes

def benchmark_scenario(name, mass, padding, num_batches=5, batch_size=4):
    print(f"Benchmarking: {name}")
    print(f"  Mass: {mass}, Padding: {padding}")
    
    generator = RippleGenerator(
        mass_1_msun=mass,
        mass_2_msun=mass,
        distance_mpc=100.0,
        front_padding_duration_seconds=padding,
        back_padding_duration_seconds=padding,
        scaling_method=ScalingMethod(
            value=gf.Distribution(value=10.0, type_=gf.DistributionType.CONSTANT), 
            type_=ScalingTypes.SNR
        ),
        network=gf.Network([gf.IFO.L1, gf.IFO.H1])
    )
    
    injection_gen = InjectionGenerator(
        waveform_generators=[generator],
        seed=42
    )
    
    # Warmup
    iterator = injection_gen(
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=4.0,
        crop_duration_seconds=0.5,
        num_examples_per_batch=batch_size
    )
    next(iterator)
    
    start_time = time.time()
    for _ in range(num_batches):
        next(iterator)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_batches
    print(f"  Average time per batch: {avg_time:.4f} s")
    return avg_time

if __name__ == "__main__":
    # Scenario 1: High mass (short signal), small padding
    # This represents the "baseline" where impact should be minimal (mostly overhead)
    t1 = benchmark_scenario("High Mass, Small Padding", 30.0, 0.1)
    
    # Scenario 2: High mass (short signal), large padding
    # This tests the impact of the buffer calculation
    t2 = benchmark_scenario("High Mass, Large Padding", 30.0, 2.0)
    
    # Scenario 3: Low mass (long signal), small padding
    # This tests the impact of forcing full signal generation
    t3 = benchmark_scenario("Low Mass, Small Padding", 10.0, 0.1)
    
    with open("benchmark_results.txt", "w") as f:
        f.write("Summary:\n")
        f.write(f"High Mass, Small Padding: {t1:.4f} s\n")
        f.write(f"High Mass, Large Padding: {t2:.4f} s\n")
        f.write(f"Low Mass, Small Padding : {t3:.4f} s\n")
        
    print("\nSummary:")
    print(f"High Mass, Small Padding: {t1:.4f} s")
    print(f"High Mass, Large Padding: {t2:.4f} s")
    print(f"Low Mass, Small Padding : {t3:.4f} s")
