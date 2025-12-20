import logging
import numpy as np
import gravyflow as gf
from gravyflow.src.dataset.noise.noise import TransientObtainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wrapper_glitch_retrieval():
    print("Reproducing issue with TransientObtainer wrapper...")
    
    # Configure for glitch acquisition
    ifo_data_obtainer = gf.IFODataObtainer(
        data_quality=gf.DataQuality.BEST,
        data_labels=[gf.DataLabel.GLITCHES],
        observing_runs=[gf.ObservingRun.O1], # O1 for speed
        saturation=1.0,
        random_sign_reversal=False,
        random_time_reversal=False,
        balanced_glitch_types=True,
    )

    # Create TransientObtainer for glitches
    glitch_obtainer = gf.TransientObtainer(
        ifo_data_obtainer=ifo_data_obtainer,
        ifos=[gf.IFO.L1],
    )
    
    # Check internal state before generation
    # ifo_data_obtainer is "fresh"
    print(f"Initial valid_segments: {ifo_data_obtainer.valid_segments}")
    
    # Create generator
    print("\nCreating generator...")
    # precache_cap=0 to disable precaching and force direct get_batch
    glitch_generator = glitch_obtainer(
        precache_cap=0, 
        sample_rate_hertz=2048.0,
        onsource_duration_seconds=1.0,
        offsource_duration_seconds=1.0,
        num_examples_per_batch=4,
        group="train", # Use 'train' to ensure segments found (if O1 has them)
        scale_factor=1.0,
        seed=42,
        crop=True,
        whiten=False
    )
    
    # After generator creation, ifo_data_obtainer (original) might be untouched or modified?
    # Reference: TransientObtainer copies it.
    print(f"Original valid_segments after gen creation: {ifo_data_obtainer.valid_segments}")
    
    # Get batch
    print("\nAcquiring batch...")
    try:
        # Note: this uses the internal COPY of the obtainer
        batch = next(glitch_generator)
        onsource = batch.get(gf.ReturnVariables.ONSOURCE)
        gps = batch.get(gf.ReturnVariables.TRANSIENT_GPS_TIME)
        glitch_types = batch.get(gf.ReturnVariables.GLITCH_TYPE)
        
        print(f"Batch acquired.")
        print(f"GPS times: {gps}")
        print(f"Glitch Types (raw): {glitch_types}")
        
        if glitch_types is not None:
             # Check for -1 (None equivalent)
             if np.all(glitch_types == -1):
                 print("ISSUE REPRODUCED: All glitch types are -1 (None)!")
             else:
                 print("Success: Glitch types found.")
                 print(glitch_types)
        else:
            print("Glitch types array is None?")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_wrapper_glitch_retrieval()
