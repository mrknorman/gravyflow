#ifndef _PyCuPhenom_H
#define _PyCuPhenom_H

#include "cuda_phenom.h"

float *pythonWrapperPhenomD(
    const int32_t num_waveforms,
    const float   sample_rate_hertz,
    const float   duration_seconds,
    const float  *mass_1_msun,
    const float  *mass_2_msun,
    const float  *inclination_radians,
    const float  *distance_mpc,
    const float  *reference_orbital_phase_in,
    const float  *ascending_node_longitude,
    const float  *eccentricity,
    const float  *mean_periastron_anomaly,
          float  *spin_1_in,
          float  *spin_2_in
    ) {
    
    // Constants:
    const frequencyUnit_t sample_rate = 
        initFrequencyHertz(sample_rate_hertz);
    const timeUnit_t      duration = 
        initTimeSeconds(duration_seconds);
    timeUnit_t      time_interval        = 
        initTimeSeconds(1.0f/sample_rate.hertz);
    approximant_e approximant = D; 
    
    float           redshift            = 0.0f;
    frequencyUnit_t reference_frequency = initFrequencyHertz(0.0f);
    
    // Init property structures:    
    system_properties_s   system_properties[num_waveforms];
    temporal_properties_s temporal_properties[num_waveforms];
    
    for (int32_t index = 0; index < num_waveforms; index++)
    {
        const massUnit_t      mass_1       = 
            initMassSolarMass(mass_1_msun[index]);
        const massUnit_t      mass_2       = 
            initMassSolarMass(mass_2_msun[index]);
                    
        const angularUnit_t   inclination  = 
            initAngleRadians(inclination_radians[index]);
        const lengthUnit_t    distance     = 
            initLengthMpc(distance_mpc[index]);
        const angularUnit_t   reference_orbital_phase = 
            initAngleRadians(reference_orbital_phase_in[index]);
            
        float *spin_1_elements = &spin_1_in[index*NUM_SPIN_DIMENSIONS];
        float *spin_2_elements = &spin_2_in[index*NUM_SPIN_DIMENSIONS];
        
        // Setup companion structures:
        spin_t spin_1 = 
        {
            .x = spin_1_elements[0],
            .y = spin_1_elements[1],
            .z = spin_1_elements[2]
        };
        spin_t spin_2 = 
        {
            .x = spin_2_elements[0],
            .y = spin_2_elements[1],
            .z = spin_2_elements[2]
        };
        companion_s companion_a = 
        {
            .mass              = mass_1,
            .spin              = spin_1,
            .quadrapole_moment = 0.0f,
            .lambda            = 0.0f
        };
        companion_s companion_b = 
        {
            .mass              = mass_2,
            .spin              = spin_2,
            .quadrapole_moment = 0.0f,
            .lambda            = 0.0f
        };

        frequencyUnit_t starting_frequency = 
            calcMinimumFrequency(
                mass_1, 
                mass_2, 
                duration
            );
    
        system_properties[index] =
            initBinarySystem(
                companion_a,
                companion_b,
                distance,
                redshift,
                inclination,
                reference_orbital_phase,
                ascending_node_longitude[index],
                eccentricity[index], 
                mean_periastron_anomaly[index]
            );

        temporal_properties[index] =
            initTemporalProperties(
                time_interval,       // <-- Sampling interval (timeUnit_t).
                starting_frequency,  // <-- Starting GW frequency (frequencyUnit_t).
                reference_frequency, // <-- Reference GW frequency (frequencyUnit_t).
                system_properties[index],
                approximant
            );
    }
    
    waveform_axes_s waveform_axes =
        generateCroppedInspiral(
            system_properties,
            temporal_properties,
            sample_rate, 
            duration, 
            num_waveforms,
            approximant
        );
        
    const int32_t num_samples = waveform_axes.strain.total_num_samples;   
            
    strain_element_t *strain = NULL;
    cudaToHost(
        (void*)waveform_axes.strain.values, 
        sizeof(strain_element_t),
        num_samples,
        (void**)&strain
    );
    
    freeWaveformAxes(waveform_axes);
            
    float *return_array = (float*)
        malloc(
              sizeof(float) 
            * (size_t)num_samples
            * (size_t)NUM_POLARIZATION_STATES
        );
    
    for (int32_t index = 0; index < num_samples; index++)
    {
        return_array[NUM_POLARIZATION_STATES*index + 0] = strain[index].plus;
        return_array[NUM_POLARIZATION_STATES*index + 1] = strain[index].cross;
    }
    
    return return_array;
}

#endif