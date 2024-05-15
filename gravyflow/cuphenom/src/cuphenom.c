#ifndef _MSC_VER
#define __STDC_WANT_LIB_EXT2__  1
#endif

#include <math.h>
#include <inttypes.h>
#include <time.h>

#include "./py_tools/py_tools.h"
#include "units.h"
#include "io_tools/console.h"

frequencyUnit_t calcMinimumFrequency(
    const massUnit_t mass_1,   //<-- Mass of first object.
    const massUnit_t mass_2,   //<-- Mass of secondary object.
    const timeUnit_t duration //<-- Duration of signal.
) {
    /*
     * Calculates minimum frequency based on inputted masses.
     */
    
    const double MC  = 
		pow(
			((((double)mass_1.kilograms*(double)mass_2.kilograms)*((double)mass_1.kilograms*(double)mass_2.kilograms)*
			  ((double)mass_1.kilograms*(double)mass_2.kilograms))/((double)mass_1.kilograms+(double)mass_2.kilograms)),
			(1.0/5.0));
    double min_frequency_hertz = pow(((double)duration.seconds/5.0),(-3.0/8.0))*(1.0/(8.0*M_PI))
            *(pow((G_SI*MC/(C_SI*C_SI*C_SI)),(-5.0/8.0)));
  
    min_frequency_hertz =
        (1.0 > min_frequency_hertz) + (1.0 <= min_frequency_hertz)
        *min_frequency_hertz;
    
    return initFrequencyHertz((float)min_frequency_hertz);
}

frequencyUnit_t calcMinimumFrequency2(
    const massUnit_t mass_1,   //<-- Mass of first object.
    const massUnit_t mass_2,   //<-- Mass of secondary object.
    const timeUnit_t duration //<-- Duration of signal.
) {
    /*
     * Calculates minimum frequency based on inputted masses.
     */
    
    const float MC = (float)
		pow(
			(((mass_1.kilograms*mass_2.kilograms)
              *(mass_1.kilograms*mass_2.kilograms)*
			  (mass_1.kilograms*mass_2.kilograms))
             /(mass_1.kilograms+mass_2.kilograms)),
			(1.0/5.0)
        );
    float min_frequency_hertz = 
        powf((duration.seconds/5.0f),(-3.0f/8.0f))
        *(1.0f/(8.0f*(float)M_PI))
        *(powf((G_SI*MC/(C_SI*C_SI*C_SI)),(-5.0f/8.0f)));
  
    min_frequency_hertz =
        (1.0f > min_frequency_hertz) + (1.0f <= min_frequency_hertz)
        *min_frequency_hertz;
    
    return initFrequencyHertz(min_frequency_hertz);
}

#include "cuda_phenom.h"
#include "lal_phenom.h"
#include "python_interface.h"

#ifdef ZOMBIE
#include "zombie_phenom.h"
#endif

void addLinearArray(
          float   *array, 
    const float    min, 
    const float    max, 
    const int32_t  num_elements
    ) {
	
	// Calculate the increase in value between two subsequent array indices:
	const float step_increase = (max - min) / (float)num_elements; 

	for (int32_t index = 0; index < num_elements; index++) 
    {	
		array[index] = min + ((float)index)*step_increase;
	}
}

int32_t plotWaveform(
    const int32_t      verbosity,
	const timeUnit_t   duration,
          float2_t *strain,
	const int32_t      num_samples,
    const char        *output_file_name
    ) {
	
	const size_t array_size = sizeof(float)*(size_t)num_samples;
	
	float *h_cross = malloc(array_size);
	float *h_plus  = malloc(array_size);
	float *duration_array = malloc(array_size); 

	for (int32_t index = 0; index < num_samples; index++)
	{
		h_cross[index] = (float)strain[index].x;
		h_plus[index]  = (float)strain[index].y;
	}	        
    addLinearArray(
        duration_array, 
        0.0f, 
        (float)duration.seconds, 
        num_samples
    );

    // Setup axis:
    int32_t num_axis = NUM_POLARIZATION_STATES;
    axis_s y_axis = {
        .name = "strain",
        .label = "Strain"
    };
    axis_s x_axis = {
        .name = "time",
        .label = "Time (Seconds)"
    };
    axis_s axis_array[] = {
        x_axis,
        y_axis
    };

    // Setup axes:
	axes_s h_cross_axes = {
        .name     = "h_cross",
        .num_axis = num_axis,        
        .axis     = axis_array
    };
    axes_s h_plus_axes = {
        .name     = "h_plus",
        .num_axis = num_axis,        
        .axis     = axis_array
    };

    // Setup series:
    series_values_s duration_values = {
        .axis_name = "time",
        .values    = duration_array
    };
    series_values_s h_cross_values = {
        .axis_name = "strain",
        .values    = h_cross
    };
    series_values_s h_plus_values = {
        .axis_name = "strain",
        .values    = h_plus
    };
    
    series_values_s h_cross_series_values[] = {
        duration_values,
        h_cross_values
    };
	series_values_s h_plus_series_values[] = {
        duration_values,
        h_plus_values
    };

    series_s h_cross_series = {
        .label         = "h_cross",
        .axes_name     = "h_cross",
        .num_elements  = num_samples,
        .num_axis      = num_axis,
        .values        = h_cross_series_values
    };
    series_s h_plus_series = {
        .label         = "h_plus",
        .axes_name     = "h_plus",
        .num_elements  = num_samples,
        .num_axis      = num_axis,
        .values        = h_plus_series_values
    };

    axes_s axes[] = {
        h_cross_axes,
        h_plus_axes
    };
    series_s series[] = {
        h_cross_series,
        h_plus_series
    };

    figure_s figure = {
        .axes       = axes,
        .num_axes   = NUM_POLARIZATION_STATES,
        .series     = series,
        .num_series = NUM_POLARIZATION_STATES,
    };
	
    plotFigure(
        verbosity,
        figure,
        output_file_name
    ); 
    
    free(duration_array);
	free(h_plus);
	free(h_cross);
	
	return 0;
}

int32_t plotWaveformComparison(
    const int32_t      verbosity,
	const timeUnit_t   duration,
          float2_t *strain_one,
		  char        *strain_one_name,
	      float2_t *strain_two,
		  char        *strain_two_name,
	const int32_t      num_samples,
    const char        *output_file_name
    ) {
	
	const size_t array_size = sizeof(float)*(size_t)num_samples;
	
	float *h_cross_one = malloc(array_size), *h_cross_two = malloc(array_size);
	float *h_plus_one  = malloc(array_size), *h_plus_two = malloc(array_size);
	float *duration_array = malloc(array_size); 

	for (int32_t index = 0; index < num_samples; index++)
	{
		h_cross_one[index] = (float)strain_one[index].x; 		
		h_cross_two[index] = (float)strain_two[index].x; 

		h_plus_one[index]  = (float)strain_one[index].y;
		h_plus_two[index]  = (float)strain_two[index].y;
	}	        
    addLinearArray(
        duration_array, 
        0.0f, 
        (float)duration.seconds, 
        num_samples
    );

    // Setup axis:
    int32_t num_axis = NUM_POLARIZATION_STATES;
    axis_s y_axis = {
        .name = "strain",
        .label = "Strain"
    };
    axis_s x_axis = {
        .name = "time",
        .label = "Time (Seconds)"
    };
    axis_s axis_array[] = {
        x_axis,
        y_axis
    };

    // Setup axes:
	axes_s h_cross_axes = {
        .name     = "h_cross",
        .num_axis = num_axis,        
        .axis     = axis_array
    };
    axes_s h_plus_axes = {
        .name     = "h_plus",
        .num_axis = num_axis,        
        .axis     = axis_array
    };

    // Setup series:
    series_values_s duration_values = {
        .axis_name = "time",
        .values    = duration_array
    };
    series_values_s h_cross_one_values = {
        .axis_name = "strain",
        .values    = h_cross_one
    };
    series_values_s h_plus_one_values = {
        .axis_name = "strain",
        .values    = h_plus_one
    };
	
	series_values_s h_cross_two_values = {
        .axis_name = "strain",
        .values    = h_cross_two
    };
    series_values_s h_plus_two_values = {
        .axis_name = "strain",
        .values    = h_plus_two
    };
    
    series_values_s h_cross_one_series_values[] = {
        duration_values,
        h_cross_one_values
    };
	series_values_s h_plus_one_series_values[] = {
        duration_values,
        h_plus_one_values
    };
	series_values_s h_cross_two_series_values[] = {
        duration_values,
        h_cross_two_values
    };
	series_values_s h_plus_two_series_values[] = {
        duration_values,
        h_plus_two_values
    };

    series_s h_cross_one_series = {
        .label         = strain_one_name,
        .axes_name     = "h_cross",
        .num_elements  = num_samples,
        .num_axis      = num_axis,
        .values        = h_cross_one_series_values
    };
    series_s h_plus_one_series = {
        .label         = strain_one_name,
        .axes_name     = "h_plus",
        .num_elements  = num_samples,
        .num_axis      = num_axis,
        .values        = h_plus_one_series_values
    };
	
	series_s h_cross_two_series = {
        .label         = strain_two_name,
        .axes_name     = "h_cross",
        .num_elements  = num_samples,
        .num_axis      = num_axis,
        .values        = h_cross_two_series_values
    };
    series_s h_plus_two_series = {
        .label         = strain_two_name,
        .axes_name     = "h_plus",
        .num_elements  = num_samples,
        .num_axis      = num_axis,
        .values        = h_plus_two_series_values
    };

    axes_s axes[] = {
        h_cross_axes,
        h_plus_axes
    };
    series_s series[] = {
        h_cross_one_series,
		h_cross_two_series,
        h_plus_one_series,
		h_plus_two_series,
    };

    figure_s figure = {
        .axes       = axes,
        .num_axes   = NUM_POLARIZATION_STATES,
        .series     = series,
        .num_series = NUM_POLARIZATION_STATES*2,
    };
	
    plotFigure(
        verbosity,
        figure,
        output_file_name
    ); 
    
    free(duration_array);
	free(h_plus_one);
	free(h_plus_two);
	free(h_cross_one);
	free(h_cross_two);
	
	return 0;
}

int32_t testRunTime(
	const Approximant     approximant,
	const massUnit_t      mass_1,
    const massUnit_t      mass_2,
    const frequencyUnit_t sample_rate,
    const timeUnit_t      duration,
    const angularUnit_t   inclination,
    const lengthUnit_t    distance,
	const int32_t         num_tests
	) {
        
	float2_t *strain = NULL;
	
	float execution_time_lal = 0.0, execution_time_cuda = 0.0;
        
    timer_s timer;
    start_timer("Timer", &timer);
    
    printf("Test LAL runtime... \n");
    //LAL
	for (int32_t index = 0; index < num_tests; index++)
	{
		generatePhenomLAL(
			approximant,
			mass_1, 
			mass_2, 
			sample_rate, 
			duration, 
			inclination, 
			distance, 
			&strain
		);
		free(strain);
	}	
    execution_time_lal = stop_timer(&timer);
       
    printf("Test cuda runtime... \n");
	// CUDA:
	for (int32_t index = 0; index < num_tests; index++)
	{
		generatePhenomCUDA(
			D,
			mass_1,
			mass_2, 
			sample_rate, 
			duration, 
			inclination, 
			distance, 
			&strain
		);
		free(strain);
	}  
	execution_time_cuda = stop_timer(&timer);
	
	printf("LAL: %f, CUDA: %f. \n", 
        execution_time_lal, 
        execution_time_cuda
    );
	
	return 0;
}


int32_t main(){
	const float mass_1_msun         =   30.0f;
    const float mass_2_msun         =   30.0f;
    const float sample_rate_hertz   = 8192.0f;
    const float duration_seconds    =    1.0f;
    const float inclination_radians =    1.0f;
    const float distance_mpc        = 1000.0f;
	
	float2_t *lal_strain = NULL, *cuda_strain = NULL;
    
    #ifdef ZOMBIE 
    float2_t *zombie_strain = NULL;
    #endif
		
	const massUnit_t      mass_1      = initMassSolarMass(mass_1_msun);
	const massUnit_t      mass_2      = initMassSolarMass(mass_2_msun);
    const frequencyUnit_t sample_rate = initFrequencyHertz(sample_rate_hertz);
    const angularUnit_t   inclination = initAngleRadians(inclination_radians);
	const lengthUnit_t    distance    = initLengthMpc(distance_mpc);
	const timeUnit_t      duration    = initTimeSeconds(duration_seconds);
	
	const int32_t num_samples = 
		(int32_t)floor(sample_rate.hertz*duration.seconds);
		
	//Approximant  approximant = IMRPhenomXPHM;
	Approximant  approximant = IMRPhenomD; 
    
	// LAL:
	generatePhenomLAL(
		approximant,
		mass_1, 
		mass_2, 
		sample_rate, 
		duration, 
		inclination, 
		distance, 
		&lal_strain
	);
    
    printf("Here! \n");
            
    generatePhenomCUDA(
		D,
		mass_1,
		mass_2, 
		sample_rate, 
		duration, 
		inclination, 
		distance, 
		&cuda_strain
    );
    
    #ifdef ZOMBIE
    generatePhenomZombie(
		approximant,
		mass_1,
		mass_2, 
		sample_rate, 
		duration, 
		inclination, 
		distance, 
		&zombie_strain
    );
    #endif
    
	float2_t *difference     = malloc(sizeof(float2_t)*(size_t)num_samples);
	float2_t  sum            = {.x = 0.0f, .y = 0.0f};
    float2_t  difference_sum = {.x = 0.0f, .y = 0.0f};

	for (int32_t index = 0; index < num_samples; index++)
	{
		difference[index].x = fabsf(lal_strain[index].x - cuda_strain[index].x);
		difference[index].y = fabsf(lal_strain[index].y - cuda_strain[index].y);
		
        sum.x += fabsf(lal_strain[index].x); sum.y += fabsf(lal_strain[index].y); 
		difference_sum.x += difference[index].x; 
        difference_sum.y += difference[index].y;
	}
	
	printf("Difference: %.4f%%, %.4f%% \n", 
        (difference_sum.x/sum.x) * 100.0f,
        (difference_sum.y/sum.y) * 100.0f);
	
	// Plotting:
		
	char *output_file_name = 
		"../cuphenom_outputs/waveform_tests/lal_waveform";
		
	plotWaveform(
		STANDARD,
		duration,
        lal_strain,
		num_samples,
		output_file_name
    );
	
	output_file_name =  
        "../cuphenom_outputs/waveform_tests/cuda_waveform";
	
	plotWaveform(
		STANDARD,
		duration,
        cuda_strain,
		num_samples,
		output_file_name
    );
	
	output_file_name =  "../cuphenom_outputs/waveform_tests/comparison";
	plotWaveformComparison(
		STANDARD,
		duration,
		lal_strain,
	    "lal_waveform",
	    cuda_strain,
		"cuda_waveform",
	    num_samples,
        output_file_name
    );
    
    #ifdef ZOMBIE
    output_file_name =  "../cuphenom_outputs/waveform_tests/zombie_comparison";
	plotWaveformComparison(
		STANDARD,
		duration,
		lal_strain,
	    "lal_waveform",
	    zombie_strain,
		"zombie_waveform",
	    num_samples,
        output_file_name
    );
    
    free(zombie_strain);
    #endif
	
	free(lal_strain); free(cuda_strain);
	
	testRunTime(
		approximant,
		mass_1,
		mass_2,
		sample_rate,
		duration,
		inclination,
		distance,
		100
	);
	
	return 0;
}