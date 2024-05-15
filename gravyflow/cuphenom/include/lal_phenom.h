#ifndef LAL_PHENOM_H
#define LAL_PHENOM_H

#include <lal/LALDict.h>
#include <lal/LALSimInspiral.h>

void generatePhenomLAL(
	const Approximant       approximant,
    const massUnit_t        mass_1, 
    const massUnit_t        mass_2, 
    const frequencyUnit_t   sample_rate, 
    const timeUnit_t        duration, 
    const angularUnit_t     inclination, 
    const lengthUnit_t      distance, 
          float2_t        **ret_strain
    ) {
	
	const int32_t num_samples = 
		(int32_t)floorf(sample_rate.hertz*duration.seconds);
	
	REAL8TimeSeries *hplus  = NULL;
	REAL8TimeSeries *hcross = NULL;
	
	double S1x          = 0.0;
	double S1y          = 0.0;
	double S1z          = 0.0;
	double S2x          = 0.0;
	double S2y          = 0.0;
	double S2z          = 0.0;
	
	double phiRef       = 0.0;
	double longAscNodes = 1.0;
	double eccentricity = 0.0;
	double meanPerAno   = 0.0;
	double deltaT       = 1.0/sample_rate.hertz;
	double f_min        = 
		calcMinimumFrequency(
			mass_1, 
			mass_2, 
			duration
		).hertz;
	
	double        f_ref       = 0.0;
	LALDict     *extraParams = NULL;
    
	//Converting to SI:
	XLALSimInspiralTD(
		&hplus,
		&hcross,
		(double)mass_1.kilograms,
		(double)mass_2.kilograms,
		S1x,
		S1y,
		S1z,
		S2x,
		S2y,
		S2z,
		(double)distance.meters,
		(double)inclination.radians,
		phiRef,
		longAscNodes,
		eccentricity,
		meanPerAno,
		deltaT,
		f_min,
		f_ref,
		extraParams,
		approximant
	);
    
	const int32_t waveform_num_samples = (int32_t)hplus->data->length;
	
	if (waveform_num_samples < num_samples) 
	{	
		fprintf(
			stderr, 
			"Warning! LAL Simulation not generating waveforms of desired "
			"num_samples.\n"
		);
	}
	
	size_t new_array_size = (size_t)num_samples * sizeof(float2_t);

	float2_t *strain = (float2_t*)malloc(new_array_size);
	int32_t new_waveform_index = 0;
    for (int32_t index = 0; index < num_samples; index++) 
	{	
		new_waveform_index = waveform_num_samples - num_samples - 1 + index;
		strain[index].x = (float)hplus->data->data[new_waveform_index];
		strain[index].y = (float)hcross->data->data[new_waveform_index];
    }
	
	free(hcross->data->data); free(hplus->data->data);

	*ret_strain = strain;
}

#endif