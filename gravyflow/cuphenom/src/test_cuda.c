#ifndef _MSC_VER
#define __STDC_WANT_LIB_EXT2__  1
#endif

#include <math.h>
#include <inttypes.h>
#include <time.h>

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

#include "cuda_phenom.h"
#include "python_interface.h"

int32_t main(){
	const float mass_1_msun         =   30.0f;
    const float mass_2_msun         =   30.0f;
    const float sample_rate_hertz   = 8192.0f;
    const float duration_seconds    =    1.0f;
    const float inclination_radians =    1.0f;
    const float distance_mpc        = 1000.0f;
	
	strain_element_t *strain = NULL;
    
	const massUnit_t      mass_1      = initMassSolarMass(mass_1_msun);
	const massUnit_t      mass_2      = initMassSolarMass(mass_2_msun);
    const frequencyUnit_t sample_rate = initFrequencyHertz(sample_rate_hertz);
    const angularUnit_t   inclination = initAngleRadians(inclination_radians);
	const lengthUnit_t    distance    = initLengthMpc(distance_mpc);
	const timeUnit_t      duration    = initTimeSeconds(duration_seconds);
		    
	// LAL:
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
	
	return 0;
}