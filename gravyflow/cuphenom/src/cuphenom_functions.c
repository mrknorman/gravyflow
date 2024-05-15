#ifndef _MSC_VER
#define __STDC_WANT_LIB_EXT2__  1
#endif

#include <math.h>
#include <inttypes.h>
#include <time.h>

#include "units.h"
#include "io_tools/console.h"

// Sort this out at some point
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