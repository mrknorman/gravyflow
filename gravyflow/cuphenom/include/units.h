#ifndef CUDA_UNITS_H
#define CUDA_UNITS_H

#include <stdarg.h>
#include <gsl/gsl_const_mksa.h>

// Hardwired constants:
#define AU_METERS 149597870700.0f
#define PC_METERS AU_METERS * (648000.0f / (float)M_PI) //-- A parsec in m
#define MPC_METERS PC_METERS * 10E6f
#define SIDEREAL_YEAR_SECONDS 365.256363004f*24.0f*3600.0f

#define G_SI 6.67430e-11f //6.67408E-11f  //-- the gravitational constant in m^3/(kilogram*s^2)
#define MASS_SUN_KILOGRAMS   (4.0f*(float)(M_PI*M_PI)*AU_METERS*AU_METERS*AU_METERS) \
						   / (G_SI*SIDEREAL_YEAR_SECONDS*SIDEREAL_YEAR_SECONDS)

#define C_SI 299792458.0f //-- the speed of light in m/s

#define EULER_MASCHERONI 0.577215664901532860606512090082402431f
#define NUM_POLARIZATION_STATES 2
#define NUM_SPIN_DIMENSIONS 3

typedef struct {
	float x; 
	float y;
} float2_t;

// Mass functions:

typedef struct {
	float msun;
	float kilograms;
	float seconds;
	float meters;
} massUnit_t;

inline float kilogramsToSeconds(const float mass_kilogram)
{
	return mass_kilogram*G_SI / (C_SI*C_SI*C_SI);
}

inline float kilogramsToMeters(const float mass_kilogram)
{
	return mass_kilogram*G_SI / (C_SI*C_SI);
}

inline float kilogramsToMsun(const float mass_kilogram)
{
	return (float)((double)mass_kilogram/(double)MASS_SUN_KILOGRAMS);
}

inline float msunToKilograms(const float mass_msun)
{
	return mass_msun*MASS_SUN_KILOGRAMS;
}

massUnit_t initMassSolarMass(
	const float mass_msun
) {
	massUnit_t mass = {
		.msun      = mass_msun,
		.kilograms = msunToKilograms   (mass_msun),
		.seconds   = kilogramsToSeconds(msunToKilograms(mass_msun)),
		.meters    = kilogramsToMeters (msunToKilograms(mass_msun))
	};
	
	return mass;
}

massUnit_t initMassKilograms(
	const float mass_kilograms
) {
	massUnit_t mass = {
		.msun      = kilogramsToMsun(mass_kilograms),
		.kilograms = mass_kilograms,
		.seconds   = kilogramsToSeconds(mass_kilograms),
		.meters    = kilogramsToMeters (mass_kilograms)
	};
	
	return mass;
}

massUnit_t scaleMass(
	const massUnit_t    mass, 
	const float scalar
) {
	massUnit_t scaled = {
		.msun      = mass.msun      * scalar,
		.kilograms = mass.kilograms * scalar,
		.seconds   = mass.seconds   * scalar,
		.meters    = mass.meters    * scalar
	};
	
	return scaled;
}

massUnit_t addMasses(
	const massUnit_t mass_1, 
	const massUnit_t mass_2
) {
	massUnit_t sum = {
		.msun      = mass_1.msun      + mass_2.msun,
		.kilograms = mass_1.kilograms + mass_2.kilograms,
		.seconds   = mass_1.seconds   + mass_2.seconds,
		.meters    = mass_1.meters    + mass_2.meters
	};
	
	return sum;
}

massUnit_t subtractMasses(
	const massUnit_t mass_1, 
	const massUnit_t mass_2
) {
	massUnit_t difference = {	
		.msun      = mass_1.msun      - mass_2.msun,
		.kilograms = mass_1.kilograms - mass_2.kilograms,
		.seconds   = mass_1.seconds   - mass_2.seconds,
		.meters    = mass_1.meters    - mass_2.meters
	};
	
	return difference;
}

massUnit_t multiplyMasses(
	const massUnit_t mass_1, 
	const massUnit_t mass_2
) {
	massUnit_t product = {
		.msun      = mass_1.msun      * mass_2.msun,
		.kilograms = mass_1.kilograms * mass_2.kilograms,
		.seconds   = mass_1.seconds   * mass_2.seconds,
		.meters    = mass_1.meters    * mass_2.meters
	};
	
	return product;
}

massUnit_t divideMasses(
	const massUnit_t mass_1, 
	const massUnit_t mass_2
) {
	massUnit_t quotient = {
		.msun      = mass_1.msun      / mass_2.msun,
		.kilograms = mass_1.kilograms / mass_2.kilograms,
		.seconds   = mass_1.seconds   / mass_2.seconds,
		.meters    = mass_1.meters    / mass_2.meters
	};
	
	return quotient;
}

// Distance functions:
typedef struct {
	float Mpc;
	float meters;
} lengthUnit_t;

inline float MpcToMeters(const float length_mpc)
{
	return length_mpc*MPC_METERS;
}

inline float MeterToMPc(const float length_meters)
{
	return length_meters/MPC_METERS;
}

lengthUnit_t initLengthMpc(
	const float length_mpc
) {
	lengthUnit_t length = {
		 .Mpc    = length_mpc,
		 .meters = MpcToMeters(length_mpc)
	 };
	
	return length;
}

lengthUnit_t initLengthMeters(
	const float length_meters
) {
	lengthUnit_t length = {
		 .Mpc    = MeterToMPc(length_meters),
		 .meters = length_meters
	 };
	
	return length;
}

lengthUnit_t scaleLength(
	const lengthUnit_t  length, 
	const float scalar
) {
	lengthUnit_t scaled = {
		.Mpc    = length.Mpc    * scalar,
		.meters = length.meters * scalar
	};
	
	return scaled;
}

// Time Functions:

typedef struct {
	float seconds;
} timeUnit_t;

timeUnit_t initTimeSeconds(
	float time_seconds
	) {
	
	timeUnit_t time = {
		 .seconds = time_seconds
	 };
	
	return time;
}

timeUnit_t _addTimes(
	const timeUnit_t time_1,
	const timeUnit_t time_2
) {
	timeUnit_t sum = {
		.seconds = time_1.seconds + time_2.seconds
	};
	
	return sum;
}

timeUnit_t addTimes(
	const int32_t num_args,
	...
	) {

	va_list valist;
   	timeUnit_t sum = {
		.seconds = 0.0
	};

   va_start(valist, num_args);
   for (int32_t index = 0; index < num_args; index++) 
   {
      sum = _addTimes(sum, (timeUnit_t)va_arg(valist, timeUnit_t));
   }
   va_end(valist);

   return sum;
}

timeUnit_t multiplyTimes(
	const timeUnit_t time_1,
	const timeUnit_t time_2
) {
	timeUnit_t product = {
		.seconds = time_1.seconds * time_2.seconds
	};
	
	return product;
}

timeUnit_t scaleTime(
	const timeUnit_t time_1,
	const float      scale
) {
	timeUnit_t scaled = {
		.seconds = time_1.seconds * scale
	};
	
	return scaled;
}

// Frequency Functions:
typedef struct {
	float hertz;
} frequencyUnit_t;

frequencyUnit_t initFrequencyHertz(
	const float frequency_hertz
	) {
	
	frequencyUnit_t frequency = {
		 .hertz = frequency_hertz
	 };
	
	return frequency;
}

// Angle Functions:
typedef struct {
	float radians;
} angularUnit_t;

angularUnit_t initAngleRadians(
	float angle_radians
	) {
	
	angularUnit_t angle = {
		 .radians = angle_radians
	 };
	
	return angle;
}


#endif