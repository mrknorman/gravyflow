#ifndef PHENOM_STRUCTURES_HU
#define PHENOM_STRUCTURES_HU

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cufft.h>
#include <cuda_fp16.h>
#include <complex.h>

#include "phenom_d_data.h"

// Enums //

// Enum that specifies the PN approximant to be used in computing the waveform.

typedef enum {
    
    //Frequency domain (non-precessing spins) inspiral-merger-ringdown templates 
    // of Husa et al, arXiv:1508.07250 and Khan et al, arXiv:1508.07253 with 
    // phenomenological coefficients defined in the Table ... 
    // @remarks Implemented in lalsimulation (frequency domain).
    D, 
    // Frequency domain, precessing with subdominant modes phenomenological 
    // IMR waveform model.
    XPHM
} approximant_e;
 
// Enumeration of allowed PN orders of tidal effects. All effects up to and
// including the given order will be included in waveforms.
// Numerically, they are equal to twice the PN order, so e.g.
// LAL_SIM_INSPIRAL_TIDAL_ORDER_5PN = 10
// In addition, LAL_SIM_INSPIRAL_TIDAL_ORDER_ALL = -1
// is a flag to include all tidal PN orders up to the default
// value (which currently is 7PN for TaylorF2, 6PN for all
// other approximants):
typedef enum {
     TIDAL_ORDER_0PN  =  0,
     TIDAL_ORDER_5PN  = 10,
     TIDAL_ORDER_6PN  = 12,
     TIDAL_ORDER_65PN = 13,
     TIDAL_ORDER_7PN  = 14,
     TIDAL_ORDER_75PN = 15,
     TIDAL_ORDER_ALL  = -1,
} tidal_order_e;

// ~~~~~~~~~~ General structures ~~~~~~~~~~:

// Vector of dimensionless spins {x,y,z}:
typedef struct 
{ 
    float x;
    float y;
    float z;
} spin_t;

typedef struct 
{
    massUnit_t mass;              // <-- Mass of companion (massUnit_t).     
    spin_t     spin;              // <-- Vector of dimensionless spins {x,y,z}.
    float      quadrapole_moment;  
    float      lambda;
} companion_s;

typedef struct 
{
    // ~~~~ Binary Companions ~~~~~ //
    companion_s companion[2];
    
    // ~~~~ Mass properties ~~~~~ //
    massUnit_t total_mass;
    massUnit_t reduced_mass;
    float      symmetric_mass_ratio;
    
    // ~~~~ Distance properties ~~~~~ //
    float       redshift;
    
    // Distance of source (lengthUnit_t)@
    lengthUnit_t distance;             

    // ~~~~ Orbital properties ~~~~~ //
    
    // Reference orbital phase (angularUnit_t):
    angularUnit_t reference_orbital_phase; 
    
    // Longitude of ascending nodes, degenerate with the polarization angle:
    float ascending_node_longitude;

    // Inclination of source (angularUnit_t):
    angularUnit_t inclination;              
    
    // Eccentrocity at reference epoch:
    float eccentricity;             

    // Mean anomaly of periastron:
    float mean_periastron_anomaly;  
} system_properties_s;

typedef struct {
    // Time interval (timeUnit_t):
    timeUnit_t        time_interval;  
    
    // Frequency interval (frequencyUnit_t):
    frequencyUnit_t       frequency_interval;  
    
    // Starting GW frequency (frequencyUnit_t):
    frequencyUnit_t   starting_frequency;  
    
    // Ending GW Frequency (frequencyUnit_t):
    frequencyUnit_t   ending_frequency;  
    
    // Reference GW frequency (frequencyUnit_t):
    frequencyUnit_t   reference_frequency; 
    
    // Extra time to include for all waveforms to take care of situations where 
    // the frequency is close to merger (and is sweeping rapidly) this is a few 
    // cycles at the low frequency:
    timeUnit_t extra_time;
    
    // Upper bound on the chirp time starting at starting_frequency:
    timeUnit_t chirp_time_upper_bound;
     
    // Upper bound on the plunge and merger time:
    timeUnit_t merge_time_upper_bound;
    
    // Upper bound on the ringdown time:
    timeUnit_t ringdown_time_upper_bound;
    
    // Upper bound on the total time:
    timeUnit_t total_time_upper_bound;
     
} temporal_properties_s;

typedef struct 
{
    cuFloatComplex plus;
    cuFloatComplex cross;
} complex_strain_element_c;

typedef struct 
{
    complex float plus;
    complex float cross;
} complex_strain_element_t;

typedef struct 
{
    float plus;
    float cross;
} strain_element_t;

typedef struct
{
    frequencyUnit_t *values;
    frequencyUnit_t *interval_of_waveform;
    float           *num_samples_in_waveform;
    int32_t          max_num_samples_per_waveform;
    int32_t          total_num_samples;
} frequency_array_s;

typedef struct
{
    timeUnit_t *values;
    timeUnit_t *interval_of_waveform;
    float      *num_samples_in_waveform;
    int32_t     max_num_samples_per_waveform;
    int32_t     total_num_samples;
} time_array_s;

typedef struct
{
    complex_strain_element_c *values;
    float                    *num_samples_in_waveform;
    int32_t                   max_num_samples_per_waveform;
    int32_t                   total_num_samples;
} complex_strain_array_s;

typedef struct
{
    strain_element_t *values;
    float            *num_samples_in_waveform;
    int32_t           max_num_samples_per_waveform;
    int32_t           total_num_samples;
} strain_array_s;

// Useful powers in GW waveforms: 1/6, 1/3, 2/3, 4/3, 5/3, 2, 7/3, 8/3, -1, 
// -1/6, -7/6, -1/3, -2/3, -5/3 calculated using only one invocation of 'pow', 
// the rest are just multiplications and divisions:
typedef struct {
    float one;
    float third;
    float two_thirds;
    float four_thirds;
    float five_thirds;
    float two;
    float seven_thirds;
    float eight_thirds;
    float inv;
    float m_seven_sixths;
    float m_third;
    float m_two_thirds;
    float m_five_thirds;
} useful_powers_s;

// ~~~~~~~~~~ Amplitude structures ~~~~~~~~~~:

typedef struct {
   float eta;         // symmetric mass-ratio
   float etaInv;      // 1/eta
   float chi12;       // chi1*chi1;
   float chi22;       // chi2*chi2;
   float eta2;        // eta*eta;
   float eta3;        // eta*eta*eta;
   float Seta;        // sqrt(1.0 - 4.0*eta);
   float SetaPlus1;   // (1.0 + Seta);
   float chi1, chi2;  // dimensionless aligned spins, convention m1 >= m2.
   float q;           // asymmetric mass-ratio (q>=1)
   float chi;         // PN reduced spin parameter
   float ringdown_frequency;         // ringdown frequency
   float damping_time;         // imaginary part of the ringdown frequency (damping time)
  
   // Frequency at which the mrerger-ringdown amplitude is maximum:
   float inspiral_merger_peak_frequency;    
  
   // Phenomenological inspiral amplitude coefficients:
   float inspiral[NUM_AMPLITUDE_INSPIRAL_COEFFICIENTS];
  
   // Phenomenological intermediate amplitude coefficients:
   float intermediate[NUM_AMPLITUDE_INTERMEDIATE_TERMS];
  
   // Phenomenological merger-ringdown amplitude coefficients:
   float merger_ringdown[NUM_AMPLITUDE_MERGER_RINGDOWN_COEFFICIENTS];
} amplitude_coefficients_s;

/**
 * used to cache the recurring (frequency-independent) prefactors of AmpInsAnsatz. Must be inited with a call to
 * init_amp_ins_prefactors(&prefactors, p);
 */
typedef struct {
     float two_thirds;
     float one;
     float four_thirds;
     float five_thirds;
     float two;
     float seven_thirds;
     float eight_thirds;
     float three;
     float amp0;
} amplitude_inspiral_prefactors_s;

// ~~~~~~~~~~ Phase structures ~~~~~~~~~~:

typedef struct {
   float eta;                // symmetric mass-ratio
   float etaInv;             // 1/eta
   float eta2;               // eta*eta
   float Seta;               // sqrt(1.0 - 4.0*eta);
   float chi1, chi2;         // dimensionless aligned spins, convention m1 >= m2.
   float q;                  // asymmetric mass-ratio (q>=1)
   float chi;                // PN reduced spin parameter
   float ringdown_frequency; // ringdown frequency
   float damping_time;       // imaginary part of the ringdown frequency (damping time)
  
   // Phenomenological inspiral phase coefficients
   float inspiral[NUM_PHASE_INSPIRAL_COEFFICIENTS];
    
   // Phenomenological intermediate phase coefficients
   float intermediate[NUM_PHASE_INTERMEDIATE_COEFFICIENTS];
  
   // Phenomenological merger-ringdown phase coefficients
   float merger_ringdown[NUM_PHASE_MERGER_RINGDOWN_COEFFICIENTS];
  
   // C1 phase connection coefficients
   float C1Int;
   float C2Int;
   float C1MRD;
   float C2MRD;
  
   // Transition frequencies:
   float intermediate_start_frequency;  
   float merger_ringdown_start_frequency;
} phase_coefficients_s;

typedef struct {
    float initial_phasing;
    float third;
    float third_with_logv;
    float two_thirds;
    float one;
    float four_thirds;
    float five_thirds;
    float two;
    float logv;
    float minus_third;
    float minus_two_thirds;
    float minus_one;
    float minus_four_thirds;
    float minus_five_thirds;
} phase_inspiral_prefactors_s;

// Structure for passing around PN phasing coefficients.
// For use with the TaylorF2 waveform:
#define PN_PHASING_SERIES_MAX_ORDER 15
typedef struct {
    float v      [PN_PHASING_SERIES_MAX_ORDER+1];
    float vlogv  [PN_PHASING_SERIES_MAX_ORDER+1];
} pn_phasing_series_s;

typedef struct{
    
    int32_t offset;
    
    amplitude_coefficients_s        amplitude_coefficients;
    float                           amplitude_prefactor_0;
    amplitude_inspiral_prefactors_s amplitude_prefactors;
    
    phase_coefficients_s         phase_coefficients; 
    phase_inspiral_prefactors_s  phase_prefactors;
    pn_phasing_series_s          phasing_series;
    
    useful_powers_s powers_of_reference_mass; 
    
    float precalculated_phase;
    float phase_shift;
} phenom_d_variables_s;

typedef union{
    phenom_d_variables_s d; 
} aproximant_variables_s;

typedef struct
{
    timeUnit_t             *merger_time_for_waveform;
    frequency_array_s       frequency;
    time_array_s            time;
    complex_strain_array_s  strain;
    temporal_properties_s  *temporal_properties_of;
    system_properties_s    *system_properties_of;
    aproximant_variables_s *aproximant_variables_of;
    int32_t                 num_waveforms;
} complex_waveform_axes_s;

typedef struct
{
    timeUnit_t             *merger_time_for_waveform;
    time_array_s            time;
    strain_array_s          strain;
    temporal_properties_s  *temporal_properties_of;
    system_properties_s    *system_properties_of;
    aproximant_variables_s *aproximant_variables_of;
    int32_t                 num_waveforms;
} waveform_axes_s;

#endif



