#ifndef DATASET_CONFIG_STRUCTS
#define DATASET_CONFIG_STRUCTS

#include "debug.h"	


typedef struct Detector{

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold detector configuration. NOTE: Organise in order of type size. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	char    *name;
	
	float   *position, *position_sphere;
	int32_t *latitude, *longitude;

	float   *noise_profile_strain, *noise_profile_frequency, *interpolated_noise_profile; 
	float   *x_arm_direction, *y_arm_direction, *up_direction;
	
	float    height_above_sea_level;

    float    sensitivity, noise_amplitude; 

	int32_t  noise_profile_length;
	int32_t  interpolated_noise_profile_length;

	int32_t  x_arm_bearing, y_arm_bearing;
	
	bool     active;
	
} detector_s;

typedef struct NetworkConfig{

	float   speed_of_light;
	float   equatorial_radius;
	float   polar_radius;

	int32_t num_detectors;
	int32_t num_dimensions;

} network_config_s;

typedef struct WaveformConfig {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold waveform configuration. NOTE: Organise in order of type size. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	char   *name;

	float   sample_rate;
	float   min_duration;
	float   max_duration;
	
	float   subwaveforms_per_stream_mu;
	float   subwaveforms_per_stream_sigma;
	float   subwaveforms_per_stream_min;
	float   subwaveforms_per_stream_max;
	
	int32_t num_subwaveforms;
	int32_t num_subwaveform_layers;
	int32_t min_num_repeats;
	int32_t max_num_repeats;
	
} waveform_config_s;

typedef struct SubwaveformConfig {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold subwaveformnal configuration. NOTE: Organise in order of type size. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	char   *name;
	char   *path;
    
	float   mass_1_min; 
	float   mass_1_max;
	float   mass_2_min;
	float   mass_2_max;
	
	float   iota_min;
	float   iota_max;
	float   distance_min;
	float   distance_max;
	float   sigma_min;
	float   sigma_max;
	float   power_min;
	float   power_max;
	float   a_max;
	float   a_min;
	float   phaze_min;
	float   phaze_max;
	float   frequency_min;
	float   frequency_max;
	float   max_mass_ratio;
	
	int32_t distribution_type;
	int32_t k_max;
	int32_t k_min;
	int32_t type;
	
	bool    active; 

} subwaveform_config_s;

typedef struct NoiseConfig {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold noise configuration. NOTE: Organise in order of type size. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	char    *name;
	float    amplitude_sigma;
	float    amplitude_mu;
	float    amplitude_min;
	float    amplitude_max;

	int32_t  type;

	bool     active;

} noise_config_s;

typedef struct NoiseGroup {

	int32_t config_index;
	int32_t source_type;

} noise_group_s;

typedef struct SnrGroup {
	
	char  *name; 
	float  snr_mu;
	float  snr_sigma;
	float  snr_min;
	float  snr_max;

} snr_group_s;

typedef struct CoherenceGroup {
	
	char    *name; 
	int32_t  config_index;
	int32_t  source_type;

} coherence_group_s;

typedef struct SubinjectionConfig {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold subinjection configuration. NOTE: Organise in order of type size.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	char     *name;
	
	int32_t   waveform_index;
	int32_t   coherence_group;
	int32_t   snr_group;
	
	int32_t   source_type;

	bool      active;

	bool      add_time_difference;
	bool      simulate_detector_response;
	bool      simulate_polarisation_response;
	bool      simulate_orientation_response;

	bool      snr_adjust;
	bool      used_in_snr;
	bool      use_network_snr;
	
} subinjection_config_s;

typedef struct InjectionConfig {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold runtime variables.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	char      *name;
	char      *debug_directory_path;
	
	float     *snr_mu;
	float     *snr_sigma;
	float     *snr_min;
	float     *snr_max;
	
	float      injections_per_stream_mu;
	float      injections_per_stream_sigma;
	float      injections_per_stream_min;
	float      injections_per_stream_max;

	float      subinjections_per_stream_mu;
	float      subinjections_per_stream_sigma;
	float      subinjections_per_stream_min;
	float      subinjections_per_stream_max;

	float      front_padding, back_padding;
	float      stream_sample_rate;
	float      center_time_min, center_time_max;
	
	int32_t    stream_length;
	int32_t    num_streams;
	int32_t    num_detectors;
	int32_t    num_injections;

	int32_t    num_injection_types;
	int32_t    num_coherence_groups;
	int32_t    num_unique_coherence_groups;
	int32_t    num_snr_groups;
	int32_t    num_unique_snr_groups;

	int32_t    num_segments;
	int32_t    segment_length;
	
    int32_t    snr_distribution;
	
} injection_config_s;

typedef struct ExampleConfig_s {
	
	char   *name;
    char   *label;
	
	int32_t num_injection_types;
	int32_t num_streams;
	int32_t noise_type;

	bool    active;
	bool    add_injections;
	bool    add_noise;

} example_config_s;

#else
// Default Parameters:
parameter_s default_parameter = 
	{"default_parameter", none_e, 0, 0, 0.0f, 0.0f};

// Detectors:
const int32_t num_detector_parameters = 19;
parameter_s detector_parameters[] = 
{
	{"name"                             , string_e     , 0, 0, -FLT_MAX, FLT_MAX},
	{"position"                         , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"position_sphere"                  , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"latitude"                         , int_array_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"longitude"                        , int_array_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"noise_profile_strain"             , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
    {"noise_profile_frequency"          , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"interpolated_noise_profile"       , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"x_arm_direction"                  , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"y_arm_direction"                  , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"up_direction"                     , float_array_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"height_above_sea_level"           , float_e      , 1, 1, -FLT_MAX, FLT_MAX},
	{"sensitivity"                      , float_e      , 1, 1,     0.0f, FLT_MAX},
	{"noise_amplitude"                  , float_e      , 1, 1,     0.0f, FLT_MAX},
	{"noise_profile_length"             , int_e        , 0, 0, -FLT_MAX, FLT_MAX},
	{"interpolated_noise_profile_length", int_e        , 0, 0, -FLT_MAX, FLT_MAX},
	{"x_arm_bearing"                    , int_e        , 1, 1,     0.0f,  360.0f},
	{"y_arm_bearing"                    , int_e        , 1, 1,     0.0f,  360.0f},
	{"active"                           , bool_e       , 1, 1, -FLT_MAX, FLT_MAX}
};

const int32_t min_num_detector_parameters = 8;
const int32_t max_num_detector_parameters = 8;

loader_config_s detector_subconfig = 
{
	.name                   = "default_detector_name",
	.name_necessity         = optional_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = true,
    .min                    = 1,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_detector_parameters,
	.min_inputed_parameters = min_num_detector_parameters,
	.max_inputed_parameters = max_num_detector_parameters,
	.defined_parameters     = detector_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(detector_s)
};

loader_config_s default_detector[] = {detector_subconfig};

loader_config_s detector_loader_config = 
{
	.name                   = "detectors",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 1,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
   
	.num_defined_parameters = num_detector_parameters,
	.min_inputed_parameters = 0,
	.max_inputed_parameters = 0,
	.defined_parameters     = detector_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 3,
	.max_num_subconfigs     = 3,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_detector,

	.struct_size            = sizeof(detector_s)
};

loader_config_s default_detectors[] = {detector_loader_config};

// Network:
const int32_t num_network_parameters = 4;
parameter_s network_parameters[] = 
{
	{"speed_of_light"   , float_e, 1, 1, 0.0f, FLT_MAX},
	{"equatorial_radius", float_e, 1, 1, 0.0f, FLT_MAX},
	{"polar_radius"     , float_e, 1, 1, 0.0f, FLT_MAX},
	{"num_dimensions"   , int_e  , 1, 1, 0.0f, FLT_MAX}	
};

loader_config_s network_loader_config = 
{
	.name                   = "network",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 1,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_network_parameters,
	.min_inputed_parameters = num_network_parameters,
	.max_inputed_parameters = num_network_parameters,
	.defined_parameters     = network_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 1,
	.min_num_subconfigs     = 1,
	.max_num_subconfigs     = 1,
	.defined_subconfigs     = default_detectors,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = 0,
	.default_subconfig      = default_detectors,

	.struct_size            = sizeof(network_config_s)
};

// Subwaveform: 

// ~~~~ Gausian Window:
const int32_t num_gaussian_parameters = 4;
parameter_s gaussian_parameters[] = 
{
	{"sigma_min", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"sigma_max", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"power_min", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"power_max", float_e, 1, 1, -FLT_MAX, FLT_MAX}
};

// ~~~~ Triangular Window:
const int32_t num_triangular_parameters = 0;
parameter_s *triangular_parameters = NULL;

// ~~~~ Welch Window:
const int32_t num_welch_parameters = 0;
parameter_s *welch_parameters = NULL;

// ~~~~ Rectangular Window:
const int32_t num_rectangular_parameters = 0;
parameter_s *rectangular_parameters = NULL;

// ~~~~ Sine Window:
const int32_t num_sine_parameters = 0;
parameter_s *sine_parameters = NULL;

// ~~~~ Power Sine Window:
const int32_t num_power_sine_parameters = 2;
parameter_s power_sine_parameters[] = 
{
	{"power_min", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"power_max", float_e, 1, 1, -FLT_MAX, FLT_MAX}
};

// ~~~~ Cosine sum Window:
const int32_t num_cosine_sum_parameters = 4;
parameter_s cosine_sum_parameters[] = 
{
	{"a_min", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"a_max", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"k_min", int_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"k_max", int_e  , 1, 1, -FLT_MAX, FLT_MAX}
};

// ~~~~ Tukey Window:
const int32_t num_tukey_parameters = 2;
parameter_s tukey_parameters[] = 
{
	{"a_min", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"a_max", float_e, 1, 1, -FLT_MAX, FLT_MAX}
};

// ~~~~ Possion Window:
const int32_t num_possion_parameters = 2;
parameter_s possion_parameters[] = 
{
	{"a_min", float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"a_max", float_e, 1, 1, -FLT_MAX, FLT_MAX}
};

// ~~~~ Sine Wave:
const int32_t num_sine_wave_parameters = 2;
parameter_s sine_wave_parameters[] = 
{
	{"frequency_min", float_e, 1, 1, 0.0f, FLT_MAX},
	{"frequency_max", float_e, 1, 1, 0.0f, FLT_MAX}
};

// ~~~~ Triangular Wave:
const int32_t num_triangular_wave_parameters = 6;
parameter_s triangular_wave_parameters[] = 
{  
	{"frequency_min", float_e, 1, 1,     0.0f, FLT_MAX},
	{"frequency_max", float_e, 1, 1,     0.0f, FLT_MAX},
	{"phase_min"    , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"phase_max"    , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"k_min"        , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"k_max"        , float_e, 1, 1, -FLT_MAX, FLT_MAX}
};

// ~~~~ Saw Wave:
const int32_t num_saw_wave_parameters = 6;
parameter_s saw_wave_parameters[] = 
{
	{"frequency_min", float_e, 1, 1,     0.0f, FLT_MAX},
	{"frequency_max", float_e, 1, 1,     0.0f, FLT_MAX},
	{"phase_min"    , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"phase_max"    , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"k_min"        , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"k_max"        , float_e, 1, 1, -FLT_MAX, FLT_MAX}
};

// ~~~~ Inspiral:
const int32_t num_inspiral_parameters = 10;
parameter_s inspiral_parameters[] = 
{
	{"distribution_type", int_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"max_mass_ratio"   , int_e  , 1, 1,     0.0f, FLT_MAX},
	{"mass_1_min"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"mass_1_max"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"mass_2_min"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"mass_2_max"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"iota_min"         , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"iota_max"         , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"distance_min"     , float_e, 1, 1,     0.0f, FLT_MAX},
	{"distance_max"     , float_e, 1, 1,     0.0f, FLT_MAX}
};

// ~~~~ IMRPhenomD:
const int32_t num_imrphenomd_parameters = 10;
parameter_s imrphenomd_parameters[] = 
{
	{"distribution_type", int_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"max_mass_ratio"   , int_e  , 1, 1,     0.0f, FLT_MAX},
	{"mass_1_min"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"mass_1_max"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"mass_2_min"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"mass_2_max"       , float_e, 1, 1,     0.0f, FLT_MAX},
	{"iota_min"         , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"iota_max"         , float_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"distance_min"     , float_e, 1, 1,     0.0f, FLT_MAX},
	{"distance_max"     , float_e, 1, 1,     0.0f, FLT_MAX}
};

// ~~~~ Load Waveform:
const int32_t num_load_parameters = 1;
parameter_s load_parameters[] = 
{
	{"path", int_e, 1, 1, -FLT_MAX, FLT_MAX},
};

// Subwaveform Priority Parameters:

char *subwaveform_names[] =
{
	"gaussian_window",
	"triangular_window",
	"welch_window",
	"rectangular_window",
	"sine_window",
	"power_sine_window",
	"cosine_sum_window",
	"tukey_window",
	"possion_window",
	"sine_wave",
	"triangular_wave",
	"saw_wave",
	"inspiral",
	"imrphenomd",
	"load"
};

int32_t subwaveform_num_priority_parameters[] = 
{
	num_gaussian_parameters,
	num_triangular_parameters,
	num_welch_parameters,
	num_rectangular_parameters,
	num_sine_parameters,
	num_power_sine_parameters,
	num_cosine_sum_parameters,
	num_tukey_parameters,
	num_possion_parameters,
	num_sine_wave_parameters,
	num_triangular_wave_parameters,
	num_saw_wave_parameters,
	num_inspiral_parameters,
	num_imrphenomd_parameters,
	num_load_parameters
};

parameter_s *subwaveform_priority_parameters[] =
{
	gaussian_parameters,
	triangular_parameters,
	welch_parameters,
	rectangular_parameters,
	sine_parameters,
	power_sine_parameters,
	cosine_sum_parameters,
	tukey_parameters,
	possion_parameters,
	sine_wave_parameters,
	triangular_wave_parameters,
	saw_wave_parameters,
	inspiral_parameters,
	imrphenomd_parameters,
	load_parameters
}; 

loader_config_s subwaveform_subconfig = 
{
	.name                   = "default_subwaveform_name",
	.name_necessity         = required_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = true,
    .min                    = 0,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = 0,
	.min_inputed_parameters = 2,
	.max_inputed_parameters = 25,
	.defined_parameters     = NULL,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(subwaveform_config_s)
};

loader_config_s default_subwaveform_subconfig[] =
	{subwaveform_subconfig};

const int32_t num_subwaveform_types = 15;
loader_config_s subwaveform_subconfigs[15];

for (int32_t index = 0; index < num_subwaveform_types; index++)
{
	subwaveform_subconfigs[index]                        = 
		subwaveform_subconfig;
	subwaveform_subconfigs[index].name                   = 
		subwaveform_names[index];
	subwaveform_subconfigs[index].num_defined_parameters =
		subwaveform_num_priority_parameters[index];
	subwaveform_subconfigs[index].defined_parameters =
		subwaveform_priority_parameters[index];
}

const int32_t num_subwaveform_parameters = 26;
parameter_s subwaveform_parameters[] = 
{
	{"name"             , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"path"             , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"mass_1_min"       , float_e , 0, 0,     0.0f, FLT_MAX},
	{"mass_1_max"       , float_e , 0, 0,     0.0f, FLT_MAX},
	{"mass_2_min"       , float_e , 0, 0,     0.0f, FLT_MAX},
	{"mass_2_max"       , float_e , 0, 0,     0.0f, FLT_MAX},
	{"iota_min"         , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"iota_max"         , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"distance_min"     , float_e , 0, 0,     0.0f, FLT_MAX},
	{"distance_max"     , float_e , 0, 0,     0.0f, FLT_MAX},
	{"sigma_min"        , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"sigma_max"        , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"power_min"        , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"power_max"        , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"a_min"            , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"a_max"            , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"phaze_min"        , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"phaze_max"        , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"frequency_min"    , float_e , 0, 0,     0.0f, FLT_MAX},
	{"frequency_max"    , float_e , 0, 0,     0.0f, FLT_MAX},
	{"max_mass_ratio"   , float_e , 0, 0,     0.0f, FLT_MAX},
	{"distribution_type", int_e   , 0, 0, -FLT_MAX, FLT_MAX},
	{"k_max"            , int_e   , 0, 0, -FLT_MAX, FLT_MAX},
	{"k_min"            , int_e   , 0, 0, -FLT_MAX, FLT_MAX},
	{"layer"            , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"active"           , bool_e  , 1, 1, -FLT_MAX, FLT_MAX}
};

loader_config_s subwaveform_loader_config = 
{
	.name                   = "subwaveforms",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 1,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_subwaveform_parameters,
	.min_inputed_parameters = 2,
	.max_inputed_parameters = 25,
	.defined_parameters     = subwaveform_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = num_subwaveform_types,
	.min_num_subconfigs     = 0,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = subwaveform_subconfigs,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = 0,
	.default_subconfig      = default_subwaveform_subconfig,

	.struct_size            = sizeof(subwaveform_config_s)
};

loader_config_s default_subwaveform[] = {subwaveform_loader_config};

// Waveform:
const int32_t num_waveform_parameters = 12;
parameter_s waveform_parameters[] = 
{
	{"name"                         , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"sample_rate"                  , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"min_duration"                 , float_e , 1, 1,     0.0f, FLT_MAX},
	{"max_duration"                 , float_e , 1, 1,     0.0f, FLT_MAX},
	{"subwaveforms_per_stream_mu"   , float_e , 1, 1,     0.0f, FLT_MAX},
	{"subwaveforms_per_stream_sigma", float_e , 1, 1,     0.0f, FLT_MAX},
	{"subwaveforms_per_stream_min"  , float_e , 1, 1,     0.0f, FLT_MAX},
	{"subwaveforms_per_stream_max"  , float_e , 1, 1,     0.0f, FLT_MAX},
	{"num_subwaveforms"             , int_e   , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_subwaveform_layers"       , int_e   , 1, 1,     0.0f, FLT_MAX},
	{"min_num_repeats"              , int_e   , 1, 1,     0.0f, FLT_MAX},
	{"max_num_repeats"              , int_e   , 1, 1,     0.0f, FLT_MAX}	
};

loader_config_s waveform_subconfig = 
{
	.name                   = "default_waveform_name",
	.name_necessity         = optional_e,

	.is_superconfig         = true,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 0,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_waveform_parameters,
	.min_inputed_parameters = 9,
	.max_inputed_parameters = 9,
	.defined_parameters     = waveform_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 1,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_subwaveform,

	.struct_size            = sizeof(waveform_config_s)
};

loader_config_s default_waveform[] = {waveform_subconfig};

loader_config_s waveform_loader_config = 
{
	.name                   = "waveforms",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 1,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_waveform_parameters,
	.min_inputed_parameters = 10,
	.max_inputed_parameters = 10,
	.defined_parameters     = waveform_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 0,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_waveform,

	.struct_size            = sizeof(waveform_config_s)
};

// Noise:
const int32_t num_noise_parameters = 7;
parameter_s noise_parameters[] = 
{
	{"name"           , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"amplitude_sigma", float_e , 1, 1,     0.0f, FLT_MAX},
	{"amplitude_mu"   , float_e , 1, 1,     0.0f, FLT_MAX},
	{"amplitude_min"  , float_e , 1, 1,     0.0f, FLT_MAX},
	{"amplitude_max"  , float_e , 1, 1,     0.0f, FLT_MAX},
	{"type"           , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"active"         , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
};

loader_config_s noise_subconfig = 
{
	.name                   = "default_noise_name",
	.name_necessity         = optional_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 0,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
	
	.num_defined_parameters = num_noise_parameters,
	.min_inputed_parameters = 6,
	.max_inputed_parameters = 6,
	.defined_parameters     = noise_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(noise_config_s)
};

loader_config_s default_noise_subconfig[] =
{
	 noise_subconfig
};

loader_config_s noise_loader_config = 
{
	.name                   = "noise",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_noise_parameters,
	.min_inputed_parameters = 6,
	.max_inputed_parameters = 6,
	.defined_parameters     = noise_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 1,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_noise_subconfig,

	.struct_size            = sizeof(noise_config_s)
};

// Subinjection:
const int32_t num_subinjection_parameters = 13;
parameter_s subinjection_parameters[] = 
{
	{"name"                          , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"waveform_index"                , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"coherence_group"               , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"snr_group"                     , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"source_type"                   , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"active"                        , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"add_time_difference"           , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"simulate_detector_response"    , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"simulate_polarisation_response", bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"simulate_orientation_response" , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"snr_adjust"                    , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"used_in_snr"                   , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"use_network_snr"               , bool_e  , 1, 1, -FLT_MAX, FLT_MAX}
};

loader_config_s subinjection_subconfig = 
{
	.name                   = "default_subinjection_name",
	.name_necessity         = optional_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 0,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_subinjection_parameters,
	.min_inputed_parameters = 12,
	.max_inputed_parameters = 12,
	.defined_parameters     = subinjection_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(subinjection_config_s)
};

loader_config_s default_subinjection_subconfig[] =
{
	 subinjection_subconfig
};

loader_config_s subinjection_loader_config = 
{
	.name                   = "subinjections",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
	
	.num_defined_parameters = num_subinjection_parameters,
	.min_inputed_parameters = 12,
	.max_inputed_parameters = 12,
	.defined_parameters     = subinjection_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 1,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_subinjection_subconfig,

	.struct_size            = sizeof(subinjection_config_s)
};

loader_config_s default_subinjection_loader_config[] =
{
	 subinjection_loader_config
};

// SNR Groups:
const int32_t num_snr_group_parameters = 5;
parameter_s snr_group_parameters[] =
{
	{"name"     , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"snr_mu"   , float_e , 1, 1,     0.0f, FLT_MAX},
	{"snr_sigma", float_e , 1, 1,     0.0f, FLT_MAX},
	{"snr_min"  , float_e , 1, 1,     0.0f, FLT_MAX},
	{"snr_max"  , float_e , 1, 1,     0.0f, FLT_MAX},
};

loader_config_s snr_group_subconfig = 
{
	.name                   = "default_snr_group_name",
	.name_necessity         = optional_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 0,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_snr_group_parameters,
	.min_inputed_parameters = 4,
	.max_inputed_parameters = 4,
	.defined_parameters     = snr_group_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(snr_group_s)
};

loader_config_s default_snr_group_subconfig[] =
{
	 snr_group_subconfig
};

loader_config_s snr_group_loader_config = 
{
	.name                   = "snr_groups",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_snr_group_parameters,
	.min_inputed_parameters = 12,
	.max_inputed_parameters = 12,
	.defined_parameters     = snr_group_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 1,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_snr_group_subconfig,

	.struct_size            = sizeof(snr_group_s)
};

// Coherence Groups:
const int32_t num_coherence_group_parameters = 3;
parameter_s coherence_group_parameters[] =
{
	{"name"        , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"source_type" , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"config_index", int_e   , 1, 1, -FLT_MAX, FLT_MAX}
};

loader_config_s coherence_group_subconfig = 
{
	.name                   = "default_coherence_group_name",
	.name_necessity         = optional_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 0,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_coherence_group_parameters,
	.min_inputed_parameters = 2,
	.max_inputed_parameters = 2,
	.defined_parameters     = coherence_group_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(coherence_group_s)
};

loader_config_s default_coherence_group_subconfig[] =
{
	 coherence_group_subconfig
};

loader_config_s coherence_group_loader_config = 
{
	.name                   = "coherence_groups",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_coherence_group_parameters,
	.min_inputed_parameters = 12,
	.max_inputed_parameters = 12,
	.defined_parameters     = coherence_group_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 1,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_coherence_group_subconfig,

	.struct_size            = sizeof(coherence_group_s)
};

// Injections:
const int32_t num_injection_parameters = 31;
parameter_s injection_parameters[] = 
{
	{"name"                          , string_e      , 0, 0, -FLT_MAX, FLT_MAX},
	{"debug_directory_path"          , string_e      , 0, 0, -FLT_MAX, FLT_MAX},
	{"snr_mu"                        , float_array_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"snr_sigma"                     , float_array_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"snr_min"                       , float_array_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"snr_max"                       , float_array_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"injections_per_stream_mu"      , float_e       , 1, 1,     0.0f, FLT_MAX},
	{"injections_per_stream_sigma"   , float_e       , 1, 1,     0.0f, FLT_MAX},
	{"injections_per_stream_min"     , float_e       , 1, 1,     0.0f, FLT_MAX},
	{"injections_per_stream_max"     , float_e       , 1, 1,     0.0f, FLT_MAX},
	{"subinjections_per_stream_mu"   , float_e       , 1, 1,     0.0f, FLT_MAX},
	{"subinjections_per_stream_sigma", float_e       , 1, 1,     0.0f, FLT_MAX},
	{"subinjections_per_stream_min"  , float_e       , 1, 1,     0.0f, FLT_MAX},
	{"subinjections_per_stream_max"  , float_e       , 1, 1,     0.0f, FLT_MAX},
	{"front_padding"                 , float_e       , 1, 1, -FLT_MAX, FLT_MAX},
	{"back_padding"                  , float_e       , 1, 1, -FLT_MAX, FLT_MAX},
	{"stream_sample_rate"            , float_e       , 0, 0, -FLT_MAX, FLT_MAX},
	{"center_time_min"               , float_e       , 0, 0, -FLT_MAX, FLT_MAX},
	{"center_time_max"               , float_e       , 0, 0, -FLT_MAX, FLT_MAX},
	{"stream_length"                 , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_streams"                   , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_detectors"                 , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_injections"                , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_injection_types"           , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_coherence_groups"          , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_unique_coherence_groups"   , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_snr_groups"                , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_unique_snr_groups"         , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_segments"                  , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"segment_length"                , int_e         , 0, 0, -FLT_MAX, FLT_MAX},
	{"snr_distribution"              , int_e         , 1, 1, -FLT_MAX, FLT_MAX}
};

loader_config_s injection_subconfig = 
{
	.name                   = "default_injection_name",
	.name_necessity         = optional_e,

	.is_superconfig         = true,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 0,
    .max                    = INT32_MAX,
	.early_exit_index       = INT32_MAX,
	
	.num_defined_parameters = num_injection_parameters,
	.min_inputed_parameters = 11,
	.max_inputed_parameters = 11,
	.defined_parameters     = injection_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 1,
	.max_num_subconfigs     = 1,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_subinjection_loader_config,

	.struct_size            = sizeof(injection_config_s)
};

loader_config_s default_injection_subconfig[] =
{
	 injection_subconfig
};

loader_config_s injection_loader_config = 
{
	.name                   = "injections",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
	
	.num_defined_parameters = num_injection_parameters,
	.min_inputed_parameters = 11,
	.max_inputed_parameters = 11,
	.defined_parameters     = injection_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = 0,
	.min_num_subconfigs     = 0,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = NULL,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_injection_subconfig,

	.struct_size            = sizeof(injection_config_s)
};

loader_config_s default_injection_loader_config[] =
{
	 injection_loader_config
};

// Example config

const int32_t num_example_parameters = 8;
parameter_s example_parameters[] =
{
	{"name"               , string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"label"              , string_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"num_injection_types", int_e   , 0, 0, -FLT_MAX, FLT_MAX},
	{"num_streams"        , int_e   , 0, 0, -FLT_MAX, FLT_MAX},
	{"noise_type"         , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"active"             , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"add_injections"     , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
	{"add_noise"          , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},
};

const int32_t num_example_subconfigs = 3;
loader_config_s example_subconfigs[] = 
{
	snr_group_loader_config,
	coherence_group_loader_config,
	injection_loader_config
};

loader_config_s example_subconfig = 
{
	.name                   = "default_example_name",
	.name_necessity         = optional_e,

	.is_superconfig         = true,
	.has_parameters         = true,
	.inherit                = false,
    .min                    = 1,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_example_parameters,
	.min_inputed_parameters = 5,
	.max_inputed_parameters = 5,
	.defined_parameters     = example_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = num_example_subconfigs,
	.min_num_subconfigs     = 0,
	.max_num_subconfigs     = 3,
	.defined_subconfigs     = example_subconfigs,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = 0,
	.default_subconfig      = default_injection_loader_config,

	.struct_size            = sizeof(example_config_s)
};

loader_config_s default_example_loader_config[] = 
{
	example_subconfig
};

loader_config_s example_loader_config = 
{
	.name                   = "examples",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
	
	.num_defined_subconfigs = num_example_subconfigs,
	.min_num_subconfigs     = 0,
	.max_num_subconfigs     = INT32_MAX,
	.defined_subconfigs     = example_subconfigs,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = INT32_MAX,
	.default_subconfig      = default_example_loader_config,

	.struct_size            = sizeof(example_config_s)
};

// Debug
const int32_t num_debug_parameters = 22;
parameter_s debug_parameters[] = 
{
	{"num_plot"           , int_e , 0, 0,     0.0f, FLT_MAX},
	{"num_write"          , int_e , 0, 0,     0.0f, FLT_MAX},
	{"num_print"          , int_e , 0, 0,     0.0f, FLT_MAX},
	{"print_depth"        , int_e , 0, 0,     0.0f, FLT_MAX},
	{"active"             , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot"               , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"plot_samples"       , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"plot_locations"     , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"plot_psd"           , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"plot_snr"           , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"plot_interpolation" , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"plot_predictions"   , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"plot_metrics"       , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"write"              , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"write_samples"      , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"write_locations"    , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"write_psd"          , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"write_snr"          , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"write_interpolation", bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"write_predictions"  , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"print"              , bool_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"print_memory"       , bool_e, 0, 0, -FLT_MAX, FLT_MAX}
};

loader_config_s debug_subconfig = 
{
	.name                   = "default_debug_name",
	.name_necessity         = required_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = true,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_debug_parameters,
	.min_inputed_parameters = 1,
	.max_inputed_parameters = num_debug_parameters,
	.defined_parameters     = debug_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(debug_config_s)
};

loader_config_s default_debug_subconfig[] =
{
	 debug_subconfig
};

const int32_t debug_detector_num_parameters = 7;
parameter_s debug_detector_parameters[] =
{
	{"print_depth"    , int_e  , 1, 1,     0.0f, FLT_MAX},
	{"plot"           , bool_e , 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_locations" , bool_e , 1, 1, -FLT_MAX, FLT_MAX},
	{"write"          , bool_e , 1, 1, -FLT_MAX, FLT_MAX},
	{"write_locations", bool_e , 1, 1, -FLT_MAX, FLT_MAX},
	{"print"          , bool_e , 1, 1, -FLT_MAX, FLT_MAX},
	{"print_memory"   , bool_e , 1, 1, -FLT_MAX, FLT_MAX}
};

const int32_t debug_noise_num_parameters = 14;
parameter_s debug_noise_parameters[] =
{
	{"num_plot"           , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_print"          , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_write"          , int_e , 1, 1,     0.0f, FLT_MAX},
	{"print_depth"        , int_e , 1, 1,     0.0f, FLT_MAX},
	{"plot"               , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_psd"           , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_interpolation" , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_samples"       , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write"              , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_psd"          , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_interpolation", bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_samples"      , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print"              , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print_memory"       , bool_e, 1, 1, -FLT_MAX, FLT_MAX}
};

const int32_t debug_waveform_num_parameters = 8;
parameter_s debug_waveform_parameters[] =
{
	{"num_plot"    , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_write"   , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_print"   , int_e , 1, 1,     0.0f, FLT_MAX},
	{"print_depth" , int_e , 1, 1,     0.0f, FLT_MAX},
	{"plot"        , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write"       , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print"       , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print_memory", bool_e, 1, 1, -FLT_MAX, FLT_MAX}
};

const int32_t debug_localisation_num_parameters = 10;
parameter_s debug_localisation_parameters[] =
{
	{"num_plot"       , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_write"      , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_print"      , int_e , 1, 1,     0.0f, FLT_MAX},
	{"print_depth"    , int_e , 1, 1,     0.0f, FLT_MAX},
	{"plot"           , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_locations" , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write"          , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_locations", bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print"          , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print_memory"   , bool_e, 1, 1, -FLT_MAX, FLT_MAX}
};

const int32_t debug_snr_num_parameters = 14;
parameter_s debug_snr_parameters[] =
{
	{"num_plot"           , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_write"          , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_print"          , int_e , 1, 1,     0.0f, FLT_MAX},
	{"print_depth"        , int_e , 1, 1,     0.0f, FLT_MAX},
	{"plot"               , bool_e, 1, 1, -FLT_MAX, FLT_MAX},	
	{"plot_psd"           , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_interpolation" , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_snr"           , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write"              , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_psd"          , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_interpolation", bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_snr"          , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print"              , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print_memory"       , bool_e, 1, 1, -FLT_MAX, FLT_MAX}
};  

const int32_t debug_physicalisation_num_parameters = 10;
parameter_s debug_physicalisation_parameters[] =
{
	{"num_plot"     , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_write"    , int_e , 1, 1,     0.0f, FLT_MAX},
	{"num_print"    , int_e , 1, 1,     0.0f, FLT_MAX},
	{"print_depth"  , int_e , 1, 1,     0.0f, FLT_MAX},
	{"plot"         , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"plot_samples" , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write"        , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"write_samples", bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print"        , bool_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"print_memory" , bool_e, 1, 1, -FLT_MAX, FLT_MAX}
};

char *debug_names[] =
{
	"detector",
	"noise",
	"waveform",
	"localisation",
	"snr",
	"physicalisation"
};

parameter_s *debug_priority_parameters[] =
{
	debug_detector_parameters,
	debug_noise_parameters,
	debug_waveform_parameters,
	debug_localisation_parameters,
	debug_snr_parameters,
	debug_physicalisation_parameters
};

int32_t debug_num_priority_parameters[] =
{
	debug_detector_num_parameters,
	debug_noise_num_parameters,
	debug_waveform_num_parameters,
	debug_localisation_num_parameters,
	debug_snr_num_parameters,
	debug_physicalisation_num_parameters
};

const int32_t num_debug_types = 6;
loader_config_s debug_subconfigs[6];

for (int32_t index = 0; index < num_debug_types; index++)
{
	debug_subconfigs[index]                        = debug_subconfig;
	debug_subconfigs[index].name                   = debug_names[index];
	debug_subconfigs[index].num_defined_parameters = debug_num_priority_parameters[index];
	debug_subconfigs[index].defined_parameters     = debug_priority_parameters[index];
}

loader_config_s debug_loader_config = 
{
	.name                   = "debug",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
	.inherit                = false,
    .min                    = 0,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_debug_parameters,
	.min_inputed_parameters = 1,
	.max_inputed_parameters = num_debug_parameters,
	.defined_parameters     = debug_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
	
	.num_defined_subconfigs = num_debug_types,
	.min_num_subconfigs     = 0,
	.max_num_subconfigs     = num_debug_types,
	.defined_subconfigs     = debug_subconfigs,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = 0,
	.default_subconfig      = default_debug_subconfig,

	.struct_size            = sizeof(debug_config_s)
};

// Dataset:

const int32_t num_defined_subconfigs = 5;
loader_config_s defined_subconfigs[] = 
{
	network_loader_config,
	waveform_loader_config,
	noise_loader_config,
	example_loader_config,
	debug_loader_config
};

loader_config_s default_subconfig[] = {network_loader_config};

loader_config_s loader_config = 
{
	.name                   = "dataset",
	.name_necessity         = excluded_e,

	.is_superconfig         = true,
	.inherit                = false,
	.has_parameters         = false,
    .reorder                = true,
    .min                    = 1,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = 0,
	.min_inputed_parameters = 0,
	.max_inputed_parameters = 0,
	.defined_parameters     = NULL,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
    
    .num_defined_subconfigs = num_defined_subconfigs,
	.min_num_subconfigs     = 4,
	.max_num_subconfigs     = 5,
	.defined_subconfigs     = defined_subconfigs,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = 0,
	.default_subconfig      = default_subconfig,

	.struct_size            = 0
};

#endif