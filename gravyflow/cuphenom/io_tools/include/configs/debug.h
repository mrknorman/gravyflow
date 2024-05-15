#ifndef DEBUG_STRUCT
#define DEBUG_STRUCT

typedef struct DebugConfig {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold debug mode configuration. NOTE: Organise in order of type size. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	int32_t num_plot;
	int32_t num_write;
	int32_t num_print;
	int32_t print_depth;

	bool    active;

	bool    plot;
	bool    plot_samples;
	bool    plot_locations;
	bool    plot_psd;
	bool    plot_snr;
	bool    plot_interpolation;
	bool    plot_predictions;
	bool    plot_metrics;

	bool    write;
	bool    write_samples;
	bool    write_locations;
	bool    write_psd;
	bool    write_snr;
	bool    write_interpolation;
	bool    write_predictions;

	bool    print;
	bool    print_memory;
	
} debug_config_s;

#endif