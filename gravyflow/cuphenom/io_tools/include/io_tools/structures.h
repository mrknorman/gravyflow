#ifndef IO_STRUCTURES_H
#define IO_STRUCTURES_H

#include "io_tools/custom_types.h"

typedef enum Necessity{ 
	
	/**
     * Enum to hold neccesity information.
     */
	
	required_e, 
	optional_e, 
	excluded_e
} necessity_e;

typedef struct Parameter{
	
	/**
     * Structure to hold infomation about variable to be loaded from config.
     */
	
	char*         name;
	type_e        type;
	int32_t       min;
	int32_t       max;
	float         lower_limit;
	float         upper_limit;
	
} parameter_s;

typedef struct LoaderConfig{
	
	/**
     * Structure to hold config variable grouping.
     */
	
	char                *name;
	necessity_e          name_necessity;
	
	bool                 inherit;
	bool                 is_superconfig;    
	bool                 has_parameters;
    bool                 reorder;
	int32_t              min;
    int32_t              max;
	int32_t              early_exit_index;
	
	int32_t              num_defined_parameters;
	int32_t              min_inputed_parameters;
	int32_t              max_inputed_parameters;
	parameter_s         *defined_parameters;
	
	int32_t              min_extra_parameters;
	int32_t              max_extra_parameters;
	parameter_s          default_parameter;
	
    int32_t              num_defined_subconfigs;
	int32_t              min_num_subconfigs;
	int32_t              max_num_subconfigs;
	struct LoaderConfig *defined_subconfigs;
    
    int32_t              min_extra_subconfigs;
	int32_t              max_extra_subconfigs;
	struct LoaderConfig *default_subconfig;
	
	size_t               struct_size;

} loader_config_s;

typedef struct LoaderData {
	
	/**
     * Union to store commonly used types.
    */
    
    void              *structure;
    char              *name;
    
    map_s              parameter_name_map;
    int32_t           *num_parameters_read;
    
    dict_s            *extra_parameters;
    dict_s            *num_extra_parameters;
    
    map_s              subconfig_name_map;
    int32_t           *num_subconfigs_read;
    int32_t            total_num_subconfigs_read;
	
	dict_s            *num_extra_configs;
    
    struct LoaderData *subconfigs;
    loader_config_s    config;
    
} loader_data_s;

typedef struct LoaderSyntax{
    const char *comment;
    const char *new_line;
    const char *value_indicator;
    const char *start_config;
    const char *end_config;
    const char *string_separator;
    const char *char_separator;
    const char *start_name;
    const char *end_name;
    const char *end_section;
	const char *start_array;
	const char *end_array;
} loader_syntax_s;

// Paser settings:
loader_syntax_s syntax = 
{ 
    .comment          = "#",
    .new_line         = ";",
    .value_indicator  = "=",
    .start_config     = "{",
    .end_config       = "}",
    .string_separator = "\"",
    .char_separator   = "\'",
    .start_name       = "[",
    .end_name         = "]",
	.start_array      = "(",
	.end_array        = ")"
};

#endif
