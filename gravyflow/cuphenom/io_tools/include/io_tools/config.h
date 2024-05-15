#ifndef IO_CONFIG_H
#define IO_CONFIG_H

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include <inttypes.h>
#include <string.h>
#include "float.h"

#include "io_tools/text.h"
#include "io_tools/strings.h"
#include "io_tools/custom_types.h"
#include "io_tools/structures.h"

typedef struct construct_s {
	
	size_t       structure_size;
	size_t      *structure_map;
	parameter_s *parameters;
	
} construct_s;

size_t getTotalTypeArraySize(
    const type_e   *types, 
    const int32_t   num_elements
    ) {

	size_t total_size = 0u;

	for (int32_t index = 0; index < num_elements; index++) 
	{	
		total_size += getSizeOfType(types[index]);
	}
	
	return total_size;
} 

bool checkTypeOrder(
	const int32_t  verbosity,
    const type_e  *types, 
    const int32_t  num_elements
   ) {
	
	bool return_value = true;
	
	size_t value      = 0u;
	size_t last_value = 0u;
	
	if (num_elements > 0)
	{
		last_value = getSizeOfType(types[0]);
	}

	for (int32_t index = 1; index < num_elements - 1; index++)
	{
        value = getSizeOfType(types[index]);
		
		if(value > last_value) 
		{
			return_value = false;
			
			if (verbosity > 0) 
			{
				fprintf(stderr, 
					"checkTypeOrder: \nError! Struture elements must be aranged"
					" in decreasing size to ensure no padding is added!"
					" Exiting. Types: %s: %zu > %s: %zu. \n", 
					typeToString(types[index]), 
					getSizeOfType(types[index]), 
					typeToString(types[index - 1]), 
					last_value
				);
			}
		}
        last_value = value;
	}
	
	return return_value; 
}

bool checkStructSize( 
	const int32_t  verbosity,
	const size_t   size,
	const size_t   compiled_size,
	const size_t   largest_memory_alignment,
	const char    *name
	) {
	
	bool return_value = true;
	
	if (size > compiled_size) 
	{		
		if (verbosity > 0) 
		{
			fprintf(
				stderr, 
				"checkStructSize: \nError! Inputted struct size: (%zu) is"
				"larger than compiled structure: (%zu), exiting. For config"
				"%s.\n", 
				size, 
				compiled_size,
				name
			);
		}
		
		return_value = false;
	} 
	else if (
			(compiled_size >= 8) 
			&& 
			(size < compiled_size - largest_memory_alignment)
		) 
	{
		
		if (verbosity > 0) 
		{
			fprintf(
				stderr, 
				"checkStructSize: \n Warning! Inputted struct size: (%zu) is"
				"smaller than compiled structure, even acounting for padding:"
				" (%zu), exiting. For config %s.\n", 
				size, 
				compiled_size - largest_memory_alignment,
				name
			);
		}
		
		return_value = false;
	}

	return return_value;
}

multi_s clipParameter(
	const int32_t     verbosity,
	      multi_s     data,
		  parameter_s parameter
	) {
	
	multi_u value = data.value;
	
	const float lower_limit = parameter.lower_limit;
	const float upper_limit = parameter.upper_limit;

	switch(data.type) 
	{	
		case(none_e        ): break;
		case(bool_e        ): break;
		case(bool_array_e  ): break;
		
		case(int_e         ): 
			if((float)value.i > upper_limit)
			{
				if (verbosity > 1)
				{
					fprintf(
						stderr, 
						"clipParameter: \nWarning! Parameter \"%s\" value (%i)"
						"greater than allowed (%i) clipping! \n", 
						parameter.name,
						value.i,
						(int32_t) upper_limit
					);
				}
				value.i = (int32_t)upper_limit;
			}
			else if 
			((float)value.i < lower_limit)
			{
				printf("%f %f\n", lower_limit, upper_limit);
				if (verbosity > 1)
				{
					fprintf(
						stderr, 
						"clipParameter: \nWarning! Parameter \"%s\" value (%i)"
						"lower than allowed (%i) clipping! \n", 
						parameter.name,
						value.i,
						(int32_t)lower_limit
					);
				}
				value.i = (int32_t)lower_limit;
			}
			break;
		case(int_array_e   ): break;
		case(int_jagged_e  ): break;			
			
		case(float_e       ): break;
			if(value.f > upper_limit)
			{
				if (verbosity > 1)
				{
					fprintf(
						stderr, 
						"clipParameter: \nWarning! Parameter \"%s\" value (%f)"
						"greater than allowed (%f) clipping! \n", 
						parameter.name,
						value.f,
						upper_limit
					);
				}
				value.f = upper_limit;
			}
			else if 
			(value.f < lower_limit)
			{
				if (verbosity > 1)
				{
					fprintf(
						stderr, 
						"clipParameter: \nWarning! Parameter \"%s\" value (%f)"
						"lower than allowed (%f) clipping! \n", 
						parameter.name,
						value.f,
						lower_limit
					);
				}
				value.f = lower_limit;
			}
			break;
		case(float_array_e ): break;
		case(float_jagged_e): break;		

		case(char_e        ): break;
		case(char_array_e  ): break;
		
		case(string_e      ): break;		
		case(string_array_e): break;
			
		default:
			if (verbosity > 0)
			{
				fprintf(
					stderr, 
					"clipParameter: \nWarning! Type \"%s\" not recognised! \n", 
					typeToString(data.type)
				);
			}
		break;
	}
	
	data.value = value;
	
	return data;
}

void castToVoid(
	const int32_t      verbosity,
	const char        *string,
	const parameter_s  parameter,
	      void         *structure
	) {
		
	const type_e type        = parameter.type;
	const size_t size = getSizeOfType(type);
	
	void *value = NULL;
	
	switch(type)
	{	
		case(none_e):
		break;
		
		case(bool_e  ):
		case(int_e   ):
		case(float_e ):
		case(char_e  ):
		case(string_e): 
			;multi_s value_m = StringToMultiS(verbosity, string, type);
			         value_m = clipParameter(verbosity, value_m, parameter);
			           value = (void*) &value_m;
		break;
		
		case(bool_array_e  ):
		case(int_array_e   ):
		case(float_array_e ):
		case(char_array_e  ):
		case(string_array_e): 
			;array_s array_m = stringToArrayS(verbosity, string, type);
			           value = (void*) &array_m.data.ff.elements;
		break;
		
		
		default:
			if (verbosity > 1) 
			{
				fprintf(
					stderr, 
					"StringToMultiS: \nWarning! Type \"%i\" not recognised! \n", 
					type
				);
			}
		break;
	}
	
	
	memcpy(structure, value, size);
}

void castToVoidArray(
	const int32_t     verbosity,
	const char       *value_string,
    const type_e     *types, 
	const parameter_s parameter,
    const int32_t     index, 
          void       *structure
    ){
	
	const size_t structure_position = 
		getTotalTypeArraySize(types, index);	
	
	castToVoid(
		verbosity,
		value_string, 
		parameter, 
		(uint8_t*) &((char*) structure)[structure_position]
	);
}

char *separateString(
	      char *string,
	const char *separator,
	const char *name
	) {
	strtok(string, separator);
	string = strtok(NULL, separator);

	if (string == NULL) {

		fprintf(
            stderr, 
            "separateString: Error! Could not find closing %s, for parameter"
			" %s. \n", 
            separator, 
            name
        );
		
		string = "";
	}
	
	return string;
}

char* pullValueFromLine(
	const char            *parameter_start,              
	const loader_syntax_s  syntax,
	const char            *parameter_name,
	const type_e           type
	) {
	//Splits string from value indicator onwards until new line symbol.
	char *parameter_copy = strdup(parameter_start);
	char *value_string   = strtok(parameter_copy, syntax.new_line);
	
	switch (type) 
	{	
		case (none_e):
		
			strtok(value_string, syntax.string_separator);
			char* string_c = strtok(NULL, syntax.string_separator);
			
            if (string_c == NULL) 
            {	
                strtok(value_string, syntax.char_separator);
				string_c = strtok(NULL, syntax.string_separator);
				
                if (string_c == NULL) 
                {
					removeStringChars(
                        value_string, 
                        " =", 
                        &value_string
                    );
				} 
                else 
                {
					value_string = string_c;
				}
			} 
            else 
            {
				value_string = string_c;
			} 
		break;
	
		case (string_e):
			value_string = 
				separateString(
                    value_string, 
                    syntax.string_separator, 
                    parameter_name
                );
		break;
		
		case (char_e):
			value_string =
				separateString(
                    value_string, 
                    syntax.char_separator, 
                    parameter_name
                );	
		break; 
		
		default:
			removeStringChars(
                value_string, 
                " =", 
                &value_string
            );
			removeStringChars(
				value_string, 
				syntax.start_array, 
				&value_string
			);
			removeStringChars(
				value_string, 
				syntax.end_array, 
				&value_string
			);

		break;
	}
	
	return value_string;
}

bool checkNumParameters(
	const int32_t  verbosity,
	const int32_t *num_read,
	const int32_t  num_parameters,
	const int32_t  min,
	const int32_t  max
	) {
	
	bool pass = true;
	
	int32_t sum = 0;
	for (int32_t index = 0; index < num_parameters; index++) 
	{
		sum += num_read[index];
	}
	
	if (sum > max) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumParameters: \nError! Total num variables (%i) greater"
				" than max allowed (%i)! \n",
				sum, 
				max
			); 
		}
	}
	else if (sum < min) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumParameters: \nError! Total num variables (%i) smaller"
				" than minimum required (%i)! \n",
				sum, 
				min
			); 
		}
	} 
	
	return pass;
}

bool checkNumConfigs(
	const int32_t  verbosity,
	const int32_t  total_num_read,
	const int32_t  min,
	const int32_t  max,
	const char    *name
	) {
	
	bool pass = true;
	
	if (total_num_read > max) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumConfigs: \nError! Total num configs (%i) greater than"
				" max allowed (%i) for config %s.\n",
				total_num_read, max, name
			); 
		}
	}
	else if (total_num_read < min) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumConfigs: \nError! Total num configs (%i) smaller than"
				" minimum required (%i) for config %s! \n",
				total_num_read, min, name
			); 
		}
	} 
	
	return pass;
}

bool checkNumExtraConfigs(
	const int32_t  verbosity,
	const int32_t  total_num_read,
	const int32_t  min,
	const int32_t  max,
	const char    *name
	) {
	
	bool pass = true;
	
	if (total_num_read > max) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumExtraConfigs: \nError! Total num extra configs (%i)"
				" greater than max allowed (%i) for config %s! \n",
				total_num_read, max, name
			); 
		}
	}
	else if (total_num_read < min) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumExtraConfigs: \nError! Total num extra configs (%i)"
				"smaller than minimum required (%i) for config %s! \n",
				total_num_read, min, name
			); 
		}
	} 
	
	return pass;
}

bool checkNumExtraParameters(
	const int32_t  verbosity,
	const dict_s  *num_extra_parameters,
	const int32_t  min,
	const int32_t  max
	) {
	
	bool pass = true;
	
	dict_entry_s **entries = 
		returnAllEntries(num_extra_parameters); 
	const int32_t num_entries = 
		num_extra_parameters->num_entries;
	
	int32_t sum = 0;
	for (int32_t index = 0; index < num_entries; index++) 
	{
		sum += entries[index]->data.value.i;
	}
	
	if (sum > max) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumExtraParameters: \nError! Total num extra variables" 
				"(%i) greater than max allowed (%i)! \n",
				sum, max
			); 
		}
	} 
	else if (sum < min) 
	{
		pass = false;
		
		if (verbosity > 0)
		{
			fprintf(
				stderr, 
				"checkNumExtraParameters: \n Error! Total num extra variables"
				" (%i) smaller than minimum required (%i)! \n",
				sum, min
			); 
		}
	} 
	
	return pass;
}

bool checkParameterRequirements(
	const int32_t          verbosity,
	const int32_t         *num_read,
	const parameter_s     *parameters,
	const int32_t          num_parameters,
    const loader_config_s  config
	) {
	
    const int32_t total_min = config.min_inputed_parameters;
    const int32_t total_max = config.max_inputed_parameters;
    
	bool pass = true;
	
	for (int32_t index = 0; index < num_parameters; index++) 
	{
	
		const parameter_s  parameter = parameters[index];
		const int32_t      min       = parameter.min;
		const int32_t      max       = parameter.max;
		const char        *name      = parameter.name;
		const int32_t      read      = num_read[index];
		
		if (read < min) 
		{
			fprintf(
				stderr, 
				"checkParameterRequirements: \nWarning! Num instances (%i) of"
				" variable \"%s\" lower than required (%i) in config %s!\n", 
				read, name, min, config.name
			);
			
			pass = false;
		}
		else if (read > max)
		{
			fprintf(
				stderr, 
				"checkParameterRequirements: \nWarning! Num instances (%i) of"
				" variable \"%s\" higher than allowed (%i) in config %s!\n", 
				read, name, max, config.name
			);
			
			pass = false;
		}
	}
	
	pass = pass &&
		checkNumParameters(
			verbosity,
			num_read,
			num_parameters,
			total_min,
			total_max
		);
	
	return pass;
}

bool checkExtraParameterRequirments(
	const int32_t          verbosity,
	const parameter_s      parameter,
	const dict_s          *num_extra_parameters,
    const loader_config_s  config
	) {
    
    const int32_t total_min = config.min_extra_parameters;
    const int32_t total_max = config.max_extra_parameters;
	
	bool pass = true;
	
	const int32_t min = parameter.min;
	const int32_t max = parameter.max;
	
	dict_entry_s **entries = 
		returnAllEntries(num_extra_parameters); 
	const int32_t num_entries = 
		num_extra_parameters->num_entries;
	
	for (int32_t index = 0; index < num_entries; index++) {
		
		const dict_entry_s *entry = entries[index];
		const char         *name  = entry->string_key;
		const int32_t       read  = entry->data.value.i;
				
		if (read < min) 
		{
			fprintf(
				stderr, 
				"checkExtraParameterRequirments: \n Warning! Num instances (%i)"
				" of variable \"%s\" lower than required (%i)! \n", 
				read, name, min
			);
			
			pass = false;
		}
		else if (read > max)
		{
			fprintf(
				stderr, 
				"checkExtraParameterRequirments: \n Warning! Num instances (%i)"
				" of variable \"%s\" higher than allowed (%i) \n", 
				read, name, max
			);
			
			pass = false;
		}
	}
	
	pass = pass &&
		checkNumExtraParameters(
			 verbosity,
			 num_extra_parameters,
			 total_min,
			 total_max
		);
	
	return pass;
}

bool checkNameRequirments(
    const bool            verbosity,
    const char           *name,
    const loader_config_s config
    ) {
    
    bool name_read = (name != NULL);
    
    const necessity_e  necessity = config.name_necessity;

    bool pass = false;
    
    switch (necessity) 
    {
        case(required_e):
            pass = name_read;
            
            if ((verbosity > 0) && !pass) 
            {
                fprintf(
					stderr, 
					"checkNameRequirments: \nError! Config Name required. \n"
				);
            }
        break;
        
        case(optional_e):
            pass = true;
        break;
        
        case(excluded_e):
            pass = !name_read;
            
            if ((verbosity > 0) && !pass) 
            {
                fprintf(
					stderr, 
					"checkNameRequirments: \nError! Unnamed configs are not"
					" allowed in this context. \n"
				);
            }
        break;
        
        default:
            pass = true;
        break;
    }

    return pass;
}

bool checkConfigRequirements(
    const int32_t          verbosity,
    const bool             is_superconfig, 
	const int32_t         *num_read,
    const int32_t          total_num_read,
	const loader_config_s *subconfigs,
	const int32_t          num_subconfigs,
	const loader_config_s  config
    ) {
    
    int32_t total_min = config.min_num_subconfigs;
    int32_t total_max = config.max_num_subconfigs;
    
    int32_t extra_min = config.min_extra_subconfigs;
    int32_t extra_max = config.max_extra_subconfigs;
    
    int32_t sum = 0;
	for (int32_t index = 0; index < num_subconfigs*is_superconfig; index++) 
	{
		sum += num_read[index];
	}
    
	bool pass = true;
	
	for (int32_t index = 0; index < num_subconfigs*is_superconfig; index++) 
	{	
		const loader_config_s  config = subconfigs[index];
		const int32_t          min    = config.min;
		const int32_t          max    = config.max;
		const char            *name   = config.name;
		const int32_t          read   = num_read[index];
        		
		if (read < min) 
		{
			fprintf(
				stderr, 
				"checkConfigRequirements: \nWarning! Num instances (%i) of"
				" config \"%s\" lower than required (%i)! \n", 
				read, name, min
			);
			
			pass = false;
		}
		else if (read > max)
		{
			fprintf(
				stderr, 
				"checkConfigRequirements: \nWarning! Num instances (%i) of "
				" config \"%s\" higher than allowed (%i) \n", 
				read, name, max
			);
			
			pass = false;
		}
	}
	
    if (is_superconfig)
    {
        pass = pass && 
            checkNumConfigs(
                verbosity,
                total_num_read,
                total_min,
                total_max,
				config.name
            );

        pass = pass &&
            checkNumExtraConfigs(
                verbosity,
                total_num_read - sum,
                extra_min,
                extra_max,
				config.name
            );
    }
    	    
    return pass;
}

type_e guessDataTypeFromString(
	const char *string
	) {

	type_e guess = float_e;
	
	if (isdigit(string[0]) || (string[0] == '-'))
	{
		if ( strchr(string, '.') ) 
		{
			guess = float_e;
		} 
		else 
		{
			guess = int_e;
		}
	} 
	else if ((!strcmp(string, "true")) || (!strcmp(string, "false"))) 
	{
		guess = bool_e;	
	} 
	else 
	{
		guess = string_e;
	}
	
	return guess;
}

type_e guessTypeFromConfigContext(
	                 char *string,
	const loader_syntax_s  syntax
	) {
	
	type_e estimated_type;
	if (strchr(string, *syntax.string_separator)) 
	{
		estimated_type = string_e;
		removeStringChars(string, syntax.string_separator, &string);
	}
	else if (strchr(string, *syntax.char_separator))
	{
		estimated_type = char_e;
		removeStringChars(string, syntax.char_separator, &string);
	} 
	else if (strchr(string, *syntax.start_array))
	{
		removeStringChars(string, syntax.start_array, &string);
		removeStringChars(string, syntax.end_array, &string);
		
		estimated_type = guessTypeFromConfigContext(string, syntax);
		
		switch (estimated_type)
		{
			case bool_e:
				estimated_type = bool_array_e;
			break;
			
			case int_e:
				estimated_type = int_array_e;
			break;
			
			case float_e:
				estimated_type = float_array_e;
			break;
			
			case char_e:
				estimated_type = char_array_e;
			break;
			
			case string_e:
				estimated_type = string_array_e;
			break;
			
			default:
				fprintf(
					stderr, 
					"guessTypeFromConfigContext \nWarning! Arrays of arrays not"
					"handled by config loader. This error could be caused by an"
					"extra \"%s\" in config file.\n",
					syntax.start_array
				);
				estimated_type = none_e;
			break;
		}
	} 
	else {
		estimated_type = 
			guessDataTypeFromString(string);
	}
		
	return estimated_type;
}

void addExtraParameter(
	const int32_t  verbosity, 
	      dict_s  *extra_parameters,
	      char    *name,
	const type_e   type,
	const int32_t  index, 
	const int32_t  start_index,
	const int32_t  num_read,
	      char    *value_string
	) {
	
	char* extra_parameter_name = name;
	if (num_read > start_index) 
	{
		asprintf(
			&extra_parameter_name, 
			"%s_%i", 
			name, 
			index - start_index
		);
	} 
	
	multi_s data = 
		StringToMultiS(verbosity, value_string, type);
	insertDictEntry(extra_parameters, data, extra_parameter_name);
}

bool checkDefaultParameter(
	const int32_t     verbosity,
	const parameter_s parameter
	) {
	
	bool pass = true;
	
	const int32_t min  = parameter.min;
	const int32_t max  = parameter.max;
	const char   *name = parameter.name;

	if (min > max) {

		if (verbosity > 0) 
		{
			fprintf(
				stderr, 
				"checkDefaultParameter: \nError! Min num parameter instances"
				"greater than max for parameter: \"%s\"!", 
				name
			);
		}

		pass = false;
	}

	return pass;
}

bool checkParameters(
	const int32_t      verbosity,
	const parameter_s *parameters,
	const int32_t      num_parameters
	) {
	
	bool pass = true;
	
	for (int32_t index = 0; index < num_parameters; index++) {
		
		const parameter_s parameter = parameters[index];
		
		const int32_t min  = parameter.min;
		const int32_t max  = parameter.max;
		const type_e  type = parameter.type;
		const char   *name = parameter.name;
		
		if (min > max) {
			
			if (verbosity > 0) 
			{
				fprintf(
					stderr, 
					"checkParameters: \nError! Min num parameter instances"
					" greater than max for parameter: \"%s\"!", 
					name
				);
			}
			
			pass = false;
		}
		
		if (type == none_e)
		{
			if (verbosity > 0) 
			{
				fprintf(
					stderr, 
					"checkParameters: \n Error! Defined parameter \"%s\" cannot"
					" have type none_e!", 
					name
				);
			}
			
			pass = false;
		}
	}
	
	return pass;
}

bool checkDefaultConfig(
	const int32_t         verbosity,
    const bool            is_superconfig,
	const loader_config_s config
	) {
	
	bool pass = true;
	
    if (is_superconfig)
    {
    
        const int32_t min  = config.min;
        const int32_t max  = config.max;
        const char   *name = config.name;

        if (min > max) {

            if (verbosity > 0) 
            {
                fprintf(
                    stderr, 
                    "checkDefaultConfig: \nError! Min num config instances"
					"greater than max for config: \"%s\"!", 
                    name
                );
            }

            pass = false;
        }
    }
    
	return pass;
}

bool checkConfigs(
	const int32_t          verbosity,
    const bool             is_superconfig,
	const loader_config_s *configs,
	const int32_t          num_configs
	) {
	
	bool pass = true;
	
    if (is_superconfig)
    {
        for (int32_t index = 0; index < num_configs; index++) {

            const loader_config_s config = configs[index];

            const int32_t min  = config.min;
            const int32_t max  = config.max;
            const char   *name = config.name;

            if (min > max) {

                if (verbosity > 0) 
                {
                    fprintf(
                        stderr, 
                        "checkConfigs: \nError! Min num config instances"
						"greater than max for config: \"%s\"!", 
                        name
                    );
                }

                pass = false;
            }
        }
    }
	
	return pass;
}

loader_config_s setupDefaultConfig(
    const loader_config_s    config 
    ) {

    loader_config_s  default_config;
    if (config.is_superconfig)
    {
        default_config = *config.default_subconfig;
    }
    else
    {
        default_config = config;
    }

    return default_config;
}

bool checkLoaderConfig(
    const int32_t         verbosity,
    const loader_config_s config
    ) {
    
    const size_t           largest_memory_alignment = sizeof(double);
    const size_t           compiled_struct_size     = config.struct_size;

    const bool             is_superconfig           = config.is_superconfig;

    const int32_t          num_defined_parameters   = 
		config.num_defined_parameters;
          parameter_s     *defined_parameters       = 
		config.defined_parameters;
          parameter_s      default_parameter        = config.default_parameter;

          int32_t          num_defined_subconfigs   = 
		config.num_defined_subconfigs;
    const loader_config_s *defined_subconfigs       = 
		config.defined_subconfigs;
    
    // Calculate struct size from types list:
    
    type_e  types[num_defined_parameters];
    for (int32_t index = 0; index < num_defined_parameters; index++) 
    {	
        types[index]           = defined_parameters[index].type;
    }
    
	const size_t struct_size = 
		getTotalTypeArraySize(types, num_defined_parameters);
    
    bool pass = true;
    
    loader_config_s default_config 
		= setupDefaultConfig( config);
    	
	pass = pass &&
        checkStructSize(
            verbosity, 
            struct_size, 
            compiled_struct_size, 
            largest_memory_alignment,
            config.name
        );

	pass = pass &&
        checkTypeOrder(
            verbosity, 
            types, 
            num_defined_parameters
        );
    
	pass = pass && 
        checkDefaultParameter(
            verbosity,
            default_parameter
        );

    pass = pass &&
        checkParameters(
            verbosity,
            defined_parameters, 
            num_defined_parameters
        ); 
        
    pass = pass &&
        checkDefaultConfig(
            verbosity,
            is_superconfig,
            default_config
        );
    
    pass = pass &&
        checkConfigs(
            verbosity,
            is_superconfig,
            defined_subconfigs, 
            num_defined_subconfigs
        );
    
    return pass;
}

map_s createParameterMap(
    const int32_t      num_parameters,
    const parameter_s *parameters
    ) {

	char   *names[num_parameters];
	for (int32_t index = 0; index < num_parameters; index++) 
	{	
		names[index] = parameters[index].name;
	}
    
    // Create parameter name maping:
	map_s map = 
        createMap(names, num_parameters);
    
    return map;
}

map_s createConfigMap(
    const loader_config_s *subconfigs,
    const int32_t          num_subconfigs,
    const loader_config_s  config
    ) {
    
     char **names;
     if (config.is_superconfig == true) 
     {   
        names = malloc(sizeof(char*) * (size_t) num_subconfigs);
        
        for (int32_t index = 0; index < num_subconfigs; index++) 
        {	
            names[index] = subconfigs[index].name;
        } 
    }
    else
    {
        names = malloc(sizeof(char*));
        *names = config.name;
    }
    
    // Create config name maping:
    map_s map = 
        createMap(names, num_subconfigs);
    
    return map;
}

parameter_s *overwriteParameters(
    const parameter_s*     default_parameters,
    const map_s            default_parameter_map,
    const loader_config_s  priority_config
    ) {
    
    const int32_t      num_priority_parameters  = 
		priority_config.num_defined_parameters;
    const parameter_s *priority_parameters      = 
		priority_config.defined_parameters;
    
    parameter_s *new_parameters = 
		malloc(sizeof(parameter_s) * (size_t) default_parameter_map.length);
    memcpy(
		new_parameters, 
		default_parameters, 
		sizeof(parameter_s) * (size_t) default_parameter_map.length
	);
	
    // Create config lists:    
    map_s priority_name_map = 
        createParameterMap(
            num_priority_parameters,
            priority_parameters
        );

    for (int32_t index = 0; index < num_priority_parameters; index++) 
    {
        const char *parameter_name = getMapKey(priority_name_map, index);

        if (parameter_name != NULL) 
        {
            int32_t default_index = 
				getMapIndex(default_parameter_map, parameter_name);

            if (default_index > -1) 
            {
                new_parameters[default_index] = priority_parameters[index];
            } 
            else
            {
                fprintf(
					stderr, 
					"overwriteParameters: \nWarning! Parameter \"%s\" name not"
					" found in defined parameters! \n", 
					parameter_name
				);
                break;
            }
        }
        else
        {
            fprintf(
				stderr, 
				"overwriteParameters: \nWarning! Parameter name at index (%i)"
				" not found, check num defined parameters \n", 
				index
			);
            break;
        }
    }
    
    return new_parameters;
}

void freeConfigData(
    loader_data_s config_data
    ) {
    
    for (
		int32_t index = 0; 
		index < config_data.total_num_subconfigs_read; 
		index++
	) {
        if (config_data.subconfigs[index].structure != NULL) 
        {
            free(config_data.subconfigs[index].structure);
        }
        if( config_data.subconfigs[index].name != NULL)
        {
            free(config_data.subconfigs[index].name);
        }
    }
    
    if (config_data.extra_parameters != NULL)
    {
        free(config_data.extra_parameters);
		config_data.extra_parameters = NULL;
    }
	
	if (config_data.subconfigs != NULL)
    {
		free(config_data.subconfigs);
		config_data.subconfigs = NULL;
    }
    
}

void **reorderConfigs(
    const loader_config_s    config,
    const loader_data_s config_data
    ) {
    
    const int32_t num_defined = config.num_defined_subconfigs;

    const map_s             name_map       = 
		config_data.subconfig_name_map;
    const int32_t          *num_read       = config_data.num_subconfigs_read;
    const int32_t           total_num_read = 
		config_data.total_num_subconfigs_read;
    const loader_data_s    *subconfig_data = config_data.subconfigs;
    
    void **config_structs = calloc((size_t) total_num_read, sizeof(void*));
    
    int32_t unique_defined_read = 0;
	
    int32_t *cumulative_missing = 
		malloc(sizeof(int32_t) * (size_t) (num_defined + 1));
    cumulative_missing[0] = 0;

    for (int32_t index = 0; index < num_defined; index++)
    {
        unique_defined_read += (bool) num_read[index];
        cumulative_missing[index + 1] = 
            cumulative_missing[index] + !((bool) num_read[index]);
    }

    int32_t undefined_index = unique_defined_read;
    for (int32_t index = 0; index < total_num_read; index++) 
    {
        const char  *name = subconfig_data[index].name;

        int32_t defined_index = -1;
        if( name != NULL)
        {
            defined_index = getMapIndex(name_map, name);
        }

        if (defined_index > -1) 
        {
            int32_t adjusted_index = 
				defined_index - cumulative_missing[defined_index];
            config_structs[adjusted_index] = subconfig_data[index].structure; 
        }
        else
        {
            config_structs[undefined_index] = subconfig_data[index].structure; 
            undefined_index++;
        }
    }
    
    return config_structs;
}

void **keepConfigs(
    const loader_data_s config_data
    ) {
    
    const int32_t        total_num_read = config_data.total_num_subconfigs_read;
    const loader_data_s *subconfig_data = config_data.subconfigs;
    
    void **config_structs = calloc((size_t) total_num_read, sizeof(void*));
    
    for (int32_t index = 0; index < total_num_read; index++) 
    {
        config_structs[index] = subconfig_data[index].structure; 
    }
    
    return config_structs;
}

void **setConfigStructs(
    const loader_config_s     config,
    const loader_data_s   config_data
    ) {
    
    const bool is_superconfig = config.is_superconfig;
    const bool reorder        = config.reorder;
    
    void **config_structs = NULL;
    
    if (is_superconfig)
    {
        if (reorder) 
        {
            config_structs = 
                reorderConfigs(config, config_data);
        }
        else
        {
            config_structs = 
                keepConfigs(config_data);
        }
    }
    else
    {
        config_structs = calloc((size_t) 1, sizeof(void*));
        config_structs[0] = config_data.structure; 
    }
    
    return config_structs;
}

loader_data_s setupConfigData(
    const loader_config_s config,
    const int32_t         num_configs
    ) {
    
    //Derived Parameters:
    const size_t       compiled_struct_size   = config.struct_size;
    
    const int32_t      num_defined_parameters = config.num_defined_parameters;
          parameter_s *defined_parameters     = config.defined_parameters;
		  int32_t      num_defined_subconfigs = config.num_defined_subconfigs;
    
    loader_data_s config_data = 
	{
       .name       = NULL,
       .subconfigs = NULL,
       .config     = config
    };
       
    // Create config lists:    
    config_data.subconfig_name_map = 
        createConfigMap(
            config.defined_subconfigs,
            config.num_defined_subconfigs,
            config
        ); 
      
    // Allocate memory for subconfigs:
    config_data.subconfigs = 
        calloc((size_t) num_configs, sizeof(loader_data_s));
        
    // Create bins to hold number of each parameter name read:
    config_data.num_subconfigs_read = 
        calloc((size_t) config.num_defined_subconfigs, sizeof(int32_t));
    
    config_data.structure = 
        calloc(1, compiled_struct_size);

    // Create parameter name maping:
    config_data.parameter_name_map = 
        createParameterMap(
            num_defined_parameters,
            defined_parameters
        );

    config_data.num_parameters_read = 
        calloc((size_t) num_defined_parameters, sizeof(int32_t));
    
    config_data.total_num_subconfigs_read = 0;
    
    config_data.extra_parameters = 
        makeDictionary((1+num_defined_parameters)*100);   

    // Create dictionary to count instances of extra parameters:
    config_data.num_extra_parameters = 
        makeDictionary((1+num_defined_parameters)*100);
	
	// Create dictionary to count instances of extra subconfigs:
	config_data.num_extra_configs = 
        makeDictionary((1+num_defined_subconfigs)*100);

    return config_data;
}

bool checkAllParameterRequirments(
    const int32_t         verbosity,
    const loader_data_s   config_data,
    const loader_config_s default_config,
    const loader_config_s superconfig
    ) {
    
    bool pass = true;
    
    loader_config_s config = default_config;

    const char *name = config_data.name;
	
    if (name != NULL)
    {
        const map_s subconfig_name_map =
            createConfigMap(
                superconfig.defined_subconfigs,
                superconfig.num_defined_subconfigs,
                superconfig
            );

        const int32_t config_name_index = 
                getMapIndex(subconfig_name_map, name);

        if (config_name_index > -1) 
        {
			if (config.inherit) 
			{
				config = superconfig.defined_subconfigs[config_name_index];

				config.defined_parameters = overwriteParameters(
					config_data.config.defined_parameters,
					config_data.parameter_name_map,
					config
				);
				
				config.num_defined_parameters = 
					config_data.config.num_defined_parameters;
			}
        }
    }

    // Check neccesity requirements:
    if (!checkParameterRequirements(
        verbosity,
        config_data.num_parameters_read,
        config.defined_parameters,
        config.num_defined_parameters,
        config
    ) 
    ||
    !checkExtraParameterRequirments(
        verbosity,
        config.default_parameter,
        config_data.num_extra_parameters,
        config
    )
    ||
    !checkNameRequirments(
        verbosity,
        config_data.name,
        config
    ))
    {
        pass = false;
    }
    
    return pass;
}

bool checkAllRequirements(
    const int32_t         verbosity,
    const loader_data_s   config_data,
    const loader_config_s config,
    const loader_config_s superconfig
    ) {
    
    bool pass = true;
    
    if (config.is_superconfig)
    {
        if (!checkConfigRequirements(
                verbosity,
                config.is_superconfig,
                config_data.num_subconfigs_read,
                config_data.total_num_subconfigs_read,
                config.defined_subconfigs,
                config.num_defined_subconfigs,
                config))
        {
            pass = false;
        }
    }
    
	if (config.has_parameters)
    {
        if (!checkAllParameterRequirments(
            verbosity,
            config_data,
            config,
            superconfig
            ))
        {
            pass = false;
        }
    }
    
    return pass;
}

loader_config_s allocateNewConfig(
          loader_data_s   *config_data,
          loader_config_s *config,
    const loader_config_s  superconfig,
          loader_config_s  default_config,
    const int32_t          config_name_index
    ) { 
    
    const int32_t num_defined_subconfigs = 
        config->num_defined_subconfigs;

    const int32_t num_defined_parameters = 
        config->num_defined_parameters;

    *config = 
        superconfig.defined_subconfigs[config_name_index];

    if (config->inherit)
    {
        config->defined_parameters = overwriteParameters(
            config_data->config.defined_parameters,
            config_data->parameter_name_map,
            *config
        );

        config->num_defined_parameters = 
            config_data->config.num_defined_parameters;
    }
    else
    {
        config_data->config = *config;
        config_data->parameter_name_map = 
            createParameterMap(
                config->num_defined_parameters,
                config->defined_parameters
            );

        config_data->num_subconfigs_read = 
            realloc(
                config_data->num_subconfigs_read, 
                (size_t) config->num_defined_subconfigs
                * sizeof(int32_t)
            );
            
        for (
            int32_t index = num_defined_subconfigs; 
            index < config->num_defined_subconfigs;
            index++
            ) {
            
            config_data->num_subconfigs_read[index] = 0;        
        }
        
        config_data->num_parameters_read = 
            realloc(
                config_data->num_parameters_read,
                (size_t) config->num_defined_parameters 
                * sizeof(int32_t)
            );
            
        for (
            int32_t index = num_defined_parameters; 
            index < config->num_defined_parameters;
            index++
            ) {
            
            config_data->num_parameters_read[index] = 0;        
        }

        if (config->is_superconfig)
        {
            config_data->subconfig_name_map = 
                createConfigMap(
                    config->defined_subconfigs,
                    config->num_defined_subconfigs,
                    *config
                ); 
        }
        
        default_config = setupDefaultConfig(*config);
    }
    
    return default_config;
}

loader_data_s readSubconfig(
    const int32_t            verbosity,
          loader_syntax_s    syntax,
          loader_config_s    config,
          loader_config_s    superconfig,
          FILE              *file
    ) {
    
    //Derived Parameters:
    const bool         is_superconfig         = config.is_superconfig;

    const int32_t      num_defined_parameters = config.num_defined_parameters;
          parameter_s *defined_parameters     = config.defined_parameters;
          parameter_s  default_parameter      = config.default_parameter;
    
    int32_t num_configs  = 1;
    int32_t config_index = 0;
    
    //Setup config_data:
    loader_data_s config_data = 
        setupConfigData(config, num_configs);
    
    if (!checkLoaderConfig(verbosity, config) ) 
    {	
		num_configs = 0;
		if (verbosity > 0) 
		{
			fprintf(stderr, "readSubconfig: \nWarning! Cannot load config. \n");
		}
	} 
	else 
	{
        // Setup Default Configs:
        loader_config_s default_config = 
            setupDefaultConfig(config);
        
        // Create parameter lists:
        type_e  types[num_defined_parameters];
        for (int32_t index = 0; index < num_defined_parameters; index++) 
        {	
            types[index]           = defined_parameters[index].type;
        }

		// Setup and reset config wide parameters:
        bool             name_read    = false;
		size_t           line_length  = 1; 
		// String which will contain the read in line:
		char            *line_string  = NULL; 
		int32_t          line_index   = 0;
        char            *config_name  = NULL;

		// Reading file, loop over lines:
		while (getline(&line_string, &line_length, file) != EOF) 
		{
			line_index++;
			
			// Setup and reset line wide parameters:
			bool     parameter_read       = false;
            bool     parameter_end        = false;
			bool     get_value            = false;
			int32_t  parameter_start      = 0;

			int32_t  parameter_index      = 0;
			type_e   parameter_type       = bool_e;
			char    *parameter_name       = calloc(line_length, sizeof(char));
			bool     parameter_recognised = false;

			int32_t  char_index = 0;
			
			// Reading Line, loop over characters:
			while(line_string[char_index] != 0) 
			{                 
				// Checks for config closing anywhere in line:
				if (strchr(syntax.end_config, line_string[char_index])) 
				{				
					config_data.total_num_subconfigs_read = config_index;
                                                            
                    if (!checkAllRequirements(
                        verbosity,
                        config_data,
                        config,
                        superconfig))
                    {
                        config_data.total_num_subconfigs_read = -1;
                    }
                                        
                    return config_data;
				} 

				// Checks for new config opening anywhere in line:
				if (
					   (strchr(syntax.start_config, line_string[char_index])) 
					&& is_superconfig
				) {                      
                    // If number of configs greater than allocated in memory,
                    // allocate more memory:
                    if ((config_index >= num_configs))
                    {
                        num_configs = 
                            (int32_t) ceil((float) num_configs * 1.5f);
                        config_data.subconfigs = 
                            realloc(
								config_data.subconfigs, 
								sizeof(loader_data_s) * (size_t) num_configs
							);
                    }

                    // Intilise subconfig to default:
                    loader_config_s old_config = config;
                                    config     = default_config;
						
					if (config.inherit)
					{
						// Set default parameters:    
						config.defined_parameters = overwriteParameters(
							config_data.config.defined_parameters,
							config_data.parameter_name_map,
							config
						);

						config.num_defined_parameters = 
							config_data.config.num_defined_parameters;
					}
                                        
                    config_data.subconfigs[config_index].config = config;
					
                    config_data.subconfigs[config_index] = readSubconfig(
                        verbosity,
                        syntax,
                        config,
                        config_data.config,
                        file
                    );  
					
					char_index = 0;

                    if (config_data.subconfigs[config_index]
                        .total_num_subconfigs_read < 0)
                    {
                        config_data.total_num_subconfigs_read = -1;
                        return config_data;
                    }

                    const char *name = 
                        config_data.subconfigs[config_index].name;
					
                    if (name != NULL)
                    {
                        const int32_t config_name_index = 
                            getMapIndex(config_data.subconfig_name_map, name);
						
                        if (config_name_index > -1) 
                        {
                            config_data
                                .num_subconfigs_read[config_name_index]++;
                        }
                        else
                        {
                            dict_entry_s* entry = 
                                findDictEntry_(
                                    config_data.num_extra_configs, 
                                    name
                                );

                            if (entry == NULL) 
                            {								
                                multi_s data;
                                data.value.i = 1;
                                data.type = int_e;

                                insertDictEntry(
                                    config_data.num_extra_configs, 
                                    data, 
                                    name
                                );		

                                data.value.i++;	
                            }
                            else
                            {
                                entry->data.value.i++;
                            }
                        }   
                    }
                    
                    config = old_config;
					config_index++; 
					
					// Checks for new segment character returns if found:
					if (config_index >= config.early_exit_index) 
					{ 
						config_data.total_num_subconfigs_read = config_index;

						if (!checkAllRequirements(
							verbosity,
							config_data,
							config,
							superconfig))
						{
							config_data.total_num_subconfigs_read = -1;
						}

						return config_data;
					}
					                    
                    break;	
				}                 
				else if (strchr(syntax.start_config, line_string[char_index]))
				{
                    fprintf(
						stderr, 
						"readSubconfig: \nWarning! Unexpected config opening:"
						" '{' in config \"%s\". Exiting read attempt! \n", 
						config_data.name
					); 

                    config_data.total_num_subconfigs_read = -1;
                    return config_data;
				}
                
                // Checks for config name character and reads name if found:
                if (strchr(syntax.start_name, line_string[char_index])) 
				{ 	
                    if (name_read == false) 
                    {
                        config_name = 
							strtok(
                                &line_string[char_index + 1], syntax.end_name
                            );
                        config_data.name = strdup(config_name);
						
                        const map_s subconfig_name_map =
                        createConfigMap(
                            superconfig.defined_subconfigs,
                            superconfig.num_defined_subconfigs,
                            superconfig
                        );
                        
                        const int32_t config_name_index = 
                                getMapIndex(subconfig_name_map, config_name);
								
                        if (config_name_index > -1) 
                        {
                        
                            default_config = allocateNewConfig(
                                &config_data,
                                &config,
                                superconfig,
                                default_config,
                                config_name_index
                            );
                        }
												                        
                        name_read = true;
                    }
                    else if (verbosity > 1)
                    {
                        fprintf(
                            stderr, 
                            "readSubconfig: \nWarning! More than one name "
							"detected inside config \"%s\". Exiting read "
							"attempt! \n",
                            config_data.name
                        );
						
						config_data.total_num_subconfigs_read = -1;
						return config_data;
                    }
					break; 
				}
    
				// Checks for comment character and starts a new line if found:
				if (strchr(syntax.comment, line_string[char_index])) 
				{ 	
					break; 
				}

				// Finding start parameter names in line:
				if (!isspace(line_string[char_index]) && !parameter_read) 
				{ 	
					parameter_start = char_index;
					parameter_read = !parameter_read;
				}
                
				// Recording parameter name into string:
				if (parameter_read && !isspace(line_string[char_index])) 
				{	
					parameter_name[char_index - parameter_start] = 
						line_string[char_index];
				}
				
				// On parameter name end:
				else if (
					   isspace(line_string[char_index]) 
					&& (!parameter_end && parameter_read)
				) {                
                    parameter_end = true;
                    
					parameter_index = 
						getMapIndex(
                            config_data.parameter_name_map, parameter_name
                        );
                        										
					if (parameter_index > -1) 
					{
						parameter_type = 
							config_data.config
                                .defined_parameters[parameter_index].type;
						config_data.num_parameters_read[parameter_index]++;
                        
						parameter_recognised = true;
					} 
					else 
					{	
						parameter_type = default_parameter.type;
						
						dict_entry_s* entry = 
							findDictEntry_(
								config_data.num_extra_parameters, 
								parameter_name
							);
						
						if (entry == NULL) 
						{								
							multi_s data;
							data.value.i = 1;
							data.type = int_e;
							
							insertDictEntry(
								config_data.num_extra_parameters, 
								data, 
								parameter_name
							);		
							
							data.value.i++;	
						}
						else
						{
							entry->data.value.i++;
						}						
						parameter_recognised = false;
						parameter_index = 0;
						
						if ((config.max_extra_parameters == 0) 
							&&
							(verbosity > 1))
						{
							fprintf(
								stderr, 
								"readSubconfig: \nWarning! Unrecognised"
								" parameter name: \"%s\" in config %s.\n", 
								parameter_name,
								config.name
							);
						}
					}
					
					get_value = true;
				}

				//Finds the value after detecting value indicatior character
				if (
					   get_value
					&& strchr(syntax.value_indicator, line_string[char_index])
				) {	
					char *value_string = 
						pullValueFromLine(
							&line_string[char_index],
							syntax,
							parameter_name,
							parameter_type 
						);
					
					if (parameter_type == none_e) 
					{
						parameter_type = guessTypeFromConfigContext(
							value_string, 
							syntax
						);
					}
					
					if (value_string != NULL) 
					{	
						int32_t extra_parameter_start_index = 0;
						
						if (   (parameter_recognised) 
							&& (config_data.num_parameters_read[parameter_index] > 1)
						) {
							extra_parameter_start_index = 2;
						} 
						else
						{
							extra_parameter_start_index = 1;
						}
						
						if (   (parameter_recognised) 
							&& (config_data.num_parameters_read[parameter_index] < 2)
						) {
							castToVoidArray(
								verbosity,
								value_string, 
								types, 
								config.defined_parameters[parameter_index],
								parameter_index, 
								config_data.structure
							);
						}
						else 
						{
							addExtraParameter(
								verbosity,
								config_data.extra_parameters,
								parameter_name,
								parameter_type,
								parameter_index, 
								extra_parameter_start_index, 
								config_data
                                    .num_parameters_read[parameter_index],
								value_string
							);
						}
					} 
                    else 
                    {
                        if (verbosity > 0)
                        {
                            fprintf(
                                stderr, 
                                "readSubconfig: \nWarning! No value string read"
								" in config %s! \n",
								config_name
                            );
                        }
					}

					//Resets parameter name:
					memset(parameter_name, 0, strlen(parameter_name));

					get_value = false;
				}

				char_index++; 
			}
		}
        
        config_data.total_num_subconfigs_read = config_index;
    }
    
    if (!checkAllRequirements(
			verbosity,
			config_data,
			superconfig,
			superconfig
		)
	) {
        config_data.total_num_subconfigs_read = -1;
    }
    
    return config_data;
}

void* readConfig(
	 const int32_t            verbosity,
     const char              *file_name,  
	 const loader_config_s    config,
	       loader_data_s     *ret_cofig_data,
           int64_t           *file_position
    ){
        
    // Initlise empty pointer:
    void    *config_structs       = NULL;
        
	// Opening file:
	FILE* file = NULL;	
        
    loader_data_s config_data;
    config_data.total_num_subconfigs_read = -1;
	config_data.extra_parameters          = NULL;
	config_data.subconfigs                = NULL;
	
    if (checkOpenFile(verbosity, file_name, "r", &file))
    {        
        fseek(file, *file_position, SEEK_SET);
        
        config_data = readSubconfig(
            verbosity,
            syntax,
            config,
            config,
            file
        );
        
        *file_position = ftell(file);
        
        fclose(file);
    } 
    
    if (config_data.total_num_subconfigs_read < 0)
    {
        freeConfigData(config_data);
        config_structs = NULL;
    }
    else
    {
        config_structs = setConfigStructs(
            config_data.config,
            config_data
        ); 
    }
		
	*ret_cofig_data = config_data;
	
	return config_structs;
}

size_t *createStructureParameterMap(
	 const int32_t  verbosity,
     const type_e  *types, 
     const int32_t  length
    ) {
    
	checkTypeOrder(verbosity, types, length);
	
	size_t *structure_map = malloc(sizeof(size_t) * (size_t) length);
	
	for (int32_t index = 0; index < length; index++)
	{
		structure_map[index] = getTotalTypeArraySize(types, index);
	}
	
	return structure_map;
}

void* pullValueFromStruct(
    const void    *structure        , 
    const size_t   parameter_position, 
    const size_t   size
    ) {
	
	//Warning check your compiler padding before using this function:
	void *value = malloc(size);
	memcpy(value, &((uint8_t*) structure)[parameter_position], size);
	
	return value;
}

void pullUInt16ArrayFromStruct(
    const int32_t      start    , 
    const int32_t      stop     , 
    const construct_s *construct, 
    const void        *structure, 
          uint16_t    *array
    ) {
	
	uint16_t *pointer = NULL;
	
	for (int32_t index = start; index < stop; index++) 
	{
		pointer = 
			(uint16_t*) pullValueFromStruct(
				structure, 
				construct->structure_map[index], 
				sizeof(uint16_t)
			);
		array[index - start] = *pointer;
	}
	
	free(pointer);
}

#endif