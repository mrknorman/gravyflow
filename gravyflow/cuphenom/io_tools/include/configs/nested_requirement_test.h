#ifndef TEST_CONFIG_STRUCT
#define TEST_CONFIG_STRUCT
//Basic test config:
typedef struct ConfigTest{
	
	char    *parameter_string;
	float    parameter_float;
	int32_t  parameter_int;
	bool     parameter_bool;
	char     parameter_char;

} test_config_s;

#else

// Parameters:
const int32_t num_defined_parameters = 5;
parameter_s defined_parameters[] = 
{
	{"parameter_string", string_e, 0, 1        , -FLT_MAX, FLT_MAX},
	{"parameter_float" , float_e , 0, INT32_MAX, -FLT_MAX, FLT_MAX},
	{"parameter_int"   , int_e   , 0, INT32_MAX, -FLT_MAX, FLT_MAX},
	{"parameter_bool"  , bool_e  , 0, 1        , -FLT_MAX, FLT_MAX},	
	{"parameter_char"  , char_e  , 0, 0        , -FLT_MAX, FLT_MAX}	
};

parameter_s default_parameter = 
	{"default_parameter", none_e, 0, 0, 0.0f, 0.0f};

// Subconfigs:
loader_config_s subconfig = 
{
	.name                   = "variable_config_test",
	.name_necessity         = required_e,

	.is_superconfig         = false,
	.has_parameters         = true,
	.inherit                = true,
    .min                    = 1,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_defined_parameters,
	.min_inputed_parameters = 0,
	.max_inputed_parameters = num_defined_parameters,
	.defined_parameters     = defined_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,

	.struct_size            = sizeof(test_config_s)
};

parameter_s priority_parameters_0[] = 
{
	{"parameter_string", string_e, 1, 1, -FLT_MAX, FLT_MAX},
	{"parameter_float" , float_e , 1, 1, -FLT_MAX, FLT_MAX},
	{"parameter_int"   , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"parameter_bool"  , bool_e  , 0, 1, -FLT_MAX, FLT_MAX},	
	{"parameter_char"  , char_e  , 0, 0, -FLT_MAX, FLT_MAX}	
};

parameter_s priority_parameters_1[] = 
{
	{"parameter_string", string_e, 0, 0, -FLT_MAX, FLT_MAX},
	{"parameter_float" , float_e , 0, 0, -FLT_MAX, FLT_MAX},
	{"parameter_int"   , int_e   , 0, 0, -FLT_MAX, FLT_MAX},
};

parameter_s priority_parameters_2[] = 
{
	{"parameter_string", string_e, 2, 2, -FLT_MAX, FLT_MAX},	
};

parameter_s priority_parameters_3[] = 
{
	{"parameter_int"   , int_e   , 1, 1, -FLT_MAX, FLT_MAX},
	{"parameter_bool"  , bool_e  , 1, 1, -FLT_MAX, FLT_MAX},	
};

parameter_s priority_parameters_4[] = 
{
	{"parameter_char"  , char_e  , 1, 1, -FLT_MAX, FLT_MAX},	
};

int32_t num_priority_parameters[] = {5,3,1,2,1};
bool    is_superconfigs[]         = {false,true,false,false,false};

parameter_s *priority_parameters[] =
{
    priority_parameters_0,
    priority_parameters_1,
    priority_parameters_2,
    priority_parameters_3,
    priority_parameters_4
}; 

const int32_t num_defined_subconfigs = 5;
loader_config_s defined_subconfigs[5];

for (int32_t index = 0; index < num_defined_subconfigs; index++)
{
    defined_subconfigs[index] = subconfig;
    
    char *name; asprintf(&name, "nested_requirement_test_%i", index);
    defined_subconfigs[index].name = name;
    
    defined_subconfigs[index].is_superconfig         = is_superconfigs[index],

    defined_subconfigs[index].num_defined_parameters = num_priority_parameters[index];
    defined_subconfigs[index].defined_parameters     = priority_parameters[index];
}

loader_config_s default_subconfig[] = {subconfig};

loader_config_s loader_config = 
{
	.name                   = "simple_config_example",
	.name_necessity         = required_e,

	.is_superconfig         = true,
	.has_parameters         = false,
    .reorder                = false,
    .min                    = 1,
    .max                    = 1,
	.early_exit_index       = INT32_MAX,
    
	.num_defined_parameters = num_defined_parameters,
	.min_inputed_parameters = 0,
	.max_inputed_parameters = num_defined_parameters,
	.defined_parameters     = defined_parameters,
	
	.min_extra_parameters   = 0,
	.max_extra_parameters   = 0,
	.default_parameter      = default_parameter,
    
    .num_defined_subconfigs = num_defined_subconfigs,
	.min_num_subconfigs     = 5,
	.max_num_subconfigs     = 5,
	.defined_subconfigs     = defined_subconfigs,
    
    .min_extra_subconfigs   = 0,
	.max_extra_subconfigs   = 0,
	.default_subconfig      = default_subconfig,

	.struct_size            = sizeof(test_config_s)
};

#endif
