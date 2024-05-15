#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <inttypes.h>

#include "config.h"
#include "test.h"
#include "structures.h"
#include "console.h"

#include "single_config_test.h"	
#include "complex_test.h"	

typedef struct TestLoaderNull{
	
	/**
     * Structure to hold infomation about variable to be loaded from config.
     */
	
	char* file_name;
	bool  null_expected;
	
} loader_null_test_s;

bool checkLoaderReturnNull(
        const int32_t             verbosity,
        const char               *config_directory_name,
        const loader_config_s     loader_config,
        const loader_null_test_s  test_config
    ) {
    
    bool pass = true;
    
    const char *file_name     = test_config.file_name;
    const bool  null_expected = test_config.null_expected;
    
	test_config_s **test_results = NULL;
    loader_data_s   config_data;
    
    char *config_file_path;
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
    
    int64_t file_position[] = {0};
    
	test_results = 
		((test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 )
		);
	free(config_file_path);
    
    if (null_expected)
    {
        pass *=
            checkNULL(test_results, "Load Config", file_name);
    }
    else
    {
        pass *=
            checkNotNULL(test_results, "Load Config", file_name);
    }
    
    return pass;
}

void setTableCells(
	const test_config_s   known_results,
	const test_config_s  *test_results,
	const int32_t         num_parameters,
	      int32_t         cell_index,
	      uni_s          *cells
	) {
	
	cells[cell_index + 0] = (uni_s) {*((uni_u*)(&known_results.parameter_string)), string_e};
	cells[cell_index + 1] = (uni_s) {*((uni_u*)(&known_results.parameter_float )), float_e};
	cells[cell_index + 2] = (uni_s) {*((uni_u*)(&known_results.parameter_int   )), int_e};
	cells[cell_index + 3] = (uni_s) {*((uni_u*)(&known_results.parameter_bool  )), bool_e};
	cells[cell_index + 4].value.c = known_results.parameter_char;
	cells[cell_index + 4].type = char_e;

	cell_index += num_parameters;

	cells[cell_index + 0] = (uni_s) {*((uni_u*)(&test_results->parameter_string)), string_e};
	cells[cell_index + 1] = (uni_s) {*((uni_u*)(&test_results->parameter_float )), float_e};
	cells[cell_index + 2] = (uni_s) {*((uni_u*)(&test_results->parameter_int   )), int_e};
	cells[cell_index + 3] = (uni_s) {*((uni_u*)(&test_results->parameter_bool  )), bool_e};
	cells[cell_index + 4].value.c = test_results->parameter_char;  
	cells[cell_index + 4].type = char_e;
}


bool configTestCompare(
	const test_config_s   known_results,
	const test_config_s  *test_results
	) {
	
	bool pass = true;
	
	pass *= !strcmp(known_results.parameter_string, test_results->parameter_string);
	pass *= (known_results.parameter_float == test_results->parameter_float);
	pass *= (known_results.parameter_int   == test_results->parameter_int);
	pass *= (known_results.parameter_bool  == test_results->parameter_bool);
	pass *= (known_results.parameter_char  == test_results->parameter_char);
	
	return pass;
}

void plotConfigTestTable(
	const int32_t  num_configs,
	const int32_t  num_parameters,
	const char    *name,
	const uni_s   *cells
	) {
	table_column_s columns[] = {
		{"Known String" , 4, 1},
		{"Known Float"  , 4, 1},
		{"Known Int"    , 4, 1},
		{"Known Bool"   , 4, 1},
		{"Known Char"   , 4, 1},
		{"Loaded String", 4, 1},
		{"Loaded Float" , 4, 1},
		{"Loaded Int"   , 4, 1},
		{"Loaded Bool"  , 4, 1},
		{"Loaded Char"  , 4, 1}
	};

	table_s table = {
		num_configs,
		num_parameters * 2,
		name,
		cells,
		columns
	};

	printTable(table);
}

bool testSingleConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
	
	const char* file_name = "single_config_test.cfg";

	bool pass = true;
	
    loader_data_s config_data;
    
    char* config_file_path;
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
	
	#include "single_config_test.h"	
	
	const int32_t expected_num_parameters = 5;
	const int32_t expected_num_configs = 1;
	      int32_t num_configs = 0; 
    
    int64_t file_position[] = {0};

	test_config_s** all_test_results = 
		(test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 );
             
    num_configs = config_data.total_num_subconfigs_read;
    test_config_s *test_results = NULL;
    if (all_test_results != NULL) 
    {
        test_results = all_test_results[0];
    }
	
	if (!num_configs == (expected_num_configs - 1)) {
		pass = false; 
		fprintf(
			stderr, 
			"Number of configs found (%i) does not match expected (%i). \n",
			num_configs, expected_num_configs
		);
	}
	
	if (test_results == NULL) 
	{
		printf("Load Config returned NULL! \n");
		pass = false;
	} 
	else
	{
		test_config_s known_results = {
			.parameter_string = "Config Test",
			.parameter_float  = 1.1f,
			.parameter_int    = 1,
			.parameter_bool   = false,
			.parameter_char   = 'a'
		};
        
		uni_s *cells = malloc(sizeof(uni_s)* (size_t) ( expected_num_configs * expected_num_parameters * 2 ));

		const int32_t cell_index = 0; 
		setTableCells(
			known_results, 
			test_results, 
			expected_num_parameters, 
			cell_index, 
			cells
		);
		
		pass *= 
			configTestCompare(known_results, test_results);
		
		plotConfigTestTable(
			expected_num_configs, 
			expected_num_parameters,
			"Single Config Test",
			cells
		);
	}
	
	printTestResult(pass, "single config test.");
	
	free(config_file_path);

	return pass;
}

bool testVariableConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
	
	const char* file_name = "variable_config_test.cfg";

	bool pass = true;
	
    loader_data_s  config_data;
    
    char *config_file_path; 
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
	
	#include "variable_config_test.h"	
	
	const int32_t         expected_num_parameters = 5;
	const int32_t         expected_num_configs = 5;
		  int32_t         num_configs = 0;   
    
    int64_t file_position[] = {0};
    test_config_s** test_results = 
		((test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 )
		);
	num_configs = config_data.total_num_subconfigs_read;
    
	if (!num_configs == expected_num_configs) 
	{
		pass = false; 
		fprintf(
			stderr, 
			"Number of configs found (%i) does not match expected (%i). \n",
			num_configs, expected_num_configs
		);
	}
	
	if (test_results == NULL) 
	{
		printf("Load Config returned NULL! \n");
		pass = false;
	} 
	else
	{
		char *  strings[] = 
			{"", "Config Test 1", "Config Test 2", "Config Test 3", "Config Test 4", "Config Test 5"};
		const float   floats[] = 
			{1.0f, 0.0f, 1.2f, 1.3f, 1.4f};
		const int32_t ints[] =
			{0, 1, 2, 3, 4};
		const bool    bools[] =
			{false, true, false, false, false};
		const char    chars[] = 
			{'a', 'b', 'c', 'd', '\0'};

		uni_s *cells = malloc(sizeof(uni_s)* (size_t) ( expected_num_configs * expected_num_parameters * 2 ));
		for (int32_t index = 0; index < expected_num_configs; index++) {
		
			test_config_s known_results = {
				.parameter_string = strings[index],
				.parameter_float  = floats[index],
				.parameter_int    = ints[index],
				.parameter_bool   = bools[index],
				.parameter_char   = chars[index]
			};

			int32_t cell_index = index*expected_num_parameters*2;
			
			const test_config_s *test_result = test_results[index];
			
			setTableCells(
				known_results, 
				test_result, 
				expected_num_parameters, 
				cell_index, 
				cells
			);

			pass *= 
				configTestCompare(known_results, test_result);
		}

		plotConfigTestTable(
			expected_num_configs, 
			expected_num_parameters,
			"Variable Config Test",
			cells
		);
	}
	
	printTestResult(pass, "variable config test.");
	
	free(config_file_path);

	return pass;
}

bool testMultiConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
	
	const char* file_name = "multi_config_test.cfg";

	bool pass = true;
	
    loader_data_s  config_data;
    
    char          *config_file_path;
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
	
	#include "multi_config_test.h"	
	
	const int32_t expected_num_parameters = 5;
	const int32_t expected_num_configs = 3;
    
    int64_t file_position[] = {0};
	test_config_s** test_results = 
		((test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 )
		);
    const int32_t num_configs = config_data.total_num_subconfigs_read;
		
	if (!num_configs == expected_num_configs) {
		pass = false; 
		fprintf(
			stderr, 
			"Number of configs found (%i) does not match expected (%i). \n",
			num_configs, expected_num_configs
		);
	}
	
	if (test_results == NULL) 
	{
		printf("Load Config returned NULL! \n");
		pass = false;
	} 
	else
	{
		const char *  initial_string = "Config Test";
		const float   initial_float  = 1.0f;
		const int32_t initial_int    = 0;
		const bool    bool_array[]   = {false, true, false};
		const char    char_array[]   = {'a', 'b', 'c'};

		uni_s *cells = malloc(sizeof(uni_s)* (size_t) ( expected_num_configs * expected_num_parameters * 2 ));
		for (int32_t index = 0; index < expected_num_configs; index++) {

			char *parameter_string;
			asprintf(&parameter_string, "%s %i", initial_string, index);

			test_config_s known_results = {
				.parameter_string = parameter_string,
				.parameter_float  = initial_float + ((float)index * 0.1f),
				.parameter_int    = initial_int + index,
				.parameter_bool   = bool_array[index],
				.parameter_char   = char_array[index]
			};

			int32_t cell_index = index*expected_num_parameters*2;
			
			const test_config_s *test_result = test_results[index];
			
			setTableCells(
				known_results, 
				test_result, 
				expected_num_parameters, 
				cell_index, 
				cells
			);

			pass *= 
				configTestCompare(known_results, test_result);
		}

		plotConfigTestTable(
			expected_num_configs, 
			expected_num_parameters,
			"Multi Config Test",
			cells
		);
	}
	
	printTestResult(pass, "multi config test.");
	
	free(config_file_path);

	return pass;
}

bool testConfigOrder(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
	
	const char* file_name = "config_order_test.cfg";

	bool pass = true;
	
    loader_data_s config_data;
    
	char* config_file_path;
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
	
	#include "multi_config_test.h"	
	
	const int32_t expected_num_parameters = 5;
	const int32_t expected_num_configs = 3;
		  int32_t num_configs = 0; 
    
    int64_t file_position[] = {0};
	test_config_s** test_results = 
		((test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 )
		);
    num_configs = config_data.total_num_subconfigs_read;
		
	if (!num_configs == expected_num_configs) {
		pass = false; 
		fprintf(
			stderr, 
			"Number of configs found (%i) does not match expected (%i). \n",
			num_configs, expected_num_configs
		);
	}
	
	if (test_results == NULL) 
	{
		printf("Load Config returned NULL! \n");
		pass = false;
	} 
	else
	{
		const char   *initial_string = "Config Test";
		const float   initial_float  = 1.0f;
		const int32_t initial_int    = 0;
		const bool    bool_array[]   = {false, true, false};
		const char    char_array[]   = {'a', 'b', 'c'};

		uni_s *cells = malloc(sizeof(uni_s)* (size_t) ( expected_num_configs * expected_num_parameters * 2 ));
		for (int32_t index = 0; index < expected_num_configs; index++) {

			char *parameter_string;
			asprintf(&parameter_string, "%s %i", initial_string, index);

			test_config_s known_results = {
				.parameter_string = parameter_string,
				.parameter_float  = initial_float + ((float)index * 0.1f),
				.parameter_int    = initial_int + index,
				.parameter_bool   = bool_array[index],
				.parameter_char   = char_array[index]
			};

			int32_t cell_index = index*expected_num_parameters*2;
			
			const test_config_s *test_result = test_results[index];
			
			setTableCells(
				known_results, 
				test_result, 
				expected_num_parameters, 
				cell_index, 
				cells
			);

			pass *= 
				configTestCompare(known_results, test_result);
		}

		plotConfigTestTable(
			expected_num_configs, 
			expected_num_parameters,
			"Config Order Test",
			cells
		);
	}
	
	printTestResult(pass, " config order test.");
	
	free(config_file_path);

	return pass;
}

bool testReqirementConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
	
	bool pass = true;
    
	#include "requirement_test.h"	
    
    const int32_t num_tests = 5;
    loader_null_test_s test_configs[] = 
    {
        {"requirement_tests/requirement_test_0.cfg", false},
        {"requirement_tests/requirement_test_1.cfg", true},
        {"requirement_tests/requirement_test_2.cfg", true},
        {"requirement_tests/requirement_test_3.cfg", false},
        {"requirement_tests/requirement_test_4.cfg", true},
    };
    
    for (int32_t index = 0; index < num_tests; index++) 
    {
        pass *= checkLoaderReturnNull(
            verbosity,
            config_directory_name,
            loader_config,
            test_configs[index]
        );
    }
	
	printf("\n");
	printTestResult(pass, "Requirement test.");
	
	return pass;
}

bool testExtraParameterConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
	
	bool pass = true;
	
	char            *file_name = NULL;
	int32_t          num_configs = 0; 
	test_config_s  **test_results = NULL;
    loader_data_s    config_data;

	char *config_file_path;
	
	#include "extra_parameter_test.h"	
	
	file_name = "extra_parameter_tests/extra_parameter_test_0.cfg";
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
    
    int64_t file_position[] = {0};
	test_results = 
		((test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 )
		);
    num_configs = config_data.total_num_subconfigs_read;
	free(config_file_path);
	
	const int32_t num_extra_parameters = 5;
	dict_s *known_results = makeDictionary(100);
	
	char   *known_string = "Extra Parameter Test";
	float   known_float  = 1.1f;
	int32_t known_int    = 0;
	bool    known_bool   = true;
	char    known_char   = 'a';
			
	multi_s known_extra_parameters[] = 
	{
		MultiS( known_string, string_e, 1, NULL),
		MultiS(&known_float , float_e , 1, NULL),
		MultiS(&known_int   , int_e   , 1, NULL),
		MultiS(&known_bool  , bool_e  , 1, NULL),
		MultiS(&known_char  , char_e  , 1, NULL)
	};

	const char *extra_parameter_names[] = 
	{
		"extra_parameter_string",
		"extra_parameter_float",
		"extra_parameter_int",
		"extra_parameter_bool",
		"extra_parameter_char"
	};
	
	for (int32_t index = 0; index < num_extra_parameters; index++) 
	{	
		insertDictEntry(
			known_results, 
			known_extra_parameters[index], 
			extra_parameter_names[index]
		);
	}
	
	pass *=
        checkNotNULL(test_results, "Load Config", file_name);
	
    if ((num_configs >= 3))
    {
        for (int32_t index = 0; index < num_extra_parameters; index++) 
        {	
            dict_entry_s *known_result = findDictEntry(
                known_results, 
                extra_parameter_names[index]
            );
            dict_entry_s *read_result = findDictEntry(
                config_data.subconfigs[3].extra_parameters, 
                extra_parameter_names[index]
            );
            
            if (read_result == NULL) 
            {
                fprintf(stderr, "Warning! Cannot find extra parameter %s \n", extra_parameter_names[index]);
                pass *= false;
            }
            else if (known_result->data.type == read_result->data.type) 
            {
                pass *= comapareMultiS(known_result->data, read_result->data);
                printf("%s, %s, %s \n", MultiStoString(read_result->data), MultiStoString(known_result->data), extra_parameter_names[index]);
            } 
            else 
			{
				fprintf(
					stderr,
					"testExtraParameterConfig: \nWarning! Values do not match:"
					" %s, %s, %s. \n", 
					MultiStoString(read_result->data), 
					MultiStoString(known_result->data),
					extra_parameter_names[index]
				);
                pass *= false;
            }
        }
    }
    else 
    {
        printf("Extra parameters %s returned NULL! Unexpected behaviour.\n", file_name);
		pass *= false;
    }
        
    file_name = "extra_parameter_tests/extra_parameter_test_1.cfg";
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
    
    *file_position = 0;
	test_results = 
		((test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 )
		);
    num_configs = config_data.total_num_subconfigs_read;
        
    free(config_file_path);
	
	pass *=
        checkNotNULL(test_results, "Load Config", file_name);
        
    const int32_t num_tests = 2;
    loader_null_test_s test_configs[] = 
    {
        {"extra_parameter_tests/extra_parameter_test_2.cfg", true},
        {"extra_parameter_tests/extra_parameter_test_3.cfg", true}
    };
    
    for (int32_t index = 0; index < num_tests; index++) 
    {
        pass *= checkLoaderReturnNull(
            verbosity,
            config_directory_name,
            loader_config,
            test_configs[index]
        );
    }
	
	printf("\n");
	printTestResult(pass, "Extra parameter config test.");
	
	return pass;
}

bool testNameRequrimentConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
    
    bool pass = true;
	
	char           *file_name        = NULL;
	test_config_s **test_results     = NULL;
    loader_data_s   config_data;

	char* config_file_path;
	
    {
        #include "name_requirement_test_0.h"	
        
        const int32_t num_tests = 3;
        loader_null_test_s test_configs[] = 
        {
            {"name_requirement_tests/name_requirement_test_0.cfg", false},
            {"name_requirement_tests/name_requirement_test_1.cfg", false},
            {"name_requirement_tests/name_requirement_test_2.cfg", false}
        };

        for (int32_t index = 0; index < num_tests; index++) 
        {
            pass *= checkLoaderReturnNull(
                verbosity,
                config_directory_name,
                loader_config,
                test_configs[index]
            );
        }
    }
    {
        #include "name_requirement_test_1.h"	

        file_name = "name_requirement_tests/name_requirement_test_3.cfg";
        asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
        
        int64_t file_position[] = {0};
        test_results = 
            ((test_config_s**) 
                 readConfig(
                     verbosity,
                     config_file_path, 
                     loader_config,
                     &config_data,
                     file_position
                 )
            );

        free(config_file_path);

        pass *= 
            checkNULL(test_results, "Load config", file_name);
    }
    
    printf("\n");
	printTestResult(pass, "Name requirement config test.");
    
    return pass;
}

bool testConfigRequriment(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
    
    bool pass = true;
    
    #include "config_requrirement_test.h"	
    
    const int32_t num_tests = 5;
    loader_null_test_s test_configs[] = 
    {
        {"config_requirement_tests/config_requirement_test_0.cfg", false},
        {"config_requirement_tests/config_requirement_test_1.cfg", false},
        {"config_requirement_tests/config_requirement_test_2.cfg", false},
        {"config_requirement_tests/config_requirement_test_3.cfg", true},
        {"config_requirement_tests/config_requirement_test_4.cfg", true},
    };
    
    for (int32_t index = 0; index < num_tests; index++) 
    {
        pass *= checkLoaderReturnNull(
            verbosity,
            config_directory_name,
            loader_config,
            test_configs[index]
        );
    }
	
	printf("\n");
	printTestResult(pass, "Config Requirement config test.");
    
    return pass;
}

bool testNestedRequriments(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
    
    bool pass = true;
    
    #include "nested_requirement_test.h"	
    
    const int32_t num_tests = 5;
    loader_null_test_s test_configs[] = 
    {
        {"nested_requirement_tests/nested_requirement_test_0.cfg", false},
        {"nested_requirement_tests/nested_requirement_test_1.cfg", true },
        {"nested_requirement_tests/nested_requirement_test_2.cfg", true },
        {"nested_requirement_tests/nested_requirement_test_3.cfg", true },
        {"nested_requirement_tests/nested_requirement_test_4.cfg", true }
    };
    
    for (int32_t index = 0; index < num_tests; index++) 
    {
        pass *= checkLoaderReturnNull(
            verbosity,
            config_directory_name,
            loader_config,
            test_configs[index]
        );
    }
	
	printf("\n");
	printTestResult(pass, "Nested Requirement config test.");
    
    return pass;
}

bool testNestedConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
    
    bool pass = true;
    
    #include "nested_config_test.h"	
    
    const int32_t num_tests = 4;
    loader_null_test_s test_configs[] = 
    {
        {"nested_config_tests/nested_config_test_0.cfg", false},
        {"nested_config_tests/nested_config_test_1.cfg", true},
        {"nested_config_tests/nested_config_test_2.cfg", true},
        {"nested_config_tests/nested_config_test_3.cfg", true}
    };
    
    for (int32_t index = 0; index < num_tests; index++) 
    {
        pass *= checkLoaderReturnNull(
            verbosity,
            config_directory_name,
            loader_config,
            test_configs[index]
        );
    }
	
	printf("\n");
	printTestResult(pass, "Nested config test.");
    
    return pass;
}

bool testMultiTypeConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
    
    #include "multi_config_test.h"	
    
    char *file_name = "multi_type_test.cfg";

	bool pass = true;
	
    loader_data_s config_data;
    
    char* config_file_path;
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
	
	int32_t expected_num_parameters = 5;
	int32_t expected_num_configs = 1;
    int32_t num_configs = 0; 
    
    int64_t file_position[] = {0};
	
	loader_config.early_exit_index = 3;
	test_config_s** all_test_results = 
		(test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 );
	loader_config.early_exit_index = INT32_MAX;
    
	expected_num_parameters = 5;
	expected_num_configs = 3;
    
	all_test_results = 
		((test_config_s**) 
			 readConfig(
			     verbosity,
				 config_file_path, 
				 loader_config,
				 &config_data,
                 file_position
			 )
		);
    num_configs = config_data.total_num_subconfigs_read;
		
	if (!num_configs == expected_num_configs) {
		pass = false; 
		fprintf(
			stderr, 
			"Number of configs found (%i) does not match expected (%i). \n",
			num_configs, expected_num_configs
		);
	}
	
	if (all_test_results == NULL) 
	{
		printf("Load Config returned NULL! \n");
		pass = false;
	} 
	else
	{
		const char *  initial_string = "Config Test";
		const float   initial_float  = 1.0f;
		const int32_t initial_int    = 0;
		const bool    bool_array[]   = {false, true, false};
		const char    char_array[]   = {'a', 'b', 'c'};

		uni_s *cells = malloc(sizeof(uni_s)* (size_t) ( expected_num_configs * expected_num_parameters * 2 ));
		for (int32_t index = 0; index < expected_num_configs; index++) {

			char *parameter_string;
			asprintf(&parameter_string, "%s %i", initial_string, index);

			test_config_s known_results = {
				.parameter_string = parameter_string,
				.parameter_float  = initial_float + ((float)index * 0.1f),
				.parameter_int    = initial_int + index,
				.parameter_bool   = bool_array[index],
				.parameter_char   = char_array[index]
			};

			int32_t cell_index = index*expected_num_parameters*2;
			
			const test_config_s *test_result = all_test_results[index];
			
			setTableCells(
				known_results, 
				test_result, 
				expected_num_parameters, 
				cell_index, 
				cells
			);

			pass *= 
				configTestCompare(known_results, test_result);
		}

		plotConfigTestTable(
			expected_num_configs, 
			expected_num_parameters,
			"Multi Type Test",
			cells
		);
	}
	
	printTestResult(pass, "multi type test.");
	
	free(config_file_path);

	return pass;
}

bool testComplexConfig(
	const int32_t  verbosity,
	const char    *config_directory_name
	) {
    
    bool pass = true;
    
    #include "complex_test.h"	
    
    const int32_t num_tests = 1;
    loader_null_test_s test_configs[] = 
    {
        {"complex_test.cfg", false}
    };
	
	loader_null_test_s test_config = test_configs[0];
	
	const char *file_name     = test_config.file_name;
    
    loader_data_s   config_data;
    
    char *config_file_path;
	asprintf(&config_file_path, "./%s/%s", config_directory_name, file_name);
    
    int64_t file_position[] = {0};
	 readConfig(
		 verbosity,
		 config_file_path, 
		 loader_config,
		 &config_data,
		 file_position
	 );
	free(config_file_path);
	
	network_config_s network_config = 
		*((network_config_s*) config_data.subconfigs[0].structure);
	
	printf("%f \n", network_config.speed_of_light);
		
	loader_data_s detector_data = config_data.subconfigs[0].subconfigs[0];
	const int32_t num_detectors = detector_data.total_num_subconfigs_read;
	
	detector_s* detectors = malloc(sizeof(detector_s) * (size_t)num_detectors);  
	for (int32_t index = 0; index < num_detectors; index++)
	{
		detectors[index] = 
			*((detector_s*) detector_data.subconfigs[index].structure);
		printf(
			"%i, %i, %i \n", 
			detectors[index].latitude[0], 
			detectors[index].latitude[1], 
			detectors[index].latitude[2]
		);
	}
	
    for (int32_t index = 0; index < num_tests; index++) 
    {
        pass *= checkLoaderReturnNull(
            verbosity,
            config_directory_name,
            loader_config,
            test_configs[index]
        );
    }
	
	printf("\n");
	printTestResult(pass, "Complex config test.");
    
    return pass;
}

int main() {
	
	const int32_t verbosity = 3;
	
	//parameters:
	const char *config_directory_name = "configs";	
		 
	bool pass = true;
	
	pass *= 
		testSingleConfig(
			verbosity,
			config_directory_name
		);
	
	pass *= 
		testMultiConfig(
			verbosity,
			config_directory_name
		);
	
	pass *= 
		testVariableConfig(
			verbosity,
			config_directory_name
		);
	
	pass *= 
		testReqirementConfig(
			verbosity,
			config_directory_name
		);
	
	pass *=
		testExtraParameterConfig(
			verbosity,
			config_directory_name
		);
        
    pass *= 
        testNameRequrimentConfig(
            verbosity,
            config_directory_name
        ); 
        
    pass *= testConfigRequriment(
            verbosity,
            config_directory_name
        );
    
    pass *=  
        testNestedRequriments(
            verbosity,
            config_directory_name
        ); 
        
    pass *=  
        testConfigOrder(
            verbosity,
            config_directory_name
        ); 
        
    pass *=  
        testNestedConfig(
            verbosity,
            config_directory_name
        ); 
    
    pass *=  
        testMultiTypeConfig(
            verbosity,
            config_directory_name
        ); 
	
	pass *=  
		testComplexConfig(
			verbosity,
			config_directory_name
		); 
	
	printTestResult(pass, "all tests.");
	
	return 0;
}