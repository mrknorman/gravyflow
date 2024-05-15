#ifndef IO_TEXT_H
#define IO_TEXT_H

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <inttypes.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>
#include <unistd.h>
#include <ctype.h>
#include <limits.h>
#include <immintrin.h>
#include <complex.h> 
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

#include "io_tools/strings.h"

typedef struct split_path {
	
	const char *full;
	const char *directory;
	const char *base;
} path_s;

path_s newPath(
    const char *full_path
    ) {
    
	
	char *full_path_copy = strdup(full_path);
    const size_t length = strlen(full_path_copy);
	
	if (full_path_copy[length - 1] == '/') 
	{
		full_path_copy[length - 1] = '\0';
	}
    
    char *base_name = malloc(sizeof(char) * length);
    char *dir_name  = malloc(sizeof(char) * length);
    
    bool in_base = true;
    
    int32_t base_index = 0, dir_index = 0;
    for (size_t index = 0; index < length; index++) 
	{
        const char letter = full_path_copy[length - 1 - index];

		if ((letter == '/') && in_base) 
		{
			in_base = false;
			continue;
		}
                        
        if (in_base) 
		{
            base_name[base_index] = letter;
            base_index++;
        } 
		else
		{
            dir_name[dir_index] = letter;
            dir_index++;
        }
        
        base_name[base_index] = '\0';
        dir_name [dir_index ] = '\0';
    }
    
    const path_s path = 
        (const path_s) {full_path_copy, strrev(dir_name), strrev(base_name)};
    
    free(dir_name); free(base_name);
    
	return path;
}

void printPath(
    const path_s path
    ) {
	
	printf("Full path: \"%s\". \n", path.full     );
	printf("Directory: \"%s\". \n", path.directory);
	printf("Base: \"%s\". \n", path.base     );
}

int32_t mkpath(
    const char   *directory, 
    const mode_t  mode
    ) {
	
	struct stat sb;
    
    if (!directory) 
	{
        errno = EINVAL;
        return 1;
    }
    else if (!stat(directory, &sb)) 
	{
        return 0;
    }
	
	path_s new_path = newPath(directory);
	
	mkpath(new_path.directory, mode);
	
	return mkdir(new_path.full, mode);
}

bool checkFileExists(
    const int32_t   verbosity, 
    const char    * file_name
    ) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Checks if file exists.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool exists = 1;

	FILE* file = fopen(file_name, "r");
	
	if (file) 
	{
	    /* File exists. */

	    fclose(file);

	} 
	else if (ENOENT == errno) 
	{
		/* Directory does not exist. */

		exists = 0;
        
        if ( verbosity >= 2 ) 
        {     
            fprintf(
                stderr, 
                "checkFileExists: \nWarning! File \"%s\" does not exist.\n", 
                file_name
            );
        }
	} 
	else 
	{
		/* opendir() failed for some other reason. */

		exists = 0;
        
        if ( verbosity >= 1 ) 
		{
            fprintf(
                stderr, 
                "checkFileExists: \n Failed to open file, \"%s\", for unknown"
                " reason.\n", 
                file_name
            );
        }
	}

	return exists;
}

bool checkOpenFile(
    const int32_t   verbosity, 
    const char     *file_name, 
    const char     *mode, 
          FILE    **ret_file
    ) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Checks if file exists, and returns open file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool exists = 1;

	FILE *file = fopen(file_name, mode);
	
	if (file) 
	{
	    /* File exists. */

	    *ret_file = file;

	} 
	else if (ENOENT == errno) 
	{
		/* Directory does not exist. */

		exists = 0;
        
        if ( verbosity >= 1 ) 
		{
            fprintf(
                stderr, 
                "checkOpenFile: \nWarning! File \"%s\" does not exist.\n", 
                file_name
            );  
        }
	} 
	else 
	{
		/* opendir() failed for some other reason. */

		exists = 0;
        
        if ( verbosity >= 1 ) 
		{
            fprintf(
                stderr, 
                "checkOpenFile: \nFailed to open file, \"%s\", for unknown"
                " reason.\n", 
                file_name
            );
        }
	}

	return exists;
}

bool createFile(
    const int32_t  verbosity, 
    const char    *file_name, 
    const char    *mode
    ) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Atempts to create file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
		
	bool success = false;

	if ( fopen(file_name, mode) == NULL) 
	{
        if ( verbosity >= 1 ) {
            fprintf(
                stderr, 
                "createFile: \nError creating file \"%s\". Returning.\n", 
                file_name
            );
        }

		success = true;
	} 
	else 
	{
		success = true;
	}

	return success;
}

bool createOpenFile(
    const int32_t    verbosity, 
    const char     * file_name, 
    const char     * mode, 
          FILE    ** ret_file
    ){

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Atempts to create file, and returns open file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool success = false;

	FILE* file = fopen(file_name, mode);

	if ( file == NULL ) 
	{
        if (verbosity >= 1) 
		{
            fprintf(
                stderr, 
                "createOpenFile: \nWarning! Error creating file \"%s\"."
                " Returning.\n",
                file_name
            );
        }
 
		success = true;
	} 
	else 
	{
		*ret_file = file;
		success   = true;
	}

	return success;
}

bool checkCreateFile(
    const int32_t   verbosity, 
    const char    * file_name, 
    const char    * mode
    ) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Checks if file exists, and creates file if not.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool success = false;
    
	if ( !checkFileExists(verbosity, file_name) ) 
	{
        if (verbosity >= 2) 
		{
            fprintf(
                stderr,
                "checkCreateFile: \n"
                " File \"%s\" does not exist. Attemping to create.\n", 
                file_name 
            );
        }

		if ( createFile(verbosity, file_name, mode) ) 
		{
			success = true;
		} else 
		{
			success = false;
		}
	} else 
	{
		success = true;
	}

	return success;
}

bool checkDirectoryExists(
	const int32_t  verbosity, 
	const char    *directory_name
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Checks if directory exists.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	bool exists = true;

	DIR* directory = opendir(directory_name);
		
	if (directory) 
	{
	    /* Directory exists. */
	    closedir(directory);
	} 
	else if (ENOENT == errno) 
	{
		exists = false;
	    /* Directory does not exist. */
	} 
	else 
	{
		exists = false;
        
        if (verbosity >= 1) 
		{
            fprintf(
                stderr, 
                "checkDirectoryExists: \n Failed to open directory \"%s\", for"
                "unknown reason.\n", 
                directory_name
            );
        }
	    /* opendir() failed for some other reason. */
	}
	
	return exists;
}

bool checkOpenDirectory(
	const char  *directory_name, 
	      DIR  **ret_directory
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Checks if directory exists, and returns open file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool exists = true;

	DIR* directory = opendir(directory_name);
	
	if (directory) 
	{
	    /* Directory exists. */
		
	    *ret_directory = directory;
	} 
	else if (ENOENT == errno) 
	{
		/* Directory does not exist. */
		
		exists = false;
		fprintf(
            stderr, 
            "Warning! Cannot find directory \"%s\".\n", 
            directory_name
        );
	} 
	else 
	{
		/* opendir() failed for some other reason. */

		exists = false;
		fprintf(
            stderr, 
            "Warning! Failed to open directory \"%s\", for unknown reason.\n", 
            directory_name
        );
	}

	return exists;
}

bool createDirectory(
	const char *directory_name
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Attempts to create directory.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool success = false;
	
	if ( mkpath(directory_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 ) 
	{
		fprintf(
            stderr, 
            "Error creating directory \"%s\". Returning. \n", 
            directory_name
        );

		success = false;
	}
	else 
	{
		success = true;
	}

	return success;
}

bool createOpenDirectory(
	const char* directory_name, 
	      DIR** ret_directory
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Attempts to create directory, and returns open directory.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool success = false;

	if ( mkpath(directory_name, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0 ) 
	{
		fprintf(
            stderr, 
            "Error creating directory \"%s\". Returning. \n", 
            directory_name
        );

		success = false;
	} 
	else 
	{
 		checkOpenDirectory(directory_name, ret_directory);

		success = true;
	}

	return success;
}

bool checkCreateDirectory(
	const int32_t  verbosity,
	const char    *directory_name
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Checks if directory exists, and creates it if not.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	bool success = false;
	
	if ( !checkDirectoryExists(1, directory_name) ) 
	{
		if (verbosity >= 2) 
		{
			printf(
                "Directory \"%s\" does not exist. Attemping to create.\n", 
                directory_name
            );
		}
				
		if ( createDirectory(directory_name))
		{
			success = true;
		} else 
		{
			success = false;
		}
		
	} else 
	{
		success = true;
	}
	
	return success;
}

void get_argf(
	      float *val, 
	const char  *delimeter
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads float from file.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	char* t = strtok(NULL, delimeter);
	*val = strtof(t, NULL);
}

int32_t countLinesInTextFile(
	FILE* file
	) {

	int32_t num_lines = 0;

	fpos_t position; 
	fgetpos(file, &position);

	char*   buffer = NULL; size_t size = 0;
	while (getline(&buffer, &size, file) != EOF) 
	{
		num_lines++;
	}

	fsetpos(file, &position);

	return num_lines;

}

bool readFileDouble(
	const char     *file_name, 
	const int32_t   mode, 
	const char     *delimeter, 
	const int32_t   num_cols, 
	const int32_t   start_line, 
	const int32_t   start_col, 
		  float   **data_ret, 
		  int32_t  *lines_read
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Reads delimited float data from input file.
	//
	// Mode 0: Column index first. 
	// Mode 1: Row index first.
	//
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
		
	bool     success   = false;

	int32_t  init_num_lines = 100;
	int32_t  curr_num_lines = init_num_lines;

	char    *buffer = NULL; 
	size_t   size = 0;

	int32_t  line_index = 0;
		
	FILE* file = NULL;
	if( checkOpenFile(3, file_name, "r", &file) ) 
	{
		int32_t  num_lines    = countLinesInTextFile(file);
		size_t   num_elements = (size_t) (num_lines * num_cols);
		
		float *data = malloc(sizeof(float) * num_elements);

		//Skips to start line:
		for (int32_t line_index = 0; line_index < start_line; ++line_index) 
		{ 
			size = (size_t) getline(&buffer, &size, file); 
		}

		switch(mode) 
		{
			case 0: //Column index first:

				while (getline(&buffer, &size, file) != EOF)
				{	
					if(line_index >= curr_num_lines)
					{
						size_t num_elements = 
                            (size_t) ((curr_num_lines + init_num_lines) 
                                    * num_cols);
						data = realloc(data, sizeof(float) * num_elements);
						curr_num_lines += init_num_lines;
					}

					char* first = strtok(buffer, delimeter);
					
					data[line_index*num_cols + 0] = strtof(first, NULL);

					//Skips to start column
					float empty = 0;
					for (
                        int32_t col_index = 1; 
                        col_index < start_col; 
                        ++col_index
                    ) { 
                        get_argf(&empty, delimeter); 
                    }

					if (start_col == 0) 
                    { 
                        for (
                            int32_t col_index = 1; 
                            col_index < num_cols; 
                            ++col_index
                        ) { 
                            get_argf(
                                &data[line_index*num_cols + col_index], 
                                delimeter
                            ); 
                        }
                    }
					else
                    { 
                        for (
                            int32_t col_index = 0; 
                            col_index < num_cols; 
                            ++col_index
                        ) { 
                            get_argf(
                                &data[line_index*num_cols + col_index], 
                                delimeter); 
                        } 
                    } 

					line_index++;
				}

			break;

			case 1: //Row index first:
				
				while (getline(&buffer, &size, file) != EOF)
				{
					char* first = strtok(buffer, delimeter);

					data[line_index*num_cols + 0] = strtof(first, NULL);

					//Skips to start column
					float empty = 0;
					for (
                        int32_t col_index = 1; 
                        col_index < start_col; 
                        ++col_index
                    ) { 
                        get_argf(&empty, delimeter); 
                    }

					if (start_col == 0) 
                    { 
                        for (
                            int32_t col_index = 1; 
                            col_index < num_cols; 
                            ++col_index
                        ) { 
                            get_argf(
                                &data[col_index*num_lines + line_index], 
                                delimeter
                            ); 
                        } 
                    }
					else                
                    { 
                        for (
                            int32_t col_index = 0; 
                            col_index < num_cols; 
                            ++col_index
                        ) { 
                            get_argf(
                                &data[col_index*num_lines + line_index], 
                                delimeter
                            );
                        } 
                    } 

					line_index++;
				}

			break;
		}

		*lines_read = line_index;
		*data_ret = data;

		success = true;
		fclose(file);
	} 
	else 
	{
		success = false;
	}
	
	return success;
}

bool writeFileDouble(
	const float   *data          , 
	const char    *file_name     , 
	const int32_t  mode          , 
	const int32_t  decimal_places, 
	const char    *delimeter     , 
	const int32_t  num_cols      , 
	const int32_t  num_lines     , 
	const int32_t  start_line    , 
	const int32_t  start_col
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Writes delimited float data to output file.
	//
	//
	// Mode 0: Column index first. 
	// Mode 1: Row index first.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	bool success = false;

	FILE* file = NULL;
	if( createOpenFile(1, file_name, "w+", &file) )
	{
		//Skips to start line:
		for (int32_t line_index = 0; line_index < start_line; ++line_index) 
        { 
            fprintf(file, "\n"); 
        }

		switch(mode) 
		{
			case 0: //Column index first:

				for (
                    int32_t line_index = 0; 
                    line_index < num_lines; 
                    line_index++
                ) {
					//Skips to start column
					for (
                        int32_t col_index = 1; 
                        col_index < start_col; 
                        ++col_index
                    ) { 
                        fprintf(file, "%s", delimeter); 
                    }

					for (
                        int32_t col_index = 0; 
                        col_index < num_cols ; 
                        ++col_index
                    ) { 
                        fprintf(
                            file, 
                            "%*e%s", 
                            decimal_places, 
                            data[line_index*num_cols + col_index], 
                            delimeter
                        ); 
                    }

					fprintf(file, "\n");
				}

			break;

			case 1: //Row index first:

				for (
                    int32_t line_index = 0; 
                    line_index < num_lines; 
                    line_index++
                ) {
					//Skips to start column
					for (
                        int32_t col_index = 1; 
                        col_index < start_col; 
                        ++col_index
                    ) { 
                        fprintf(file, "%s", delimeter); 
                    }

					for (
                        int32_t col_index = 0; 
                        col_index < num_cols ; 
                        ++col_index
                    ) { 
                        fprintf(
                            file, 
                            "%f%s", 
                            data[col_index*num_lines + line_index], 
                            delimeter
                        ); 
                    }

					fprintf(file, "\n");
				}

			break;
		}

		success = true;
		
		if (file != NULL)
		{
			fclose(file);
		}
	} 
	else 
	{
		success = false;
		fprintf(stderr, "Warning! Failed to create file \"%s\"!\n", file_name);
	}

	return success;
}

bool readDirectoryContents(
	const char      *directory_name, 
	      char    ***strings_ret, 
	      int32_t   * num_strings
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Gets the names of all files within dir_name, and stores them in 
    // stings_ret. Number of files found is stored in num_strings.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	bool    success = false;

	int32_t   init_num_strings = 100;
	int32_t   curr_num_strings = init_num_strings;
	char    **strings = malloc(sizeof(char*)* (size_t) init_num_strings);

	int32_t name_index = 0;

	DIR* directory;
	if ( checkOpenDirectory(directory_name, &directory) ) 
	{
		struct dirent* ent;
		while((ent = readdir(directory)) != NULL)
		{

			if(name_index >= curr_num_strings)
			{
				strings = 
                    realloc(
                        strings, 
                        sizeof(char*)* (size_t) curr_num_strings*2
                    );
				curr_num_strings *= 2;
			}

			size_t name_sz = strlen(ent->d_name)+1;
			strings[name_index] = malloc(sizeof(char)*name_sz);

			strcpy(strings[name_index], ent->d_name);

			name_index++;
		}

		success = true;
		closedir(directory);

	} 
	else 
	{
		fprintf(
            stderr, 
            "Warning! Failed to open directory \"%s\"!\n", 
            directory_name
        );
		success = false;
	}

	if (name_index == 0)
	{
		*strings = NULL;
	}

	*strings_ret = strings;
	*num_strings = name_index;

	return success;
}

void filterbyExtension(
	const char      *ext, 
	      char     **input, 
	const int32_t    num_input, 
	      char    ***filt_ret, 
	      int32_t   *num_filt
	) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Filters a list of filenames, imput, separating the files with extension, 
    // ext, into a new list filt_ret. Number of items retained after filtration 
    // being given by num_filt. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	const int32_t init_num_filt = 100;

	if (num_input == 0)
	{
		*filt_ret = NULL;
		*num_filt = 0;
		return;
	}

	int32_t filt_index = 0;
	int32_t check_ext  = 0;
	int32_t curr_num_filt = init_num_filt;

	char  *dot_pos = NULL;
	char **filt    = malloc(sizeof(char*)*(size_t)curr_num_filt);

	for (int32_t input_index = 0; input_index < num_input; ++input_index)
	{
		dot_pos = strrchr(input[input_index], '.');
		if (!dot_pos || dot_pos == input[input_index]) { continue; }
		else 
		{
			check_ext = strcmp(dot_pos+1, ext);
			if (check_ext == 0) 
			{	
				if(filt_index >= curr_num_filt)
				{
					filt = 
                        realloc(filt, sizeof(char*)*(size_t) curr_num_filt*2);
					curr_num_filt *= 2;
				}

				size_t filt_sz = strlen(input[input_index])+1;
				filt[filt_index] = malloc(sizeof(char)* (size_t) filt_sz);

				strcpy(filt[filt_index],input[input_index]);

				filt_index++;
			}
		}
	}

	if (num_filt == 0)
	{
		*filt_ret = NULL;
		*num_filt = 0;

		return;
	}

	*filt_ret = filt;
	*num_filt = filt_index;
}

void filterbyPrefix(
	const char      *prefix, 
	      char     **input, 
	const int32_t    num_input, 
	      char    ***filt_ret, 
	      int32_t   *num_filt
	) {
	
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Filters a list of filenames, imput, separating the files with a prefix, 
    // prefix, into a new list filt_ret. Number of items retained after 
    // filtration being given by num_filt. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

	if (num_input == 0)
	{
		*filt_ret = NULL;
		*num_filt = 0;
		return;
	}

	int32_t filt_index = 0;
	size_t  prefix_length = strlen(prefix);

	int32_t init_num_filt = 100;
	int32_t curr_num_filt = init_num_filt;
	char** filt       = malloc(sizeof(char*)* (size_t) init_num_filt);

	for (int32_t input_index = 0; input_index < num_input; ++input_index)
	{
		char test_prefix[prefix_length];
		sprintf(test_prefix, "%.*s", (int) prefix_length, input[input_index]);

		if (!strcmp(test_prefix, prefix))
		{
			if(filt_index >= curr_num_filt)
			{
				filt = 
                    realloc(filt, sizeof(char*) * (size_t) curr_num_filt * 2);
				curr_num_filt *= 2;
			}

			size_t filt_sz = strlen(input[input_index])+1;
			filt[filt_index] = malloc(sizeof(char)* (size_t) filt_sz);

			strcpy(filt[filt_index],input[input_index]);

			filt_index++;
		}
	}

	if (num_filt == 0)
	{
		*filt_ret = NULL;
		*num_filt = 0;

		return;
	}

	*filt_ret = filt;
	*num_filt = filt_index;
}

bool writeStringToFile(
	const char    *file_name, 
	const char    *string, 
	const int32_t  verbosity
	) {
    
    int32_t return_value = 0;
    
    FILE* file;    
    if ( checkOpenFile(verbosity, file_name, "w+", &file) ) 
	{
        fputs(string, file);
        
        fclose(file);
        
        return_value = 1;
    } 
	else 
	{
      return_value = 0;
    
      if (verbosity > 0) 
	  {
          printf("Warning! Failed to open or create the file: %s\n", file_name);
      }
    }
    
    return return_value;
}

#endif