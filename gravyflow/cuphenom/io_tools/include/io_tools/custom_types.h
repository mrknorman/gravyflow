#ifndef CUSTOM_TYPES_H
#define CUSTOM_TYPES_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <ctype.h>

#include <stdbool.h>
#include <inttypes.h>

typedef enum Type {
	
	/**
     * Enum to represent type.
     */
	
	none_e,
	bool_e,
	bool_array_e,
	int_e, 
	int_array_e,
	int_jagged_e,
	float_e, 
	float_array_e, 
	float_jagged_e, 
	char_e,
	char_array_e,
	string_e, 
	string_array_e
} type_e;

char *typeToString(const type_e type);

typedef union UniversalU {
	
	/**
     * Union to store commonly used types.
     */
	
	bool    b;
	int32_t i; 
	float   f;
	char    c;
	char  * s;
} uni_u;

typedef struct UniversalS {
	
	/**
     * Structure to store commonly used types and their type.
     */
	  	
	uni_u  value;
	type_e type;
	
} uni_s;

uni_s UniS(
	void    *value  ,
	type_e   type   
) {

	uni_s return_union;
	if (type == string_e) 
	{
		return_union = (uni_s) {*((uni_u*)(&value)), type};
	}
	else 
	{
		return_union = (uni_s) {*((uni_u*)(value)), type};
	}
	 
	return return_union;
}

typedef union UniversalMultiU {
	
	/**
     * Union to store commonly used types and array pointers.
     */
	
	bool      b;
	bool     *bb;
	
	int32_t   i; 
	int32_t  *ii;
	int32_t **iii;
	
	float     f;
	float    *ff;
	float   **fff;
	
	char      c;
	char     *cc;
	
	char     *s;
	char    **ss;
	
} multi_u;

typedef struct UniversalMultiS {
	
	/**
     * Structure to store commonly used types and array pointers, 
	 * as well as their types and lengths.
     */

	multi_u  value;
	type_e   type;
	int32_t  length;
	int32_t *lengths;	
} multi_s;

multi_s MultiS(
	void    *value  ,
	type_e   type   ,
	int32_t  length ,
	int32_t *lengths	
) {
	
	multi_s data;
	if (type == string_e) 
	{
		data = (multi_s) {*((multi_u*)(&value)), type, length, lengths};
	}
	else 
	{
		data = (multi_s) {*((multi_u*)(value)), type, length, lengths};
	}
	
	return data;
}

typedef struct BoolArray {
	int32_t  num_elements;
	bool    *elements;
} bool_array_s;

typedef struct IntArray {
	int32_t  num_elements;
	int32_t *elements;
} int_array_s;

typedef struct FloatArray {
	int32_t  num_elements;
	float   *elements;
} float_array_s;

typedef struct CharArray {
	int32_t  num_elements;
	char    *elements;
} char_array_s;

typedef struct StringArray {
	int32_t   num_elements;
	char    **elements;
} string_array_s;

typedef union multiArray{
	bool_array_s   bb;
	int_array_s    ii;
	float_array_s  ff;
	char_array_s   cc;
	string_array_s ss;
} array_u;

typedef struct multiArrayS{
	array_u data;
	type_e  type;
} array_s;

array_s ArrayS(
	int32_t  verbosity,
	void    *elements,
	int32_t  num_elements,
	type_e   type
	) {
	
	array_s array;
	
	array.type              = type;
	
	switch (type) 
	{
		case(bool_array_e): 
			array.data.bb.elements     = (bool*   ) elements;
			array.data.bb.num_elements =            num_elements;
		break;
		
		case(int_array_e ): 
			array.data.ii.elements     = (int32_t*) elements;
			array.data.ii.num_elements =            num_elements;
		break;
		
		case(float_array_e ): 
			array.data.ff.elements     = (float*  ) elements;
			array.data.ff.num_elements =            num_elements;
		break;
		
		case(char_array_e):
			array.data.cc.elements     = (char*  ) elements;
			array.data.cc.num_elements =           num_elements;
		break;
		
		case(string_array_e):
			array.data.ss.elements     = (char** ) elements;
			array.data.ss.num_elements =           num_elements;
		break;
		
		default:
			if (verbosity > 1)
			{
				fprintf(
					stderr,
					"ArrayS: \nWarning! Type %s not supported when string from"
					"array. \n",
					typeToString(type)
				);
			}
		break;
	}
	
	return array;
}

size_t getSizeOfType(
	const type_e type
	) {
	
	/**
     * Return the size, in bytes, of entered type enum.
     * @param 
     *     const type_e type: type enum to return size of.
     * @see
     * @return size_t size: size of type.
     */
	
	size_t size = 0;
	switch(type) 
	{	
		case(none_e        ): size = 0                ; break;
		case(bool_e        ): size = sizeof(bool     ); break;
		case(bool_array_e  ): size = sizeof(bool*    ); break;
		
		case(int_e         ): size = sizeof(int32_t  ); break;
		case(int_array_e   ): size = sizeof(int32_t* ); break;
		case(int_jagged_e  ): size = sizeof(int32_t**); break;			
			
		case(float_e       ): size = sizeof(float    ); break;
		case(float_array_e ): size = sizeof(float*   ); break;
		case(float_jagged_e): size = sizeof(float**  ); break;		

		case(char_e        ): size = sizeof(char     ); break;
		case(char_array_e  ): size = sizeof(char*    ); break;
		
		case(string_e      ): size = sizeof(char*    ); break;		
		case(string_array_e): size = sizeof(char**   ); break;
			
		default:
			
			fprintf(
				stderr, 
				"getSizeOfType: \nWarning! Type \"%s\" not recognised! \n", 
				typeToString(type)
			);
		break;
	}
	
	return size;
}

type_e getBaseType(
	const type_e type
	) {
	
	type_e base_type = none_e;
	switch(type) 
	{	
		case(none_e        ): base_type = none_e           ; break;
		
		case(bool_e        ): base_type = bool_e  ; break;
		case(bool_array_e  ): base_type = bool_e  ; break;
		
		case(int_e         ): base_type = int_e  ; break;
		case(int_array_e   ): base_type = int_e  ; break;
		case(int_jagged_e  ): base_type = int_e  ; break;			
			
		case(float_e       ): base_type = float_e ; break;
		case(float_array_e ): base_type = float_e ; break;
		case(float_jagged_e): base_type = float_e ; break;		

		case(char_e        ): base_type = char_e  ; break;
		case(char_array_e  ): base_type = char_e  ; break;
		
		case(string_e      ): base_type = string_e; break;		
		case(string_array_e): base_type = string_e; break;
			
		default:
			
			fprintf(
				stderr, 
				"getSizeOfType: \nWarning! Type \"%s\" not recognised! \n", 
				typeToString(type)
			);
			
			base_type = none_e;
		break;
	}
	
	return base_type;
}

char *typeToString(
	const type_e type
	) {
	
	/**
     * Convert type value to string.
     * @param 
     *     const type type: type_e enum to return string name.
     * @see
     * @return char *string: name of type as string.
     */
	
	char *string = NULL;
	switch(type) 
	{	
		case(none_e        ): string = "none"              ; break;
		case(bool_e        ): string = "bool"              ; break;
		case(bool_array_e  ): string = "bool array"        ; break;

		case(int_e         ): string = "int"               ; break;
		case(int_array_e   ): string = "int array"         ; break;
		case(int_jagged_e  ): string = "jagged int array"  ; break;			
			
		case(float_e       ): string = "float"             ; break;
		case(float_array_e ): string = "float array"       ; break;
		case(float_jagged_e): string = "jagged float array"; break;		

		case(char_e        ): string = "char"              ; break;
		case(char_array_e  ): string = "char array"        ; break;
		
		case(string_e      ): string = "string"            ; break;		
		case(string_array_e): string = "string array"      ; break;
			
		default:
			
			fprintf(
				stderr,
				"typeToString: \nWarning! Type \"%i\" not recognised! \n", 
				type
			);
		break;
	}
	
	return string;
}

bool comapareMultiS(
	const multi_s data_a,
	const multi_s data_b
	) {
	
	/**
     * Convert type value to string.
     * @param 
     *     const type type: type_e enum to return string name.
     * @see
     * @return char *string: name of type as string.
	 */
	
	type_e type = data_a.type;
	
	bool pass = false;
	switch(type) 
	{	
		case(none_e        ): pass = false; break;
		case(bool_e        ): pass = (data_a.value.b == data_b.value.b); break;
		case(bool_array_e  ): pass = false; break;		
		
		case(int_e         ): pass = (data_a.value.b == data_b.value.b); break;
		case(int_array_e   ): pass = false; break;
		case(int_jagged_e  ): pass = false; break;			
			
		case(float_e       ): pass = (data_a.value.b == data_b.value.b); break;
		case(float_array_e ): pass = false; break;
		case(float_jagged_e): pass = false; break;		

		case(char_e        ): pass = (data_a.value.b == data_b.value.b); break;
		case(char_array_e  ): pass = false; break;		

		case(string_e      ): pass = 
            (bool) !strcmp(data_a.value.s, data_b.value.s); break;		
		case(string_array_e): pass = false; break;
			
		default:
			
			fprintf(
				stderr, 
				"comapareMultiS: \nWarning! Type \"%s\" not recognised! \n",
				typeToString(type)
			);
		break;
	}
	
	return pass;
}

bool stringToBool(
	const int32_t  verbosity,
	const char    *string
	) {
	
	bool    value     = false;
	
	if isdigit(string[0]) 
	{
		int32_t bool_value = atoi(string);

		if ((bool_value > 1) || (bool_value < 0)) 
		{
			if (verbosity > 1) 
			{
				fprintf(
					stderr, 
					"stringToBool: \nWarning! Bool value %s not recognised! \n", 
					string
				);
			}
			value = false;
		} 
		else 
		{
			value = (bool) atoi(string); 
		}
	} 
	else if (!strcmp(string, "true")) 
	{
		value = true;
	} 
	else if (!strcmp(string, "false")) 
	{
		value = false;
	} 
	else 
	{
		if (verbosity > 1) 
		{
			fprintf(
				stderr, 
				"stringToBool: \nWarning! Bool value %s not recognised! \n", 
				string
			);
		}
		value = false;
	}

	return value;
}

int32_t stringToInt(
	const int32_t  verbosity,
	const char    *string
	) {
	
	int32_t value = 0;
	
	if ( strchr(string, '.') ) 
	{
		fprintf(
			stderr, 
			"stringToInt: \nWarning! Decimal detected in: %s, ignoring post"
			" decimal values. \n", 
			string
		);

		char *string_copy = strdup(string);
		strtok(string_copy, ".");

		string = string_copy;
	}

	if (isdigit(string[0]) || (string[0] == '-'))
	{
		value = (int32_t) atoi(string); 
	} 
	else 
	{	
		if (verbosity > 1) 
		{
			fprintf(
				stderr, 
				"stringToInt: \nWarning! Int value %s not recognised! \n", 
				string
			);
		}
		value = 0;
	}
	
	return value;
}

float stringToFloat(
	const int32_t  verbosity,
	const char    *string
	) {
	
	float value = 0.0f;
	
	if (isdigit(string[0]) || (string[0] == '-')) 
	{
		value = (float) atof(string); 
	} 
    else 
	{
		if (verbosity > 1) 
		{
			fprintf(
				stderr, 
				"stringToFloat: \nWarning! Float value %s not recognised! \n",
				string
			);
		}
		value = 0.0f;
	}
	
	return value;
}

multi_s StringToMultiS(
	const int32_t  verbosity,
	const char    *string,
	const type_e   type
	) {
	
	/**
     * Convert string to void pointer with structure of type.
     * @param 
     *     const multi_s data: multi_s structure to return in string format.
     * @see
     * @return char *string: inputted multi_s value in string format.
     */
	multi_s value;
	value.type = type;
		
	switch(type)
	{	
		case(none_e):
		break;
		
		case(bool_e  ): 
			value.value.b = stringToBool(
				verbosity,
				string
			);
		break;
		
		case(int_e   ): 
			value.value.i = stringToInt(
				verbosity,
				string
			);
		break;
		
		case(float_e ):
			value.value.f = stringToFloat(
				verbosity,
				string
			);
		break;
		
		case(char_e  ): 
			value.value.c = (char) string[0]; 
		break;

		case(string_e): 
			asprintf(&value.value.s, "%s", string); 
		break;
	
		default:
			if (verbosity > 1) 
			{
				fprintf(
					stderr, 
					"StringToMultiS: \nWarning! Type \"%s\" not recognised! \n", 
					typeToString(type)
				);
			}
			value.value.ff = NULL;
		break;
	}
	
	return value;
}

array_s stringToArrayS(
	const int32_t  verbosity,
	const char    *string,
	const type_e   type
	) {
	
	const char *delimiter = ",";
	
	char *string_copy = strdup(string);
	
	const type_e  base_type            = getBaseType(type);
	const size_t  size                 = getSizeOfType(base_type);
	const int32_t initial_num_elements = 1;
	      int32_t num_elements         = initial_num_elements;
		 	
	void *elements = malloc(sizeof(multi_s)*(size_t)num_elements);

	char *buffer = NULL;
	buffer = strtok(string_copy, delimiter);
	
	int32_t index = 0;	   
	/* walk through other tokens */
	while( buffer != NULL ) 
	{				
		int32_t position = index*(int32_t)size;
		multi_s element = 
			StringToMultiS(
					verbosity,
					buffer,
					base_type
				);
				
		memcpy(&((char*)elements)[position], &element, size);			
		index++;
		
		if (index > num_elements)
		{
			num_elements *= 2; 
			elements = realloc(
                elements, size * (size_t) num_elements + sizeof(multi_s));
		}
		
		buffer = strtok(NULL, delimiter);
	}
	
	num_elements = index;

	elements = realloc(elements, size * (size_t) num_elements);
	
	array_s array = ArrayS(
		verbosity,
		elements,
		num_elements,
		type
	);
		
	free(string_copy);
	
	return array;
}

char *MultiStoString(
	const multi_s data
	) {
	
	/**
     * Convert multi_s value to string.
     * @param 
     *     const multi_s data: multi_s structure to return in string format.
     * @see
     * @return char *string: inputted multi_s value in string format.
     */
	
	char* string;
	switch(data.type) 
	{	
		case(none_e        ): 
            printf("MultiStoString: \nWarning! Cannot convert None to string!"); 
        break;
        
		case(bool_e        ): 
            asprintf(&string, "%i", (int32_t) data.value.b  ); 
        break;
        
		case(bool_array_e  ): 
            asprintf(&string, "%p", (void *)  data.value.bb ); 
        break;
        
		case(int_e         ): 
            asprintf(&string, "%i",           data.value.i  ); 
        break;
        
		case(int_array_e   ): 
            asprintf(&string, "%p", (void *)  data.value.ii ); 
        break;
        
		case(int_jagged_e  ): 
            asprintf(&string, "%p", (void *)  data.value.iii); 
        break;			
			
		case(float_e       ): 
            asprintf(&string, "%f",           data.value.f  ); 
        break;
        
		case(float_array_e ): 
            asprintf(&string, "%p", (void *)  data.value.ff ); 
        break;
        
		case(float_jagged_e): 
            asprintf(&string, "%p", (void *)  data.value.fff); 
        break;		

		case(char_e        ): 
            asprintf(&string, "%c", data.value.c  ); 
        break;
        
		case(char_array_e  ): 
            asprintf(&string, "%p", (void *)  data.value.cc ); 
        break;

		case(string_e      ): 
            string = data.value.s; 
        break;		
        
		case(string_array_e): 
            asprintf(&string, "%p", (void *)  data.value.ss ); 
        break;
			
		default:
			fprintf(
				stderr, 
				"MultiStoString: \nWarning! Type \"%s\" not recognised! \n", 
				typeToString(data.type)
			);
		break;
	}
	
	return string;
}

void* allocateTypeArray(
	const type_e  type, 
	const int32_t num_elements
    ) {
	
	/**
     * Allocate array of size of type with num_elements.
     * @param 
     *     const type_e  type        : type to allococate as array elements.
	 *     const int32_t num_elements: num_elements to allocate.
     * @see
     * @return void *array: pointer to newly assigned memory.
     */
	
	return malloc(getSizeOfType(type) * (size_t) num_elements);
}

void freeMultiS(
	multi_s data
	) {
	
	/**
     * Free memory assigned to multi_s structure.
     * @param 
     *     const multi_s data: multi_s structure to return in string format.
     * @see
     * @return none.
     */
	
	switch(data.type) {
		
		case(none_e   ): 
		
		break;
		
		case(bool_array_e   ): 
			
			if (data.value.bb  != NULL) 
			{
				free(data.value.bb); 
			} 
		break;
		
		case(int_array_e   ): 
			
			if (data.value.ii  != NULL) 
			{
				free(data.value.ii); 
			} 
		break;
		
		case(float_array_e ): 
			
			if (data.value.ff  != NULL) 
			{
				free(data.value.ff);
			} 
		break;
		
		case(char_array_e   ): 
			
			if (data.value.cc != NULL) 
			{
				free(data.value.cc); 
			} 
		break;
		
		case(string_array_e): 
			
			if (data.value.ss  != NULL) 
			{
				free(data.value.ss); 
			}
		break;
		
		case(int_jagged_e  ): 
			
			if (data.value.iii != NULL) 
			{		
				for (int32_t index = 0; index < data.length; index++) 
				{
					if (data.value.iii[index] != NULL) 
					{

						free(data.value.iii[index]);
					}

					if (data.lengths != NULL) {

						free(data.lengths);
					}

				}

				free(data.value.iii);
			} 
		break;
			
		case(float_jagged_e):
			
			if (data.value.fff != NULL) 
			{
				for (int32_t index = 0; index < data.length; index++) 
				{	
					if (data.value.fff[index] != NULL) 
					{	
						free(data.value.fff[index]);
					}
				}
                
                if (data.lengths != NULL) 
				{		
					free(data.lengths);
				}
                
                free(data.value.fff);
			} 
		break;
		
		default:

			fprintf(
				stderr, 
				"freeMultiS: \nWarning! Type \"%s\" not recognised! \n", 
				typeToString(data.type)
			);
		break;
	}
}

typedef struct DictEntry {
	
	      multi_s     data;   
	
          int32_t     hash; 
    const char       *string_key;
	
	struct DictEntry *previous_entry;
	struct DictEntry *next_entry;
	
} dict_entry_s;

typedef struct Dictionary {
	
	int32_t        length;
	int32_t        num_entries;
	
	dict_entry_s **entries;
	dict_entry_s  *last_entry;
	
} dict_s;

dict_s* makeDictionary(
	const int32_t length
	) {
	
	/**
     * Create dictionary data structure.
     * @param 
     *     const int32_t length: number of elements to allocate for storage of 
     *                           unique hashes.
     * @see
     * @return dict_s *dictionary: pointer to newly created dictionary.
     */
	
	dict_s* dict      = malloc(sizeof(dict_s));
	
	dict->length      = length;
	dict->num_entries = 0;
	dict->entries     = calloc( (size_t) length, sizeof(dict_entry_s*));
	dict->last_entry  = NULL;
	
	return dict;
}

int32_t StringToKey(
	const char *string_key
	) {
	
	/**
     * Generate pseudo-random number from string.
     * @param 
     *     const char string_key: String key to convert to numerical key.
     * @see getDictHashCode()
     * @return int32_t key: pseudo random numerical key.
     */
	
	int     index   = 0;
	char    current = string_key[index];
	int32_t key     = 0;
	
	while (current != '\0') {
		
		key += (int) current*index;	
		index++;
		
		current = string_key[index];
	}
	return key;
}

int32_t getDictHashCode(
	const dict_s *dict, 
	const char   *string_key
	) {
	
	/**
     * Find dict hash given string key input. Will
	 * always return same hash for the same string,
	 * but Dose not garantee different hash for 
	 * differnt strings.
     * @param 
	 *     const dict_s *dict     : Dictionary to hash with string_key.
     *     const char  *string_key: String key to link to dictionary index.
     * @see StringToKey()
     * @return int32_t hash: hash to dictionary index.
     */

   const int32_t key = StringToKey(string_key);
   
   int32_t hash = -1;
   
   if (dict->length > 0) 
   {
        hash = key % dict->length;
   }
   
   return hash;
}

int32_t insertDictEntry(
	      dict_s  *dict, 
	const multi_s  data, 
    const char    *string_key_o
	) {
	
	/**
     * Insert new multi_s entry into dictionary pointed to by dict.
     * @param 
	 *     const dict_s  *dict      : Dictionary to hash with string_key.
	 *     const multi_s *data      : Data to insert into dictionary.
     *     const char    *string_key: String key to insert data at given hash.
     * @see getDictHashCode(), multi_s.
     * @return int32_t return_value: Returns 0 if successful, else return 1.
     */
	
	const char *string_key = strdup(string_key_o);
	
	dict->num_entries++;
	if (dict->num_entries > dict->length) 
    {
		fprintf(
            stderr, 
            "Warning dictionary too small! Cannot add entry: %s. \n", 
            string_key
        );
        
		return 1;
	}
	
	const int32_t hash = getDictHashCode(dict, string_key);
	
    if ((hash >= 0) && (hash < dict->length))
    {
        dict_entry_s* entry = malloc(sizeof(dict_entry_s));

        *entry = (dict_entry_s) {  
            data            ,
            hash            ,
            string_key      ,
            dict->last_entry,
            NULL
        };

        if ( dict->last_entry != NULL) 
        {
            dict->last_entry->next_entry = entry;
        }

        int32_t index = hash;
        while (dict->entries[index] != NULL)
        {
            index++; index %= dict->length; //Wrap around. 
        }
    
        dict->entries[index] = entry;
        dict->last_entry     = entry;
        
        return 0;
    }
	
	return 1;
}

int32_t findHashIndex(
	const dict_s *dict, 
	const char   *string_key
	) {
	
	/**
     * Find dictionary entry index from entered string key.
     * @param 
	 *     const dict_s  *dict      : Dictionary to find hash for specified 
     *                                string key.
     *     const char    *string_key: String key to find hash index from.
     * @see getDictHashCode().
     * @return int32_t index: Returns hash index if successful, else return -1.
     */
	 
	const int32_t hash = getDictHashCode(dict, string_key);
    
    if ((hash >= 0) && (hash < dict->length))
    {
              int32_t index = hash;

        const int32_t stop_index = 
            ((hash - 1) < 0) ? (dict->length - 1) : (hash - 1);

        while (dict->entries[index] != NULL) 
        {
            if (!strcmp(dict->entries[index]->string_key, string_key)) 
            {
                return index;
            } 
            else if (index == stop_index) 
            {
                return -1;		
            }

            index++; index %= dict->length; //Wrap around. 
        }
    }
	
	return -1;
}

dict_entry_s *findDictEntry_(
	const dict_s *dict, 
	const char   *string_key
	) {
	
	/**
     * Find dictionary entry from entered string key.
     * @param 
	 *     const dict_s  *dict      : Dictionary to find entry for specified 
     *                                string key.
     *     const char    *string_key: String key to find dictionary entry of.
     * @see findHashIndex().
     * @return dict_entry_s entry: Returns dict entry if successful, else return 
     *                             NULL.
     */
	
	const int32_t hash  = findHashIndex(dict, string_key);
    dict_entry_s *entry = NULL;

    if ((hash >= 0) && (hash < dict->length))
    {
        const int32_t index = hash;
        
        if (index >= 0) 
        {	
            entry = dict->entries[index];
        }
    }

	return entry;
}

dict_entry_s *findDictEntry(
	const dict_s *dict, 
	const char   *string_key
	) {
	
	/**
     * Find dictionary entry from entered string key, with error wrapper.
     * @param 
	 *     const dict_s  *dict      : Dictionary to find entry for specified 
     *                                string key.
     *     const char    *string_key: String key to find dictionary entry of.
     * @see findHashIndex().
     * @return dict_entry_s entry: Returns dict entry if successful, else return 
     *                             NULL.
     */
		
	dict_entry_s *entry = findDictEntry_(dict, string_key);
	
	if (entry == NULL)
	{
		fprintf(
            stderr, 
            "findDictEntry: \nWarning! Cannot find entry! %s. \n", 
            string_key
        );
        
		return NULL;
	}
	
	return entry;
}

void deleteDictEntry(
	        dict_s *dict,
	const   char   *string_key
	) {
	
	/**
     * Delete dictionary entry at entered string key.
     * @param 
	 *     const dict_s  *dict      : Dictionary to find entry for specified 
     *                                string key.
     *     const char    *string_key: String key to delete dictionary entry at.
     * @see findHashIndex().
     * @return none.
     */
	
	const int32_t hash = findHashIndex(dict, string_key);
	
    if ((hash >= 0) && (hash < dict->length))
    {
        const int32_t index = hash; 

        if (index >= 0) {

            if (dict->entries[index]->previous_entry != NULL) 
            {
                if (dict->entries[index]->next_entry != NULL) 
                {
                    dict->entries[index]->previous_entry->next_entry = 
                        dict->entries[index]->next_entry;
                    dict->entries[index]->next_entry->previous_entry = 
                        dict->entries[index]->previous_entry;

                } 
                else 
                {
                    dict->entries[index]->previous_entry->next_entry = NULL;
                    dict->last_entry = dict->entries[index]->previous_entry;
                }
            } 

            dict->num_entries--;
            free(dict->entries[index]);

            dict->entries[index] = NULL;
        }
    }
}

dict_entry_s** returnAllEntries(
	const dict_s* dict
	) {
	
	/**
     * Returns array of all dictionary entries
     * @param 
	 *     const dict_s  *dict: Dictionary to find entry for specified string 
     *                          key.
     * @see findHashIndex().
     * @return dict_entry_s** entries: array of dict entries.
     */
	
	dict_entry_s** entries = malloc(1);
	if ((dict->num_entries > 0) && (dict->last_entry != NULL)) 
    {
		entries = 
            realloc(
                entries, sizeof(dict_entry_s*) * (size_t) dict->num_entries
            );
		
		entries[dict->num_entries - 1] = dict->last_entry;
		for (int32_t index = dict->num_entries - 2; index >= 0; index--)
        {
			entries[index] = entries[index + 1]->previous_entry;
		}
	}
	
	return entries;
}

void printDictKeys(
	const dict_s* dict
	) {
	
	/**
     * Print all dictionary keys.
     * @param 
	 *     const dict_s  *dict: Dictionary to print all keys.
     * @see returnAllEntries().
     * @return none
     */
	
	dict_entry_s** entries = returnAllEntries(dict);

	for (int32_t index = 0; index < dict->num_entries; index++) 
    {
		printf("%s \n", entries[index]->string_key);	
	}
}

void freeDictEntry(
	dict_entry_s* entry
	) {	
	
	/**
     * Deallocate memory assigned to dictinary entry
     * @param 
	 *     const dict_entry_s  *entry: Dictionary entry to deallocate.
     * @see freeMultiS().
     * @return none
     */
	
	freeMultiS(entry->data);
	free(entry);
}

void freeDictionary(
	dict_s* dict
	) {
	
	/**
     * Deallocate all memory assigned to dictinary and its entry.
     * @param 
	 *     const dict_s  *dict      : Dictionary entry to deallocate.
     * @see freeMultiS().
     * @return none
     */
	
	dict_entry_s **entries = returnAllEntries(dict);
	
	for (int32_t index = 0; index < dict->num_entries; index++) 
    {
		freeDictEntry(entries[index]);
	}
	
	free(entries);
	free(dict->entries);
	free(dict);
}

typedef struct Mapping {

	/**
     * Structure to hold map to convert string to index.
     */
    
    dict_s   *dict;
    char    **keys;    
    int32_t   length;
} map_s;

typedef struct Map_Pairing {
	
	/**
	 * Structure to hold index key pair.
	 */
    
    char    *key;
    int32_t  index;
    map_s    map;
} map_pair_s;

map_s createMap(
          char    **keys    , 
    const int32_t   num_keys
    ) {
	
	/**
     * Create map between array of strings and array of ints
     * @param 
	 *           char    ** keys    : Strings to map.
	 *     const int32_t    num_keys: Number of strings to map.
     * @see makeDictionary(), insertDictEntry().
     * @return map_s map: map between keys and index.
     */
	    
    map_s map;
    map.dict = makeDictionary(10*num_keys);
    
	map.keys = malloc(sizeof(char*) * (size_t) num_keys);
    for (int32_t index = 0; index < num_keys; index++) 
    {
        const multi_s data = (multi_s) {*((multi_u*)(&index)), int_e, 1, NULL};
        insertDictEntry(map.dict, data, keys[index]);
		map.keys[index] = strdup(keys[index]);
	}   
    
	map.length = num_keys;
    
    return map;
}

int32_t getMapIndex(
        const map_s   map, 
        const char   *key
    ) {
	
	/**
     * Return index of key in map.
     * @param 
	 *     const map_s  map: Key to index map.
	 *     const char  *key: String to find index of in map.
     * @see findDictEntry(), createMap().
     * @return int32_t index: Index corresponding to inputted key.
     */
	
	int32_t index = -1;
	
	dict_entry_s *entry = 
		findDictEntry_(map.dict, key);
	
	if ( entry!= NULL ) 
	{
		index = entry->data.value.i;
	} 
    else 
	{
		index = -1;
	}
    
    return index;
}

char *getMapKey(
    const map_s   map,
    const int32_t index
    ) {
	
	/**
     * Find string corresponding to inputted index
     * @param 
	 *     const map_s   map: Key to index map.
	 *     const int32_t index: Index to find correspinding string in map.
     * @see findDictEntry(), createMap().
     * @return char *key: String_key corresponding to inputted index.
     */
	
	char* key = NULL;
	if ((index < map.length) && (map.length > 0)) 
    {
		key = map.keys[index];
	} 
    else 
    {
		key = NULL;
	}
    
    return key;
}
    
#endif