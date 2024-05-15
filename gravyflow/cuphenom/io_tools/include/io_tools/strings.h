#ifndef IO_STRINGS_H
#define IO_STRINGS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

#include <inttypes.h>
#include <string.h>

char *strrev(
    const char *string
    ) {
	
	/**
     * Reverse order of characters in string.
     * @param 
     *     const char *string: string to be reversed.
     * @see
     * @return char *rev_string: reversed string. Caller must free this memory 
     * when no longer needed.
     */
    
	// Get length of string to be reversed:
    const size_t length = strlen(string);
    
	// Allocate memory for reversed string:
    char *rev_string = (char*)malloc(sizeof(char) * (length + (size_t)1u));
	
	// Loop through string length:
    for (int32_t index = 0; index < length; index++) 
    {
		// Set new string character as reverse of old string:
        rev_string[length - 1 - index] = string[index];
    }
	
	//Add null terminating string:
    rev_string[length] = '\0';
	
    return rev_string;
}

int32_t compareStrings(
    const void *string_a, 
	const void *string_b
	) { 

	/**
     * Function for use in string sort. 
     * @param 
     *     const char *string_a: first string to compare.
	 *     const char *string_b: second string to compare.
     * @see
     * @return int32_t comparison value: 
     */

    const char **ia = (const char **)string_a;
    const char **ib = (const char **)string_b;
	
    return strcmp(*ia, *ib);
} 

void splitString(
    const char      *string, 
    const int32_t    num_strings, 
    const char      *delimiter, 
          char    ***ret_string_array
    ) {
	
	/**
     * Split larger string into substrings a delimeter for specified number of 
     * substrings.
     * @param 
     *     const char      *string          : string to be split into 
     *                                        substrings.
	 *     const int32_t    num_strings     : number of substrings to look for.
	 *     const char      *delimieter      : character by which to split the 
     *                                        strings.
	 *     const char    ***ret_string_array: pointer to array of substrings 
     *                                        split from string. Caller must 
     *                                        free this memory when no longer 
     *                                        needed.
     * @see
     * @return none
     */
	
	// Make a copy of old string as strtok is an inplace operation:
	char  *string_copy  = strdup(string);
	
	// Allocate array to hold pointers to substrings:
	char **string_array = (char**)malloc(sizeof(char*) * (size_t) num_strings);
	
	// Get first substring:
	string_array[0] = strtok(string_copy, delimiter);

	// Loop through remaining strings to separate substrings:
	for (int32_t string_idx = 1; string_idx < num_strings; string_idx++)
    {
		// Get pointer to start of next substring and replace delimiter with end 
        // of string character:
		string_array[string_idx] = strtok(NULL, delimiter);
	}
	
	// Return separated substring array:
	*ret_string_array = string_array;
}

void splitStringDy(
    const char      *string, 
    const char      *delimiter, 
          char    ***ret_string_array, 
          int32_t   *ret_num_strings
    ) {
	
    /**
     * Split larger string into substrings a delimeter for dynamic number of 
     * substrings.
     * @param 
     *     const char      *string          : string to be split into 
     *                                        substrings.
	 *     const char      *delimieter      : character by which to split the 
     *                                        strings.
	 *     const char    ***ret_string_array: pointer to array of substrings 
     *                                        split from string. Caller must 
     *                                        free this memory when no longer 
     *                                        needed.
	 *     const int32_t   *ret_num_strings : number of substrings found.
     * @see
     * @return none
     */
	
	// Number of strings to look for initially:
	const int32_t initial_num_strings = 2;
	
	// Make a copy of old string as strtok is an inplace operation:
    char *string_copy  = strdup(string);
	
	// Allocate array to hold pointers to substrings:
	char    **string_array = 
        (char**)malloc(sizeof(char*) * (size_t)initial_num_strings);
	
	//Set number of strings to initial value:
	int32_t   num_strings = initial_num_strings;
		
	//Intilize token to hold substring pointer:
	char    *token;
    
	// Get first substring:
	token = strtok(string_copy, delimiter);
    
	int32_t  index = 0;
	while (token != NULL) 
    {
		// Expand string pointer array if number of strings found 
		// exceeds intial number allocated:
		if (index >= initial_num_strings) 
        {
			// Increase number of strings in string array:
			num_strings *= 2;
			
			string_array = 
				(char**)realloc(
                    string_array, sizeof(char*) * (size_t) (num_strings)
                );
		}
		
		// Set string array value to new pointer:
		string_array[index] = token;
				
		// Get pointer to start of next substring and replace delimiter with end 
        // of string character:
		token = strtok(NULL, delimiter);
		
		// Incrament string index:
		index++;
	}
	
	//Reallocate memory to final number of strings found:
	string_array = 
		(char**)realloc(string_array, sizeof(char*) * (size_t)index);
	
	// Set return string length and num returned substrings:
	*ret_string_array = string_array;
	*ret_num_strings  = index;
}

void removeStringChars(
    const char  *string, 
    const char  *remove, 
          char **ret_string_removed
    ) {
	
	/**
     * Remove all isntances of character specified by remove from inputted 
     * string.
     * @param 
     *     const char  *string            : old string to remove character 
     *                                      from.
	 *     const char  *remove            : character to remove from string.
	 *     const char **ret_string_removed: new string with character removed. 
     * @see
     * @return none
     */
	
	// Allocate memory for new string with removed character:
	char *string_removed = 
		(char*)calloc(strlen(string) + (size_t)1u, sizeof(char));

	int32_t remove_char_idx = 0; // Index of new string.
	int32_t char_idx        = 0; // Index of old string.
	while (string[char_idx])
    {
		// Strchr returns index of next instance of remove,
		// if it returns 0, current index is instance of string,
		// so do not add to new string:
		if ( !strchr(remove, string[char_idx]))
        {
			// Set new string to old string value if not equal
			// to remove character:
			string_removed[remove_char_idx] = string[char_idx];
			remove_char_idx++;
		}
		char_idx++;
	}
	
	// Set return pointer to new string with string removed:
	*ret_string_removed = string_removed;
}

char *intToString(
    const int32_t value
    ) {
	
	/**
     * Convert an integer to string.
     * @param 
     *     const int32_t *value: integer value to convert to string.
     * @see
     * @return char *: string: converted string.
     */
		
	char *string; // Initilise pointer to hold new string.
	asprintf(&string, "%i", value); //Convert integer to string.
	
	return string;
}

char* replaceWord(
    const char *string, 
    const char *old_word,
    const char *new_word
    ) {
	
	/**
     * Replace all instances of old_word in string with long word.
     * @param 
     *     const char *string  : string in which to replace words.
	 *     const char *old_word: word to replace with new_word.
	 *     const char *new_word: word to replace old_word.
     * @see
     * @return char *: string: string with replaced words. Caller must free this 
     * memory when no longer needed.
     */
	
	// Intilise pointer to hold resulting pointer:
          char   *result;
		  
	// Get length of old word and replacment:
    const size_t  new_word_length = strlen(new_word);
    const size_t  old_word_length = strlen(old_word);
	
	// Intilise count and index:
	      size_t  index, count  = 0u;

    // Counting the number of times old word
    // occur in the string:
    for (index = 0; string[index] != '\0'; index++) 
    {
        if (strstr(&string[index], old_word) == &string[index]) 
        {
            count++;
  
            // Jumping to index after the old word.
            index += old_word_length - 1u;
        }
    }
  
    // Making new string of enough length:
    result = 
		(char*)malloc(
			sizeof(char) * 
			(index + count * (new_word_length - old_word_length) + (size_t)1u)
		);
	  
	// Reset Index:
    index = 0;
    while (*string) 
    {
        // Compare the substring with the result
        if (strstr(string, old_word) == string) 
        {
            strcpy(&result[index], new_word);
            index += new_word_length;
            string += old_word_length;
        }
        else
            result[index++] = *string++;
    }
    
	//Set end of string character:
    result[index] = '\0';
	
    return result;
}

char* multiplyString(
    const char    *string, 
    const int32_t  value 
    ) {
	
	/**
     * Multiply string by value.
     * @param 
     *     const char    *string: string to multiply.
	 *     const int32_t *value : integer value to copy string by.
     * @see
     * @return char *string: string multiplied by value. Caller must free this 
     * memory when no longer needed.
     */
    
	// Get length of old string:
    const size_t old_length = sizeof(char) * strlen(string);
	
	// Calculate length of new string:
	const size_t new_length = old_length * (size_t) value;
	
	// Allocate value for new string:
	char *new_string = (char*)malloc(new_length + (size_t)1u);  
    
	// Loop through string and set multplicants:
    for (size_t index = 0; index < new_length; index++) 
    {
        new_string[index] = string[index % strlen(string)];
    }
    
	//Set end of string character:
    new_string[new_length] = '\0';
         
    return new_string;
}

#endif