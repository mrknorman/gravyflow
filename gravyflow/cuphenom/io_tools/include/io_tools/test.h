#ifndef IO_TEST_H
#define IO_TEST_H

void printTestResult(
    const int32_t  pass,
    const char    *test_name
    ) {
    
    /**
     * Prints result of test
     * @param 
     *     const bool       pass: test result.
     *     const char *test_name: test name.
     * @see
     * @return void
     */
    
    const char *result = pass ? "PASSED" : "FAILED";
    printf("%s: %s.\n\n", test_name, result); 
}

bool checkNotNULL(
    const void *pointer,
    const char *test,
    const char *name
    ) {
    
    bool pass = false;
    
    if (pointer != NULL) 
	{
		printf("%s %s did not return NULL! As expected.\n", test, name);
		pass = true;
	} 
	else
	{
		printf("%s %s returned NULL! Unexpected behaviour.\n", test, name);
		pass = false;
	}
    
    return pass;
}

bool checkNULL(
    const void *pointer,
    const char *test,
    const char *name
    ) {
    
    bool pass = false;
    
	if (pointer == NULL) 
	{
		printf("%s %s returned NULL! As expected.\n", test, name);
		pass = true;
	} 
	else
	{
		printf(
            "%s %s did not return NULL! Unexpected behaviour.\n", 
            test, 
            name
        );
        
		pass = false;
	}
    
    return pass;
}

#endif