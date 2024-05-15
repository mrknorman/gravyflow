#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include <inttypes.h>

#include "test.h"
#include "text.h"
#include "console.h"

int main() {
	
	const int32_t verbosity = 3;
	bool pass = true;
    
	const char *file_path = "./test_path/test_path/test_file";
	const char *base      = "test_file";
	const char* directory = "./test_path/test_path";
	
	path_s path = newPath(file_path);
	
	printPath(path);
	
	pass *= !strcmp(path.full, file_path);
	pass *= !strcmp(path.base, base);
	pass *= !strcmp(path.directory, directory);

	printTestResult(pass, "Path test");
	
	pass *= createDirectory(directory);
	
	struct stat sb;
	pass *= !stat(directory, &sb);
	
	printTestResult(pass, "Recursive create test");
	
	remove(directory);
	remove(newPath(directory).directory);
	
    return 0;
}