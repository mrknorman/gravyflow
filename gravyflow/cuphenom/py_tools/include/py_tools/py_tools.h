#ifndef PY_TOOLS_H
#define PY_TOOLS_H

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#include <inttypes.h>
#include <math.h>

#include <Python.h>
#include <numpy/arrayobject.h>

#include "io_tools/text.h"

#ifndef VERBOSITIES
#define VERBOSITIES

#define SILENT 0
#define ERROR_ONLY 1
#define ERROR_AND_WARNINGS_ONLY 2
#define STANDARD 3
#define EXTRA 4

#endif

#define INIT_PY(...) do { \
    if (!Py_IsInitialized()) { initPy(); }\
    if(Py_IsInitialized()) {__VA_ARGS__}\
    else { \
        if (verbosity > 0)\
        {\
            fprintf(\
                stderr, \
                "PyFunctions: \nPython not intilizised.\n"\
            );\
        }\
    return 1000; }\
    } while(0)

bool initPy() {
	
	/**
     * Runs Py_Initialize() from the C Python API in order to initialize the 
	 * Python interpreter, and also adds current directory to the path. Run
	 * before runing any other python functions. Can be found in INIT_PY macro.
     * @param 
     * @see deinitPy
     * @return bool : returns true if initlization was succesfull.
     */
    
	// Initlize Python interpreter, because initsigs is 0, it skips 
	// registration of signal handlers, which might be useful when Python is 
	// embedded:
	Py_InitializeEx(0);
	
	// Add current path to system path:
	PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
	
	// Return true if Python is initilized:
	return Py_IsInitialized();
}

void deinitPy() {
	
	/**
     * Runs Py_FinalizeEx() from the C Python API in order to deinitilize the 
	 * Python interpreter. Currently unused as deintilisation and subsequent 
	 * reintilization causes segfault with some common Python packages like 
	 * NumPy.
     * @param 
     * @see initPy
     * @return void
     */
	
	// Dereference threadinf module:
    Py_DECREF(PyImport_ImportModule("threading"));
	
	// Exits Python interpreter:
    Py_FinalizeEx();
}

PyObject *makePyArrayFloat(
    const float   *array, 
    const int32_t  num_elements
    ) {
	
	/**
     * Converts C array of floats into NumPy PyArray using Numpy C API.
     * @param 
	 * const float   *array       : array to convert to numpy array.
	 * const int32_t  num_elements: number of elements in the array to be
	 *                              converted.
     * @see makePyArrayShort, makePyArrayInt, makeStringList
     * @return PyObject* : Newly created NumPy PyArray.
     */
	
	// Import and initilise NumPy array module:
	import_array();
	
	// Convert num_elements into Numpy int format;
	npy_intp dims = (npy_intp) num_elements;
	
	// Returns newly created Numpy float array:
	return PyArray_SimpleNewFromData(1, &dims, NPY_FLOAT, (void*)array);	
}

PyObject *makePyArrayShort(
    const uint16_t *array, 
    const int32_t   num_elements
    ) {
	
	/**
     * Converts C array of unsigned 16 bit integers into NumPy PyArray using 
	 * Numpy C API.
     * @param 
	 *     const uint16_t *array       : array to convert to numpy array.
	 *     const int32_t   num_elements: number of elements in the array to be
	 *                                   converted.
     * @see makePyArrayFloat, makePyArrayInt, makeStringList
     * @return PyObject* : Newly created NumPy PyArray.
     */
	
	// Import and initilise NumPy array module:
	import_array();
	
	// Convert num_elements into Numpy int format;
	npy_intp dims = (npy_intp) num_elements;
	
	// Returns newly created Numpy int array:
	return PyArray_SimpleNewFromData(1, &dims, NPY_SHORT, (void*)array);	
}

PyObject *makePyArrayInt(
    const int32_t *array, 
    const int32_t  num_elements
    ) {
	
	/**
     * Converts C array of 32 bit integers into NumPy PyArray using 
	 * Numpy C API.
     * @param 
	 *     const int32_t *array       : array to convert to NumPy array.
	 *     const int32_t  num_elements: number of elements in the array to be
	 *                                  converted.
     * @see makePyArrayShort, makePyArrayFloat, makeStringList
     * @return PyObject* : Newly created NumPy PyArray.
     */

	// Import and initilise NumPy array module:
	import_array();
	
	// Convert num_elements into NumpyInt format;
	npy_intp dims = (npy_intp) num_elements;
	
	// Returns newly created Numpy int array:
	return PyArray_SimpleNewFromData(1, &dims, NPY_INT, (void*)array);	
}

PyObject *makeListString(
          char    **string, 
    const int32_t   num_elements
    ) {
	
	/**
     * Converts C array of strings into Python list of strings using Python C
	 * API.
     * @param
	 *     const int32_t *array       : array to convert to Python list.
	 *     const int32_t  num_elements: number of elements in the array to be
	 *                              converted.
     * @see makePyArrayShort, makePyArrayInt
     * @return PyObject* : Newly created NumPy PyArray.
     */
	
	// Create new Python list:
    PyObject* list = PyList_New(num_elements);
	
	// Loop over string array elements and assign each to list element after 
	// converting to Python unicode format:
    for (int32_t index = 0; index < num_elements; index++) 
	{
        PyList_SetItem(list, index, PyUnicode_FromString(string[index]));
    }
	
	// Return newly created Python list:
    return list;
}

float *makeArr(
          PyObject *list, 
    const int32_t   num_elements
    ) {
	
	/**
     * Converts Python list of floats into C array of floats using Python C
	 * API.
     * @param
	 *     const PyObject *list       : Python list to convert to numpy list.
	 *     const int32_t  num_elements: number of elements in the python list
	 *                                  converted.
     * @see
     * @return float* : Newly created C array.
     */

	// Allocate C array to hold list values:
    float *array = malloc(sizeof(float)*(size_t)num_elements);
	
	// Loop over Python list elements, convert back to floats, and assign to 
	// newly created array:
    for (int32_t index = 0; index < num_elements; index++) 
	{
        array[index] = (float)PyFloat_AsDouble(PyList_GetItem(list, index));
    }
	
	// Return newly created C array: 
    return array;
}

bool runPythonFunc(
	const int32_t    verbosity,
    const int32_t    num_args, 
    const char      *module_path, 
    const char      *function_name, 
          PyObject **args, 
          float    **output_array, 
          int32_t    num_output_elements
    ) {
	
	/**
     * Runs Python function and returns output to C float array.
     * @param
	 *    const int32_t verbosity: 
	 *	      Dictates the level of logs printed to the
	 *        console. Can hold the following values, higher
	 *        values are more verbose:
	 *           * SILENT     = 0 : Nothing printed.
	 *           * ERROR_ONLY = 1 : Only errors are printed.
	 *           * ERROR_AND_WARNINGS_ONLY = 2: 
	 *               Only errors and warnings are printed.
	 *           * STANDARD   = 3 : All standard log messages are printed.
	 *           * EXTRA     := 4 : More detailed log messages are printed.
	 *    const int32_t num_args: 
	 *        Number of arguments to requested Python function.
	 *    const char *module_path: 
	 *        Path to the Python file containing requested function.
	 *    const char *function_name: Name of the requested function.
	 *    PyObject **args: 
	 *        Array of argyments to requested function.
	 *    float **output_array: 
	 *        Pointer to fill with output array from requested Python function.
	 *    int32_t num_output_elements:
     *        Num outputted function elements.
	 * 
     * @see
     * @return float* : Newly created C array.
     */
	
	int32_t error_code = 0;
	
    PyObject *pModule, *pFunc;
    PyObject *pArgs, *pValue;
	
	path_s model_path = newPath(module_path);
	
	PyObject *sys_path = PySys_GetObject("path");
	PyList_Append(sys_path, PyUnicode_FromString(model_path.directory));

    pModule = PyImport_ImportModule(model_path.base);

    if (pModule != NULL) 
	{
        pFunc = PyObject_GetAttrString(pModule,  function_name);

        if (pFunc && PyCallable_Check(pFunc)) 
		{
            pArgs = PyTuple_New(num_args);
            for (int32_t index = 0; index < num_args; index++) 
			{
                pValue = args[index];
                if (!pValue) {
				
                    Py_DECREF(pArgs);
                    Py_DECREF(pModule);
					
					if (verbosity > SILENT)
					{
						fprintf(
							stderr, 
							"runPythonFunc:\n Error! Cannot convert argument: "
							"%i.\n", 
							index
						);
					}
					
                    error_code += 1;
					return error_code;
                }
				
                PyTuple_SetItem(pArgs, index, pValue);
            }

            pValue = PyObject_CallObject(pFunc, pArgs);

            *output_array = makeArr(pValue, num_output_elements);

            Py_DECREF(pArgs);
            if (pValue != NULL)
			{
                Py_DECREF(pValue);
            } 
			else 
			{
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
				
				if (verbosity > SILENT)
				{
					fprintf(stderr,"runPythonFunc:\n Error! Call failed.\n");
				}
				
                error_code += 10;
				return error_code;
            }
			
        } 
		else 
		{
            if (PyErr_Occurred()) 
			{
				PyErr_Print();
			}
			
			if (verbosity > SILENT)
			{
				fprintf(
                    stderr, 
                    "runPythonFunc:\n Cannot find function: \"%s\".\n", 
                    function_name
                );
			}
		}
		
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    } 
	else 
	{
        PyErr_Print();
		
		if (verbosity > SILENT)
		{
			fprintf(
                stderr, 
                "runPythonFunc:\n Error! Failed to load: \"%s\".\n", 
                model_path.base
            );
		}
		
        error_code += 100;
		return error_code;
    }

    return error_code;
}

//Python Plotting Functions:
typedef struct Axis{
    char *name;
    char *label;
} axis_s;

PyObject *convertAxisToPyObject(
    const axis_s axis
    ) {
    
    // Unpack axis:
    const char *name  = axis.name;
    const char *label = axis.label;
    
    // Set type format:
    const char *template = "ss";
    return Py_BuildValue(template, name, label);
}

typedef struct Axes{ 
    char    *name;
    int32_t  num_axis;
    axis_s  *axis;
} axes_s;

PyObject *convertAxesToPyObject(
    const axes_s axes
    ) {
    
    // Unpack axes:
    const char    *name     = axes.name;
    const int32_t  num_axis = axes.num_axis;
    const axis_s  *axis     = axes.axis;
    
    PyObject *py_axis = PyTuple_New(num_axis);
    for (int32_t index = 0; index < num_axis; index++)
    {
        PyTuple_SetItem(
            py_axis,
            index,
            convertAxisToPyObject(axis[index])
        );
    }
    
    // Set type format:
    const char *template = "sO";
    return Py_BuildValue(template, name, py_axis);
}

typedef struct SeriesValues{
    char   *axis_name;
    float  *values;
} series_values_s;

PyObject *convertSeriesValuesToPyObject(
    const series_values_s values,
    const int32_t         num_elements
    ) {
    
    // Unpack axes:
    const char  *axis_name   = values.axis_name;
    const float *values_     = values.values;
    
    PyObject *py_values = makePyArrayFloat(values_, num_elements);
    
    // Set type format:
    const char *template = "sO";
    return Py_BuildValue(template, axis_name, py_values);
}

typedef struct Series{
    char   *label;
    char   *axes_name;
    int32_t num_elements;   
    int32_t num_axis;

    series_values_s *values;
} series_s;

PyObject *convertSeriesToPyObject(
    const int32_t  verbosity,
    const series_s series
    ) {
    
    // Unpack series:
    const char            *label        = series.label;
    const char            *axes_name    = series.axes_name;
    const int32_t          num_elements = series.num_elements;
    const int32_t          num_axis     = series.num_axis;
    const series_values_s *values       = series.values;
    
    PyObject *py_values = PyTuple_New(num_axis);
    for (int32_t index = 0; index < num_axis; index++)
    {
        if (( 
		PyTuple_SetItem(
            py_values,
            index,
            convertSeriesValuesToPyObject(values[index], 
			num_elements
		)) != 0 ) 
		&& (verbosity > SILENT)) {
		
         fprintf(
                stderr, 
                "convertSeriesToPyObject: \n"
                "Error! cannot set tuple item values %s.",
                values[index].axis_name
            );
        }
    }
    
    // Set type format:
    const char *template = "ssO";
    
    PyObject *tuple = Py_BuildValue(template, label, axes_name, py_values);
    if ((tuple == NULL) && (verbosity > SILENT))
    {
        fprintf(
            stderr, 
            "ConvertFigureToPyObject:\n"
            "Error! Cannot build value.\n"
        );
    }
    
    return tuple;
}

typedef struct Figure{
    
    axes_s   *axes;
    int32_t   num_axes;

    series_s *series;
    int32_t   num_series;
} figure_s;

PyObject *convertFigureToPyObject(
    const int32_t  verbosity, 
    const figure_s figure
    ) {
    
    const int32_t   num_axes = figure.num_axes;
    const axes_s   *axes     = figure.axes;
    
    PyObject *py_axes = PyTuple_New(num_axes);
    for (int32_t index = 0; index < num_axes; index++)
    {
        if (PyTuple_SetItem(
            py_axes,
            index,
            convertAxesToPyObject(axes[index])
            ) != 0
        ) {
            if (verbosity > SILENT)
            {
                fprintf(
                    stderr, 
                    "ConvertFigureToPyObject:\n"
                    "Error! Cannot set tuple item axis %s.",
                    axes[index].name
                );
                
                return NULL;
            }
        }
    }
    
    const int32_t   num_series = figure.num_series;
    const series_s *series     = figure.series;
    
    PyObject *py_series = PyTuple_New(num_series);
    for (int32_t index = 0; index < num_series; index++)
    {
        if (PyTuple_SetItem(
            py_series,
            index,
            convertSeriesToPyObject(verbosity, series[index])
            ) != 0
        ) {
            if (verbosity > SILENT)
            {
                fprintf(
                    stderr, 
                    "ConvertFigureToPyObject:\n"
                    "Error! Cannot set tuple item series %s.\n",
                    series[index].label
                );
                
                return NULL;
            }
        }
    }
    
    if ((py_series == NULL) && (verbosity > SILENT))
    {
		fprintf(
			stderr, 
			"ConvertFigureToPyObject:\n"
			"Error! Py series is NULL!\n"
		);
    }
    else if ((py_axes == NULL) && (verbosity > SILENT))
    {
        fprintf(
			stderr, 
			"Error! Py axes is NULL!\n"
		);
    }
    
    PyObject *tuple = Py_BuildValue("OO", py_axes, py_series);
    
    if ((tuple == NULL) && (verbosity > SILENT))
    {
        fprintf(
            stderr, 
            "ConvertFigureToPyObject:\n"
            "Error! Cannot build value!\n"
        );
    }
    
    return tuple;
}

int32_t plotFigure(
	const int32_t   verbosity,
    const figure_s  figure,
    const char     *output_file_path
    ) {
    
    int32_t error_code = 0;
    
    const char* output_directory_path = newPath(output_file_path).directory; 
        
	INIT_PY(
        if (checkCreateDirectory(verbosity, output_directory_path))
            {
            float  *output_array = NULL;

            const int32_t num_output_elements = 1;
            const int32_t num_args      = 3;
            PyObject *args[] = {
                PyLong_FromLong((long)verbosity),
                convertFigureToPyObject(verbosity, figure),
                PyUnicode_FromString(output_file_path),
                NULL
            };

            error_code += runPythonFunc(
                verbosity,
                num_args,
                "./py_tools/include/py_tools/py_tools", 
                "plot_figure", 
                args, 
                &output_array, 
                num_output_elements
            );
        }
        else
        {
            error_code = 10000;
        }
    );
     
	return error_code;
}

#endif