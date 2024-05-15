#ifndef CONSOLE_API_H
#define CONSOLE_API_H

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include <inttypes.h>
#include <time.h>

#include "io_tools/custom_types.h"

typedef struct timer 
{
    char  *name;

    struct timespec start;
    struct timespec stop;
} timer_s;

void start_timer(
    char    *name,
    timer_s *timer
    ) {
    
    timer->name = name;
    clock_gettime(CLOCK_REALTIME, &timer->start);
}

float stop_timer(
    timer_s *timer
    ) {
    
    clock_gettime(CLOCK_REALTIME, &timer->stop);
    float time_elapsed = (float)(timer->stop.tv_sec - timer->start.tv_sec)
        + (float)(timer->stop.tv_nsec - timer->start.tv_nsec) 
        / (float)1E9;
    clock_gettime(CLOCK_REALTIME, &timer->start);
    
    return time_elapsed;
}

void print_timer(
    const char *lap_name,
    timer_s    *timer
    ) {

    printf(
        "Timer: %s, Lap: %s. Reading: %f Seconds.\n", 
        timer->name, 
        lap_name, 
        stop_timer(timer)
    );
}


typedef struct loading_s {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Structure to hold loading bar incolumnion. 
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //

	int32_t check_interval, check_index, num_ticks;
	size_t  bar_length, max_eta_length;
	clock_t start_time, end_time;
	float   loop_time, total_time, avg_time;
	float   time_to_completion; 
    
    int32_t tick_index;

} loading_s;

loading_s setupLoadingConfig(
    const size_t  bar_length, 
    const int32_t check_interval,
    const int32_t num_ticks
) {

    loading_s loading = {
        .loop_time  = 0, .total_time = 0, .avg_time = 0,
        .check_index = 0, .time_to_completion = 0,
        .max_eta_length = 0,

        .bar_length     = bar_length,
        .num_ticks      = num_ticks,
        .check_interval = check_interval
    };
    
    return loading;
}

void printProgress(
    loading_s *loading
    ) {
        
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Function to print loading bar into console and update.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    
	const float     percent       = 
		((float) loading->tick_index / (float) loading->num_ticks);
    const int32_t   num_pips      = 
		(int32_t) floor(percent* (float) loading->bar_length);
	const int32_t   max_pips      = 
		(int32_t) loading->bar_length; 
	const char    * prefix        = "Progress: [";
	const char    * suffix        = "]";
    const char      pip           = '#';
	const size_t    prefix_length = strlen(prefix);
	const size_t    suffix_length = strlen(suffix);
	      char    * buffer        = 
		  calloc((size_t) max_pips 
		        + prefix_length 
				+ suffix_length 
			    + (size_t) 1, // +1 for \0
				sizeof(char)
			); 

	strcpy(buffer, prefix);
    
	for (int32_t index = 0; index < max_pips; index++) 
	{
		buffer[(int32_t) prefix_length + index] = 
            (index < num_pips) ? pip : ' ';
	}

	strcpy(&buffer[(int32_t) prefix_length + max_pips], suffix);

	int32_t days = 0, hours = 0, minutes = 0;
	int32_t seconds = (int32_t) loading->time_to_completion;

    char * estimated_time;
	if(seconds > 86400) 
	{
        minutes = seconds/60; 
        seconds = seconds%60; 
        hours = minutes/60; 
        minutes = minutes%60; 
        days = hours/24; 
        hours = days%24;
        
		asprintf(
            &estimated_time, 
            "ETA: %d:%d:%d:%d days:hours:minutes:seconds", 
            days, hours, minutes, seconds
        );
	} 
	else if (seconds > 3600) 
	{   
        minutes = seconds/60; 
        seconds = seconds%60; 
        hours = minutes/60; 
        minutes = minutes%60;
        
		asprintf(
            &estimated_time, 
            "ETA: %d:%d:%d hours:minutes:seconds", 
            hours, minutes, seconds
        );
	}
	else if (seconds > 60) 
	{
		minutes = seconds/60; 
        seconds = seconds%60;
        
		asprintf(
            &estimated_time, 
            "ETA: %d:%d minutes:seconds", 
            minutes, seconds
        );
	}
	else
	{    
		asprintf(
            &estimated_time, 
            "ETA: %d seconds", 
            seconds
        );
	}
    
    //Set max length of eta string to ensure to cover with spaces when printing over.
    size_t eta_length = strlen(estimated_time);
    if (eta_length > loading->max_eta_length) 
	{
        loading->max_eta_length = eta_length;
    }
    
	printf(
        "\b%c\r%s %.2f%%. %*s", 
        5, buffer, percent*100, 
        - (int32_t) loading->max_eta_length, 
        estimated_time
    );

	fflush(stdout);
	free(buffer); free(estimated_time);
}

void progressCheck(
    loading_s*   loading
    ) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Checks progress and prints loading bar.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
    
    const int32_t tick_index = loading->tick_index;
    
    if (
           tick_index % loading->check_interval == 0 
        && tick_index != 0
    ) { 
    
        if (loading->check_index != 0) { 

            loading->end_time            = clock();
            loading->loop_time           = 
                (float)(loading->end_time - loading->start_time) 
                / CLOCKS_PER_SEC;
            loading->total_time         += loading->loop_time;
            loading->avg_time            = 
                (float)loading->total_time / (float)tick_index;
            loading->time_to_completion  = 
                loading->avg_time
               *((float)loading->num_ticks - (float)loading->tick_index);

            printProgress(loading);
        }

        loading->start_time = clock();
        loading->check_index++;
    }
    
    loading->tick_index++;
} 

typedef struct TableColumn {
	  	
	      char*         header        ;
	      int32_t       decimal_places;
	      bool          bold  ;	
} table_column_s;

typedef struct Table {
	  
	const int32_t         num_rows   ;
	const int32_t         num_columns;
	const char*           title      ; 
	
	const uni_s          *cells      ;
	const table_column_s *columns    ;
} table_s;

typedef enum {top_e, mid_top_e, mid_e, bottom_e} table_pos_e; 

void printTableSeparator(
	const table_pos_e     pos,
	const int32_t         num_columns,
	const table_column_s *columns,
	const int32_t        *widths
	) {

	const char* left       [] = {"╔","╠","╠","╚"};
	const char* centre     [] = {"═","╦","╬","╩"};
	const char* right      [] = {"╗","╣","╣","╝"};
	
	const char* centre_thin[] = {"═","╤","╪","╧"};

	printf(left[pos]);
	
	for (int index = 0; index < num_columns; index++ ) 
	{
		for (int width_index = 0; width_index < widths[index]; width_index++ ) 
		{
			printf("═");
		}
		if (index != (num_columns-1)) 
		{
			bool bold = columns[index + 1].bold;
			printf(bold?centre[pos]:centre_thin[pos]);
		}
 	}
	
	printf(right[pos]);

	printf("\n");
}

char* convertCellToString(
    const uni_s   cell, 
    const int32_t width, 
    const int32_t decimal_places
    ) {
	
	char* string;
	
	switch(cell.type) 
	{			
		case(bool_e):
			asprintf(&string, " %*i "  , width, (int) cell.value.b);
		break;	
		
		case(int_e):
			asprintf(&string, " %*i "  , width, cell.value.i);
		break;
		
		case(float_e):
			asprintf(&string, " %*.*f ", width, decimal_places, cell.value.f);
		break;
		
		case(char_e):
			if (cell.value.c != '\0') 
			{
				asprintf(&string, " %*c ", width, cell.value.c);
			}
			else
			{
				asprintf(&string, " %*s ", width, " ");
			}
		break;
		
		case(string_e):
			if (strcmp(cell.value.s, "")) 
			{
				asprintf(&string, " %*s ", width, cell.value.s);
			}
			else
			{
				asprintf(&string, " %*s ", width, " ");
			}	
		break;
		
		default: 
			fprintf(stderr, "Warning! Type \"%i\" not recognised!", cell.type);
		break;
	}
	
	return string;
}

void printTable(
    const table_s table
    ) {

	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	//
	// Prints a table into the console.
	//
	// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ //
	
	//Assigning variables for readablity:
    const int32_t          num_rows     = table.num_rows;
    const int32_t          num_columns  = table.num_columns;
    const table_column_s * columns      = table.columns;

	//Calculate column required widths:
		
    int32_t* widths = malloc(sizeof(int) * (size_t) num_columns);
    for (
        int32_t column_index = 0; 
        column_index < num_columns; 
        column_index++
    ) {				
        widths[column_index] = 
            (int32_t) strlen(columns[column_index].header) + 2;
    }

    for (int32_t row_index = 0; row_index < num_rows; row_index++ ) 
    {
        for (
            int32_t column_index = 0; 
            column_index < num_columns; 
            column_index++
        ) {
            int32_t cell_index = row_index*num_columns + column_index;

            table_column_s column = columns[column_index]; 
            uni_s cell = table.cells[cell_index];
            widths[column_index] = 
                (int32_t) strlen(
                    convertCellToString(cell, widths[column_index] - 2, 
                    column.decimal_places)
                );
        }
    }
		
	//Calculate total table width:
    
    int32_t table_width = 1; //<-- Add width for last colum separator.

    for (int32_t column_index = 0; column_index < num_columns; column_index++) 
    {
        //Add extra width for separators/.
        table_width += widths[column_index] + 1;
    }
	
	//Print Spacing Before table:
	printf("\n");
	
	//Print table top:
	printTableSeparator(top_e, num_columns, columns, widths);

 	//Calculate centre of table for title start position:
    int32_t title_length = (int32_t) strlen(table.title); 
    int32_t title_start  = table_width - title_length - 2; 

    if (title_start < 0) 
    {
        title_start = 0;
    } 
    else 
    {
        title_start /= 2;
    }

	//Calculate remaining space needed to reach end of table: 
	int32_t title_end = table_width - title_start - title_length - 2;

	//Print title column: 
    printf("║");

    for (int32_t index = 0; index < (int32_t) title_start; index++ ) 
    {    
        printf(" ");
    }

    printf("%s", table.title);

    for (int32_t index = 0; index < (int32_t) title_end; index++ ) 
    {
        printf(" ");
    }

    printf("║\n");

 	//Print Title Separator:
	
	printTableSeparator(mid_top_e, num_columns, columns, widths);

	//Print headers:
    for (int32_t column_index = 0; column_index < num_columns; column_index++ ) 
    {

        table_column_s column = columns[column_index];

        bool bold = (( column.bold ) || ( column_index == 0 ));
        printf(bold?"║":"│");

        printf(" %*s ", widths[column_index] - 2, columns[column_index].header);
    }

		printf("║\n");

	//Print header separator:

		printTableSeparator(mid_e, num_columns, columns, widths);

	//Print data:

    for (int32_t row_index = 0; row_index < num_rows; row_index++) 
    {
        for (
            int32_t column_index = 0; 
            column_index < num_columns; 
            column_index++     
        ) {
            table_column_s column = columns[column_index];
            int32_t cell_index = row_index*num_columns + column_index;

            bool bold = (( column.bold ) || ( column_index == 0 ));
            printf(bold?"║":"│");

            int32_t   width          = widths[column_index] - 2;
            int32_t   decimal_places = column.decimal_places;
            uni_s cell           = table.cells[cell_index];

            printf(convertCellToString(cell, width, decimal_places));					
        }

        printf("║\n");
    }

	//Print table bottom:

		printTableSeparator(bottom_e, num_columns, columns, widths);
	
	//Print Spacing post table:
		
		printf("\n");
}

void printTitle(
    const char    *title, 
    const int32_t  width
    ) {

	//Calculate centre of table for title start position:
    const int32_t title_length = (int32_t) strlen(title) + 4; 
    const int32_t title_start  = 
        (title_length >= 0) ? (width - (int32_t) title_length) / 2 : 0;

	//Calculate remaining space needed to reach end of table: 
	const int32_t title_end = width - title_start - title_length;

	//Print title column: 
    printf("*");

    for (int32_t index = 0; index < title_start; index++ ) {

        printf("~");
    }

    printf("* %s *", title);

    for (int32_t index = 0; index < title_end; index++ ) {

        printf("~");
    }

    printf("*\n");
}

void printArray(
    const char   *title, 
    const float  *array,
    const int32_t num_elements
    ) {
	
	printf("%s: [", title);
	for (int32_t index = 0; index < num_elements; index++)
	{
		printf("%f,", (float) array[index]);
	}
	printf("] \n");
}

void printArrayE(
    const  char   *title , 
    const  float  *array , 
    const  int32_t num_elements
    ) {
	
	printf("%s: [", title);
	for (int32_t index = 0; index < num_elements; index++)
	{
		printf(" %.4e,", (float) array[index]);
	}
	printf("] \n");
}

void printArrayBool(
    const char    *title, 
    const bool    *array,
    const int32_t  num_elements
    ) {
	
	printf("%s: [", title);
	for (int32_t index = 0; index < num_elements; index++)
	{
		printf(" %i,", (int32_t) array[index]);
	}
	printf("] \n");
}

void printArrayInt(
    const char    *title, 
    const int32_t *array, 
    const int32_t  num_elements
    ) {
	
	printf("%s: [", title);
	for (int32_t index = 0; index < num_elements; index++)
	{
		printf(" %i,", (int32_t) array[index]);
	}
	printf("] \n");
}

void printArrayUInt16(
    const char     *title, 
    const uint16_t *array, 
    const int32_t   num_elements
    ) {
	
	printf("%s: [", title);
	for (int32_t index = 0; index < num_elements; index++) 
	{
		printf(" %u,", (uint16_t) array[index]);
	}
	printf("] \n");
}

void printArrayStrings(
    const char     *title, 
          char    **array, 
    const int32_t   num_elements
    ) {
	
	printf("%s: [", title);
	for (int32_t index = 0; index < num_elements; index++) 
	{
		printf(" %s,", array[index]);
	}
	printf("] \n");
}

void printBits(
    const unsigned int number, 
    const size_t       size
    ) {
		
	unsigned int mask = (unsigned int) 1<<(size * 8-1);
	while (mask) 
	{
		printf("%u ", number&mask ? 1 : 0);
		mask >>= 1;
	}
	
	printf("\n");
}

#endif