#ifndef IO_UNITS_H
#define IO_UNITS_H

typedef enum SI_Units{
    no_unit_e          =  1,
    second_e           =  2,
    metre_e            =  3,
    kilogram_e         =  5,
    ampere_e           =  7,
    kelvin_e           =  11,
    mole_e             =  13,
    candela_e          =  17
} si_unit_e;

typedef struct UnitValue{
    float      value;
    si_unit_e  units;
    si_unit_e  inverse_units;
} unit_value_e;


#endif