#ifndef PHENOM_DATA_H
#define PHENOM_DATA_H

/***************************** PhenomD static constants*******************************/

// Dimensionless frequency (Mf) at which define the end of the waveform:
#define f_CUT 0.2f

// Dimensionless frequency (Mf) at which the inspiral amplitude switches to the 
// intermediate amplitude:
#define AMPLITUDE_INTERMEDIATE_START_FREQUENCY 0.014f

// Minimal final spin value below which the waveform might behave pathological
// because the ISCO frequency is too low. For more details, see the review wiki
// page https://www.lsc-group.phys.uwm.edu/ligovirgo/cbcnote/WaveformsReview/IMRPhenomDCodeReview/PhenD_LargeNegativeSpins:
#define MIN_FINAL_SPIN -0.717f

// Dimensionless frequency (Mf) at which the inspiral phase
// switches to the intermediate phase:
#define PHASE_INTERMEDIATE_START_FREQUENCY 0.018f

//PI^(-1/6) [http://oeis.org/A093207]:
#define PI_M_SIXTH cbrtf((float)M_PI)/sqrtf((float)M_PI)

//PI^(1/3)
#define CUBE_ROOT_PI cbrtf((float)M_PI)
#define PI_SQUARED (float)(M_PI*M_PI)

// Powers of pi
#define PI_POWER_ONE          (float) M_PI

#define PI_POWER_TWO          (PI_POWER_ONE*PI_POWER_ONE)
#define PI_POWER_SIXTH        (sqrtf(PI_POWER_ONE)/cbrtf(PI_POWER_ONE))
#define PI_POWER_ONE_THIRD    PI_POWER_SIXTH*PI_POWER_SIXTH
#define PI_POWER_TWO_THIRDS   PI_POWER_ONE_THIRD*PI_POWER_ONE_THIRD
#define PI_POWER_FOUR_THIRDS  PI_POWER_ONE*PI_POWER_ONE_THIRD
#define PI_POWER_FIVE_THIRDS  PI_POWER_FOUR_THIRDS*PI_POWER_ONE_THIRD
#define PI_POWER_SEVEN_THIRDS PI_POWER_ONE_THIRD*PI_POWER_TWO
#define PI_POWER_EIGHT_THIRDS PI_POWER_TWO_THIRD*PI_POWER_TWO

#define PI_POWER_MINUS_ONE          (1.0f/PI_POWER_ONE)
#define PI_POWER_MINUS_SIXTH        (1.0f/PI_POWER_SIXTH)
#define PI_POWER_MINUS_SEVEN_SIXTHS PI_POWER_MINUS_ONE*PI_POWER_MINUS_SIXTH
#define PI_POWER_MINUS_ONE_THIRD    PI_POWER_MINUS_SIXTH*PI_POWER_MINUS_SIXTH
#define PI_POWER_MINUS_TWO_THIRDS   PI_POWER_MINUS_ONE_THIRD*PI_POWER_MINUS_ONE_THIRD
#define PI_POWER_MINUS_FIVE_THIRDS  PI_POWER_MINUS_ONE*PI_POWER_MINUS_TWO_THIRDS

#define TWO_TIMES_SQRT_PI_OVER_FIVE 2.0f*sqrtf((float)M_PI/5.0f)

/************************ Amplitude Coefficient Terms *************************/

//////////////// Amplitude: Inspiral Coefficient Terms /////////////////////////

// Phenom coefficient terms rho1, ..., rho3 from direct fit
// AmpInsDFFitCoeffChiPNFunc[eta, chiPN]

#define NUM_AMPLITUDE_INSPIRAL_COEFFICIENTS 3

// Rho coefficient terms. See corresponding row in Table 5 arXiv:1508.07253:

#ifdef __cplusplus // Only used in gpu code
static const float AMPLITUDE_INSPIRAL_COEFFICIENT_TERMS_1[] = 
{
      3931.8979897196696f, -17395.758706812805f  ,   3132.375545898835f,
    343965.86092361377f  , -1.2162565819981997e6f, -70698.00600428853f ,
    1.383907177859705e6f , -3.9662761890979446e6f, -60017.52423652596f ,
    803515.1181825735f   , -2.091710365941658e6f 
};
static const float AMPLITUDE_INSPIRAL_COEFFICIENT_TERMS_2[] = 
{
    -40105.47653771657f   , 112253.0169706701f   ,  23561.696065836168f,
    -3.476180699403351e6f , 1.137593670849482e7f , 754313.1127166454f,
    -1.308476044625268e7f , 3.6444584853928134e7f, 596226.612472288f,
    -7.4277901143564405e6f, 1.8928977514040343e7f
};
static const float AMPLITUDE_INSPIRAL_COEFFICIENT_TERMS_3[] = 
{
    83208.35471266537f   , -191237.7264145924f   , -210916.2454782992f   ,  
    8.71797508352568e6f  , -2.6914942420669552e7f, -1.9889806527362722e6f, 
    3.0888029960154563e7f, -8.390870279256162e7f , -1.4535031953446497e6f,
    1.7063528990822166e7f, -4.2748659731120914e7f
};
static const float *AMPLITUDE_INSPIRAL_COEFFICIENT_TERMS[] =
{
    AMPLITUDE_INSPIRAL_COEFFICIENT_TERMS_1,
    AMPLITUDE_INSPIRAL_COEFFICIENT_TERMS_2,
    AMPLITUDE_INSPIRAL_COEFFICIENT_TERMS_3
};
#endif

///////////////// Amplitude: Merger-Ringdown Coefficient Terms /////////////////

// Phenom coefficient terms gamma1, ..., gamma3
// AmpMRDAnsatzFunc[]

#define NUM_AMPLITUDE_MERGER_RINGDOWN_COEFFICIENTS 3

// Gamma coefficient terms. See corresponding rows in Table 5 arXiv:1508.07253:

#ifdef __cplusplus // Only used in gpu code
static const float AMPLITUDE_MERGER_RINGDOWN_COEFFICIENT_TERMS_1[] = 
{
     0.006927402739328343f, 0.03020474290328911f, 0.006308024337706171f , 
    -0.12074130661131138f , 0.26271598905781324f, 0.0034151773647198794f,
    -0.10779338611188374f , 0.27098966966891747f, 0.0007374185938559283f,
    -0.02749621038376281f , 0.0733150789135702f
};
static const float AMPLITUDE_MERGER_RINGDOWN_COEFFICIENT_TERMS_2[] = 
{
     1.010344404799477f ,  0.0008993122007234548f, 0.283949116804459f  , 
    -4.049752962958005f , 13.207828172665366f    , 0.10396278486805426f, 
    -7.025059158961947f , 24.784892370130475f    , 0.03093202475605892f,
    -2.6924023896851663f,  9.609374464684983f
};
static const float AMPLITUDE_MERGER_RINGDOWN_COEFFICIENT_TERMS_3[] = 
{
     1.3081615607036106f , -0.005537729694807678f, -0.06782917938621007f ,
    -0.6689834970767117f ,  3.403147966134083f   , -0.05296577374411866f ,
    -0.9923793203111362f ,  4.820681208409587f   , -0.006134139870393713f,
    -0.38429253308696365f,  1.756175442198598f
};
static const float *AMPLITUDE_MERGER_RINGDOWN_COEFFICIENT_TERMS[] =
{
    AMPLITUDE_MERGER_RINGDOWN_COEFFICIENT_TERMS_1,
    AMPLITUDE_MERGER_RINGDOWN_COEFFICIENT_TERMS_2,
    AMPLITUDE_MERGER_RINGDOWN_COEFFICIENT_TERMS_3
};
#endif

////////////////// Amplitude: Intermediate coefficient terms ///////////////////

// Amplitude Intermediate Collocation Fit coefficient terms.

// This is the 'v2' value in Table 5 of arXiv:1508.07253:

#ifdef __cplusplus // Only used in gpu code
static const float AMPLITUDE_INTERMEDIATE_COLLOCATION_FIT_COEFFICIENT_TERMS[] = 
{
     0.8149838730507785f, 2.5747553517454658f, 1.1610198035496786f, 
    -2.3627771785551537f, 6.771038707057573f , 0.7570782938606834f,
    -2.7256896890432474f, 7.1140380397149965f, 0.1766934149293479f,
    -0.7978690983168183f, 2.1162391502005153f
};
#endif


/****************************** Phase functions *******************************/
 
///////////////////// Phase: Ringdown Coeffecient Terms ////////////////////////
 
// Phenom coefficient terms alpha_1...alpha_5 are the phenomenological 
// intermediate coefficient terms depending on eta and chiPN PhiRingdownAnsatz 
// is the ringdown phasing in terms of the alpha_i coefficients

// Alpha 1 phenom coefficient terms. See corresponding row in Table 5 
// rXiv:1508.07253:

#define NUM_PHASE_MERGER_RINGDOWN_COEFFICIENTS 5

#ifdef __cplusplus // Only used in gpu code
static const float PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_1[] = 
{
      43.31514709695348f,   638.6332679188081f, -32.85768747216059f ,
    2415.8938269370315f , -5766.875169379177f , -61.85459307173841f ,
    2953.967762459948f  , -8986.29057591497f  , -21.571435779762044f,
     981.2158224673428f , -3239.5664895930286f
};
static const float PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_2[] = 
{
   -0.07020209449091723f, -0.16269798450687084f, -0.1872514685185499f  ,
    1.138313650449945f  , -2.8334196304430046f , -0.17137955686840617f , 
    1.7197549338119527f , -4.539717148261272f  , -0.049983437357548705f,
    0.6062072055948309f , -1.682769616644546f
};
static const float PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_3[] = 
{
        9.5988072383479f, -397.05438595557433f, 16.202126189517813f,
    -1574.8286986717037f, 3600.3410843831093f , 27.092429659075467f,
    -1786.482357315139f , 5152.919378666511f  , 11.175710130033895f,
     -577.7999423177481f, 1808.730762932043f
};
static const float PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_4[] = 
{
    -0.02989487384493607f, 1.4022106448583738f , -0.07356049468633846f ,
     0.8337006542278661f , 0.2240008282397391f , -0.055202870001177226f,
     0.5667186343606578f , 0.7186931973380503f , -0.015507437354325743f, 
     0.15750322779277187f, 0.21076815715176228f
};
static const float PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_5[] = 
{
    0.9974408278363099f, -0.007884449714907203f, -0.059046901195591035f, 
    1.3958712396764088f, -4.516631601676276f   , -0.05585343136869692f ,
    1.7516580039343603f, -5.990208965347804f   , -0.017945336522161195f,
    0.5965097794825992f, -2.0608879367971804f
};
static const float *PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS[] =
{
    PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_1,
    PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_2,
    PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_3,
    PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_4,
    PHASE_MERGER_RINGDOWN_COEFFICIENT_TERMS_5
};
#endif
 
///////////////////// Phase: Intermediate Coefficient Terms ////////////////////
 
// Phenom coefficient terms beta_1...beta_3 are the phenomenological 
// intermediate coefficient terms depending on eta and chiPN PhiIntAnsatz is the 
// intermediate phasing in terms of the beta_i coeffecients \[Beta]1Fit = 
// PhiIntFitCoeff\[Chi]PNFunc[\[Eta], \[Chi]PN][[1]] Beta phenom coefficient 
// terms. See corresponding row in Table 5 arXiv:1508.07253:

#define NUM_PHASE_INTERMEDIATE_COEFFICIENTS 3

#ifdef __cplusplus // Only used in gpu code
static const float PHASE_INTERMEDIATE_COEFFICIENT_TERMS_1[] = 
{
       97.89747327985583f,  -42.659730877489224f, 153.48421037904913f,
    -1417.0620760768954f , 2752.8614143665027f  , 138.7406469558649f ,
    -1433.6585075135881f , 2857.7418952430758f  , 41.025109467376126f, 
     -423.680737974639f  ,  850.3594335657173f  
};
static const float PHASE_INTERMEDIATE_COEFFICIENT_TERMS_2[] = 
{
    -3.282701958759534f,   -9.051384468245866f, -12.415449742258042f ,
    55.4716447709787f  , -106.05109938966335f , -11.953044553690658f ,
    76.80704618365418f , -155.33172948098394f ,  -3.4129261592393263f,
    25.572377569952536f,  -54.408036707740465f
};
static const float PHASE_INTERMEDIATE_COEFFICIENT_TERMS_3[] = 
{
    -0.000025156429818799565f, 0.000019750256942201327f, -0.000018370671469295915f,
     0.000021886317041311973f, 0.00008250240316860033f ,  7.157371250566708e-6f   ,
    -0.000055780000112270685f, 0.00019142082884072178f ,  5.447166261464217e-6f   ,
    -0.00003220610095021982f ,  0.00007974016714984341f
};
static const float *PHASE_INTERMEDIATE_COEFFICIENT_TERMS[] =
{
    PHASE_INTERMEDIATE_COEFFICIENT_TERMS_1,
    PHASE_INTERMEDIATE_COEFFICIENT_TERMS_2,
    PHASE_INTERMEDIATE_COEFFICIENT_TERMS_3
};
#endif



////////////////////// Phase: Inspiral Coefficient Terms //////////////////////
 
// sigma_i i=1,2,3,4 are the phenomenological inspiral coefficient terms 
// depending on eta and chiPN PhiInsAnsatzInt is a souped up Tf[1][1]phasing which 
// depends on the sigma_i coefficients.

#define NUM_PHASE_INSPIRAL_COEFFICIENTS 4

#ifdef __cplusplus // Only used in gpu code
// Sigma coefficient terms. See corresponding row in Table 5 arXiv:1508.07253:
static const float PHASE_INSPIRAL_COEFFICIENT_TERMS_1[] = 
{
      2096.551999295543f,    1463.7493168261553f, 1312.5493286098522f , 
     18307.330017082117f,  -43534.1440746107f   , -833.2889543511114f , 
     32047.31997183187f , -108609.45037520859f  ,  452.25136398112204f, 
      8353.439546391714f,  -44531.3250037322f
};
static const float PHASE_INSPIRAL_COEFFICIENT_TERMS_2[] = 
{
    -10114.056472621156f, -44631.01109458185f  , -6541.308761668722f ,
    -266959.23419307504f, 686328.3229317984f   ,  3405.6372187679685f,
    -437507.7208209015f , 1.6318171307344697e6f, -7462.648563007646f , 
    -114585.25177153319f, 674402.4689098676f
};
static const float PHASE_INSPIRAL_COEFFICIENT_TERMS_3[] = 
{
    22933.658273436497f  , 230960.00814979506f   , 14961.083974183695f,
    1.1940181342318142e6f, -3.1042239693052764e6f, -3038.166617199259f,
    1.8720322849093592e6f, -7.309145012085539e6f , 42738.22871475411f , 
    467502.018616601f    , -3.064853498512499e6f
};
static const float PHASE_INSPIRAL_COEFFICIENT_TERMS_4[] = 
{
    -14621.71522218357f   , -377812.8579387104f   ,  -9608.682631509726f,
    -1.7108925257214056e6f, 4.332924601416521e6f  , -22366.683262266528f,
    -2.5019716386377467e6f, 1.0274495902259542e7f , -85360.30079034246f ,
    -570025.3441737515f   , + 4.396844346849777e6f,
};
static const float *PHASE_INSPIRAL_COEFFICIENT_TERMS[] =
{
    PHASE_INSPIRAL_COEFFICIENT_TERMS_1,
    PHASE_INSPIRAL_COEFFICIENT_TERMS_2,
    PHASE_INSPIRAL_COEFFICIENT_TERMS_3,
    PHASE_INSPIRAL_COEFFICIENT_TERMS_4
};
#endif

/************************ Delta Terms *************************/

// The following terms DELTA_0_TERMS ...DELTA_4_TERMS were derived in 
// mathematica according to the constraints detailed in arXiv:1508.07253, 
// section 'Region IIa - intermediate'. These are not given in the paper. Can be 
// rederived by solving Equation 21 for the constraints given in Equations 22-26 
// in arXiv:1508.07253:

#define NUM_AMPLITUDE_INTERMEDIATE_TERMS 5

#define NUM_AMPLITUDE_INTERMEDIATE_TERMS_0 \
    (                                  \
            d2*f[0][5]*f[1][2]*f[2][1] \
    -  2.0f*d2*f[0][4]*f[1][3]*f[2][1] \
    +       d2*f[0][3]*f[1][4]*f[2][1] \
    -       d2*f[0][5]*f[1][1]*f[2][2] \
    +       d2*f[0][4]*f[1][2]*f[2][2] \
    -       d1*f[0][3]*f[1][3]*f[2][2] \
    +       d2*f[0][3]*f[1][3]*f[2][2] \
    +       d1*f[0][2]*f[1][4]*f[2][2] \
    -       d2*f[0][2]*f[1][4]*f[2][2] \
    +       d2*f[0][4]*f[1][1]*f[2][3] \
    +  2.0f*d1*f[0][3]*f[1][2]*f[2][3] \
    -  2.0f*d2*f[0][3]*f[1][2]*f[2][3] \
    -       d1*f[0][2]*f[1][3]*f[2][3] \
    +       d2*f[0][2]*f[1][3]*f[2][3] \
    -       d1*f[0][1]*f[1][4]*f[2][3] \
    -       d1*f[0][3]*f[1][1]*f[2][4] \
    -       d1*f[0][2]*f[1][2]*f[2][4] \
    +  2.0f*d1*f[0][1]*f[1][3]*f[2][4] \
    +       d1*f[0][2]*f[1][1]*f[2][5] \
    -       d1*f[0][1]*f[1][2]*f[2][5] \
                                       \
    +  4.0f*v1*f[0][2]*f[1][3]*f[2][2] \
    -  3.0f*v1*f[0][1]*f[1][4]*f[2][2] \
    -  8.0f*v1*f[0][2]*f[1][2]*f[2][3] \
    +  4.0f*v1*f[0][1]*f[1][3]*f[2][3] \
    +       v1        *f[1][4]*f[2][3] \
    +  4.0f*v1*f[0][2]*f[1][1]*f[2][4] \
    +       v1*f[0][1]*f[1][2]*f[2][4] \
    -  2.0f*v1        *f[1][3]*f[2][4] \
    -  2.0f*v1*f[0][1]*f[1][1]*f[2][5] \
    +       v1        *f[1][2]*f[2][5] \
    -       v2*f[0][5]        *f[2][2] \
    +  3.0f*v2*f[0][4]        *f[2][3] \
    -  3.0f*v2*f[0][3]        *f[2][4] \
    +       v2*f[0][2]        *f[2][5] \
    -       v3*f[0][5]*f[1][2]         \
    +  2.0f*v3*f[0][4]*f[1][3]         \
    -       v3*f[0][3]*f[1][4]         \
    +  2.0f*v3*f[0][5]*f[1][1]*f[2][1] \
    -       v3*f[0][4]*f[1][2]*f[2][1] \
    -  4.0f*v3*f[0][3]*f[1][3]*f[2][1] \
    +  3.0f*v3*f[0][2]*f[1][4]*f[2][1] \
    -  4.0f*v3*f[0][4]*f[1][1]*f[2][2] \
    +  8.0f*v3*f[0][3]*f[1][2]*f[2][2] \
    -  4.0f*v3*f[0][2]*f[1][3]*f[2][2] \
    )

#define NUM_AMPLITUDE_INTERMEDIATE_TERMS_1 \
    (                                  \
    -       d2*f[0][5]*f[1][2]         \
    +  2.0f*d2*f[0][4]*f[1][3]         \
    -       d2*f[0][3]*f[1][4]         \
    -       d2*f[0][4]*f[1][2]*f[2][1] \
    +  2.0f*d1*f[0][3]*f[1][3]*f[2][1] \
    +  2.0f*d2*f[0][3]*f[1][3]*f[2][1] \
    -  2.0f*d1*f[0][2]*f[1][4]*f[2][1] \
    -       d2*f[0][2]*f[1][4]*f[2][1] \
    +       d2*f[0][5]        *f[2][2] \
    -  3.0f*d1*f[0][3]*f[1][2]*f[2][2] \
    -       d2*f[0][3]*f[1][2]*f[2][2] \
    +  2.0f*d1*f[0][2]*f[1][3]*f[2][2] \
    -  2.0f*d2*f[0][2]*f[1][3]*f[2][2] \
    +       d1*f[0][1]*f[1][4]*f[2][2] \
    +  2.0f*d2*f[0][1]*f[1][4]*f[2][2] \
    -       d2*f[0][4]        *f[2][3] \
    +       d1*f[0][2]*f[1][2]*f[2][3] \
    +  3.0f*d2*f[0][2]*f[1][2]*f[2][3] \
    -  2.0f*d1*f[0][1]*f[1][3]*f[2][3] \
    -  2.0f*d2*f[0][1]*f[1][3]*f[2][3] \
    +       d1        *f[1][4]*f[2][3] \
    +       d1*f[0][3]        *f[2][4] \
    +       d1*f[0][1]*f[1][2]*f[2][4] \
    -  2.0f*d1        *f[1][3]*f[2][4] \
    -       d1*f[0][2]        *f[2][5] \
    +       d1        *f[1][2]*f[2][5] \
                                       \
    -  8.0f*v1*f[0][2]*f[1][3]*f[2][1] \
    +  6.0f*v1*f[0][1]*f[1][4]*f[2][1] \
    + 12.0f*v1*f[0][2]*f[1][2]*f[2][2] \
    -  8.0f*v1*f[0][1]*f[1][3]*f[2][2] \
    -  4.0f*v1*f[0][2]        *f[2][4] \
    +  2.0f*v1*f[0][1]        *f[2][5] \
    +  2.0f*v2*f[0][5]        *f[2][1] \
    -  4.0f*v2*f[0][4]        *f[2][2] \
    +  4.0f*v2*f[0][2]        *f[2][4] \
    -  2.0f*v2*f[0][1]        *f[2][5] \
    -  2.0f*v3*f[0][5]        *f[2][1] \
    +  8.0f*v3*f[0][2]*f[1][3]*f[2][1] \
    -  6.0f*v3*f[0][1]*f[1][4]*f[2][1] \
    +  4.0f*v3*f[0][4]        *f[2][2] \
    - 12.0f*v3*f[0][2]*f[1][2]*f[2][2] \
    +  8.0f*v3*f[0][1]*f[1][3]*f[2][2] \
    )

#define NUM_AMPLITUDE_INTERMEDIATE_TERMS_2 \
    (                                  \
            d2*f[0][5]*f[1][1]         \
    -       d1*f[0][3]*f[1][3]         \
    -  3.0f*d2*f[0][3]*f[1][3]         \
    +       d1*f[0][2]*f[1][4]         \
    +  2.0f*d2*f[0][2]*f[1][4]         \
    -       d2*f[0][5]        *f[2][1] \
    +       d2*f[0][4]*f[1][1]*f[2][1] \
    -       d1*f[0][2]*f[1][3]*f[2][1] \
    +       d2*f[0][2]*f[1][3]*f[2][1] \
    +       d1*f[0][1]*f[1][4]*f[2][1] \
    -       d2*f[0][1]*f[1][4]*f[2][1] \
    -       d2*f[0][4]        *f[2][2] \
    +  3.0f*d1*f[0][3]*f[1][1]*f[2][2] \
    +       d2*f[0][3]*f[1][1]*f[2][2] \
    -       d1*f[0][1]*f[1][3]*f[2][2] \
    +       d2*f[0][1]*f[1][3]*f[2][2] \
    -  2.0f*d1        *f[1][4]*f[2][2] \
    -       d2        *f[1][4]*f[2][2] \
    -  2.0f*d1*f[0][3]        *f[2][3] \
    +  2.0f*d2*f[0][3]        *f[2][3] \
    -       d1*f[0][2]*f[1][1]*f[2][3] \
    -  3.0f*d2*f[0][2]*f[1][1]*f[2][3] \
    +  3.0f*d1        *f[1][3]*f[2][3] \
    +       d2        *f[1][3]*f[2][3] \
    +       d1*f[0][2]        *f[2][4] \
    -       d1*f[0][1]*f[1][1]*f[2][4] \
    +       d1*f[0][1]        *f[2][5] \
    -       d1*f[1][1]        *f[2][5] \
                                       \
    +  4.0f*v1*f[0][2]*f[1][3]         \
    -  3.0f*v1*f[0][1]*f[1][4]         \
    +  4.0f*v1*f[0][1]*f[1][3]*f[2][1] \
    -  3.0f*v1        *f[1][4]*f[2][1] \
    - 12.0f*v1*f[0][2]*f[1][1]*f[2][2] \
    +  4.0f*v1        *f[1][3]*f[2][2] \
    +  8.0f*v1*f[0][2]        *f[2][3] \
    -       v1*f[0][1]        *f[2][4] \
    -       v1                *f[2][5] \
    -       v2*f[0][5]                 \
    -       v2*f[0][4]        *f[2][1] \
    +  8.0f*v2*f[0][3]        *f[2][2] \
    -  8.0f*v2*f[0][2]        *f[2][3] \
    +       v2*f[0][1]        *f[2][4] \
    +       v2                *f[2][5] \
    +       v3*f[0][5]                 \
    -  4.0f*v3*f[0][2]*f[1][3]         \
    +  3.0f*v3*f[0][1]*f[1][4]         \
    +       v3*f[0][4]        *f[2][1] \
    -  4.0f*v3*f[0][1]*f[1][3]*f[2][1] \
    +  3.0f*v3        *f[1][4]*f[2][1] \
    -  8.0f*v3*f[0][3]        *f[2][2] \
    + 12.0f*v3*f[0][2]*f[1][1]*f[2][2] \
    -     4*v3        *f[1][3]*f[2][2] \
    )

#define NUM_AMPLITUDE_INTERMEDIATE_TERMS_3  \
    (                                  \
    -  2.0f*d2*f[0][4]*f[1][1]         \
    +    d1*f[0][3]*f[1][2]            \
    +  3.0f*d2*f[0][3]*f[1][2]         \
    -    d1*f[0][1]*f[1][4]            \
    -    d2*f[0][1]*f[1][4]            \
    +  2.0f*d2*f[0][4]        *f[2][1] \
    -  2.0f*d1*f[0][3]*f[1][1]*f[2][1] \
    -  2.0f*d2*f[0][3]*f[1][1]*f[2][1] \
    +       d1*f[0][2]*f[1][2]*f[2][1] \
    -       d2*f[0][2]*f[1][2]*f[2][1] \
    +       d1        *f[1][4]*f[2][1] \
    +       d2        *f[1][4]*f[2][1] \
    +       d1*f[0][3]        *f[2][2] \
    -       d2*f[0][3]        *f[2][2] \
    -  2.0f*d1*f[0][2]*f[1][1]*f[2][2] \
    +  2.0f*d2*f[0][2]*f[1][1]*f[2][2] \
    +       d1*f[0][1]*f[1][2]*f[2][2] \
    -       d2*f[0][1]*f[1][2]*f[2][2] \
    +       d1*f[0][2]        *f[2][3] \
    -       d2*f[0][2]        *f[2][3] \
    +  2.0f*d1*f[0][1]*f[1][1]*f[2][3] \
    +  2.0f*d2*f[0][1]*f[1][1]*f[2][3] \
    -  3.0f*d1        *f[1][2]*f[2][3] \
    -       d2        *f[1][2]*f[2][3] \
    -  2.0f*d1*f[0][1]        *f[2][4] \
    +  2.0f*d1*f[1][1]        *f[2][4] \
                                       \
    -  4.0f*v1*f[0][2]*f[1][2]         \
    +  2.0f*v1        *f[1][4]         \
    +  8.0f*v1*f[0][2]*f[1][1]*f[2][1] \
    -  4.0f*v1*f[0][1]*f[1][2]*f[2][1] \
    -  4.0f*v1*f[0][2]        *f[2][2] \
    +  8.0f*v1*f[0][1]*f[1][1]*f[2][2] \
    -  4.0f*v1        *f[1][2]*f[2][2] \
    -  4.0f*v1*f[0][1]        *f[2][3] \
    +  2.0f*v1                *f[2][4] \
    +  2.0f*v2*f[0][4]                 \
    -  4.0f*v2*f[0][3]        *f[2][1] \
    +  4.0f*v2*f[0][1]        *f[2][3] \
    -  2.0f*v2                *f[2][4] \
    -  2.0f*v3*f[0][4]                 \
    +  4.0f*v3*f[0][2]*f[1][2]         \
    -  2.0f*v3     *f[1][4]            \
    +  4.0f*v3*f[0][3]        *f[2][1] \
    -  8.0f*v3*f[0][2]*f[1][1]*f[2][1] \
    +  4.0f*v3*f[0][1]*f[1][2]*f[2][1] \
    +  4.0f*v3*f[0][2]        *f[2][2] \
    -  8.0f*v3*f[0][1]*f[1][1]*f[2][2] \
    +  4.0f*v3        *f[1][2]*f[2][2] \
    )                   

#define NUM_AMPLITUDE_INTERMEDIATE_TERMS_4 \
    (                                  \
            d2*f[0][3]*f[1][1]         \
    -       d1*f[0][2]*f[1][2]         \
    -  2.0f*d2*f[0][2]*f[1][2]         \
    +       d1*f[0][1]*f[1][3]         \
    +       d2*f[0][1]*f[1][3]         \
    -       d2*f[0][3]        *f[2][1] \
    +  2.0f*d1*f[0][2]*f[1][1]*f[2][1] \
    +       d2*f[0][2]*f[1][1]*f[2][1] \
    -       d1*f[0][1]*f[1][2]*f[2][1] \
    +       d2*f[0][1]*f[1][2]*f[2][1] \
    -       d1        *f[1][3]*f[2][1] \
    -       d2        *f[1][3]*f[2][1] \
    -       d1*f[0][2]        *f[2][2] \
    +       d2*f[0][2]        *f[2][2] \
    -       d1*f[0][1]*f[1][1]*f[2][2] \
    -  2.0f*d2*f[0][1]*f[1][1]*f[2][2] \
    +  2.0f*d1        *f[1][2]*f[2][2] \
    +       d2        *f[1][2]*f[2][2] \
    +       d1*f[0][1]        *f[2][3] \
    -       d1        *f[1][1]*f[2][3] \
                                       \
    +  3.0f*v1*f[0][1]*f[1][2]         \
    -  2.0f*v1        *f[1][3]         \
    -  6.0f*v1*f[0][1]*f[1][1]*f[2][1] \
    +  3.0f*v1        *f[1][2]*f[2][1] \
    +  3.0f*v1*f[0][1]        *f[2][2] \
    -    v1                   *f[2][3] \
    -    v2*f[0][3]                    \
    +  3.0f*v2*f[0][2]        *f[2][1] \
    -  3.0f*v2*f[0][1]        *f[2][2] \
    +       v2                *f[2][3] \
    +       v3*f[0][3]                 \
    -  3.0f*v3*f[0][1]*f[1][2]         \
    +  2.0f*v3        *f[1][3]         \
    -  3.0f*v3*f[0][2]        *f[2][1] \
    +  6.0f*v3*f[0][1]*f[1][1]*f[2][1] \
    -  3.0f*v3        *f[1][2]*f[2][1] \
    )

// Other tables:
static const size_t QNMData_length = 1003u;

static const double QNMData_a[] = {
    #include "qnm_data_a.csv" 
};

static const double QNMData_fring[] = {
    #include "qnm_ringdown_frequency_real.csv"
};

static const double QNMData_fdamp[] = {
    #include "qnm_ringdown_frequency_imaginary.csv"
};

#endif