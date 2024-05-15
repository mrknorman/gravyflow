#ifndef ZOMBIE_PHENOM_H
#define ZOMBIE_PHENOM_H

#include <lal/LALDict.h>
#include <lal/LALSimInspiral.h>

 static double zombieFixReferenceFrequency(const double f_ref, const double f_min, const Approximant approximant)
 {
     if (f_ref == 0)
         switch (approximant) {
         case SpinTaylorT1:
         case SpinTaylorT5:
         case SpinTaylorT3:
         case SpinTaylorT4:
         case SpinTaylorT5Fourier:
         case SpinTaylorT4Fourier:
         case SpinTaylorF2:
         case IMRPhenomP:
         case IMRPhenomPv2:
         case IMRPhenomPv3:
         case IMRPhenomPv3HM:
         case IMRPhenomPv2_NRTidal:
         case IMRPhenomPv2_NRTidalv2:
                         return f_min;
                 case IMRPhenomXP:
                 case IMRPhenomXPHM:
         case NRSur4d2s:
         case IMRPhenomT:
         case IMRPhenomTHM:
         case IMRPhenomTP:
         case IMRPhenomTPHM:
         case TEOBResumS:
             return f_min;
         default:
             break;
         }
     return f_ref;
 }

const LALUnit lalDimensionlessUnit = { 0, { 0, 0, 0, 0, 0, 0, 0}, { 0, 0, 0, 0, 0, 0, 0} };

int ZombieInspiralFD(
     COMPLEX16FrequencySeries **hptilde,     /**< FD plus polarization */
     COMPLEX16FrequencySeries **hctilde,     /**< FD cross polarization */
     REAL8 m1,                               /**< mass of companion 1 (kg) */
     REAL8 m2,                               /**< mass of companion 2 (kg) */
     REAL8 S1x,                              /**< x-component of the dimensionless spin of object 1 */
     REAL8 S1y,                              /**< y-component of the dimensionless spin of object 1 */
     REAL8 S1z,                              /**< z-component of the dimensionless spin of object 1 */
     REAL8 S2x,                              /**< x-component of the dimensionless spin of object 2 */
     REAL8 S2y,                              /**< y-component of the dimensionless spin of object 2 */
     REAL8 S2z,                              /**< z-component of the dimensionless spin of object 2 */
     REAL8 distance,                         /**< distance of source (m) */
     REAL8 inclination,                      /**< inclination of source (rad) */
     REAL8 phiRef,                           /**< reference orbital phase (rad) */
     REAL8 longAscNodes,                     /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
     REAL8 eccentricity,                     /**< eccentricity at reference epoch */
     REAL8 meanPerAno,                       /**< mean anomaly of periastron */
     REAL8 deltaF,                           /**< sampling interval (Hz) */
     REAL8 f_min,                            /**< starting GW frequency (Hz) */
     REAL8 f_max,                            /**< ending GW frequency (Hz) */
     REAL8 f_ref,                            /**< Reference frequency (Hz) */
     LALDict *LALparams,                     /**< LAL dictionary containing accessory parameters */
     Approximant approximant                 /**< post-Newtonian approximant to use for waveform production */
     )
 {
           XLAL_CHECK(f_max > 0, XLAL_EDOM, "Maximum frequency must be > 0\n");
  
     const double extra_time_fraction = 0.1; /* fraction of waveform duration to add as extra time for tapering */
     const double extra_cycles = 3.0; /* more extra time measured in cycles at the starting frequency */
     double chirplen, deltaT, f_nyquist;
     int chirplen_exp;
     int retval;
         size_t n;
  
     /* adjust the reference frequency for certain precessing approximants:
      * if that approximate interprets f_ref==0 to be f_min, set f_ref=f_min;
      * otherwise do nothing */
     f_ref = zombieFixReferenceFrequency(f_ref, f_min, approximant);
  
     /* apply redshift correction to dimensionful source-frame quantities */
     REAL8 z=XLALSimInspiralWaveformParamsLookupRedshift(LALparams);
     if (z != 0.0) {
         m1 *= (1.0 + z);
         m2 *= (1.0 + z);
         distance *= (1.0 + z);  /* change from comoving (transverse) distance to luminosity distance */
     }
     /* set redshift to zero so we don't accidentally apply it again later */
     z = 0.0;
     if (LALparams)
       XLALSimInspiralWaveformParamsInsertRedshift(LALparams,z);
  
         /* Apply condition that f_max rounds to the next power-of-two multiple
          * of deltaF.
          * Round f_max / deltaF to next power of two.
          * Set f_max to the new Nyquist frequency.
          * The length of the chirp signal is then 2 * f_nyquist / deltaF.
          * The time spacing is 1 / (2 * f_nyquist) */
     f_nyquist = f_max;
         if (deltaF != 0) {
             n = round(f_max / deltaF);
             if ((n & (n - 1))) { /* not a power of 2 */
                         frexp(n, &chirplen_exp);
                         f_nyquist = ldexp(1.0, chirplen_exp) * deltaF;
                 XLAL_PRINT_WARNING("f_max/deltaF = %g/%g = %g is not a power of two: changing f_max to %g", f_max, deltaF, f_max/deltaF, f_nyquist);
         }
         }
     deltaT = 0.5 / f_nyquist;
  
     if (XLALSimInspiralImplementedFDApproximants(approximant)) {
  
         /* generate a FD waveform and condition it by applying tapers at
          * frequencies between a frequency below the requested f_min and
          * f_min; also wind the waveform in phase in case it would wrap-
          * around at the merger time */
  
         double tchirp, tmerge, textra, tshift;
         double fstart, fisco;
         double s;
         size_t k, k0, k1;
  
         /* if the requested low frequency is below the lowest Kerr ISCO
          * frequency then change it to that frequency */
         fisco = 1.0 / (pow(9.0, 1.5) * LAL_PI * (m1 + m2) * LAL_MTSUN_SI / LAL_MSUN_SI);
         if (f_min > fisco)
             f_min = fisco;
  
         /* upper bound on the chirp time starting at f_min */
         tchirp = XLALSimInspiralChirpTimeBound(f_min, m1, m2, S1z, S2z);
  
         /* upper bound on the final plunge, merger, and ringdown time */
         switch (approximant) {
         case TaylorF2:
         case TaylorF2Ecc:
         case TaylorF2NLTides:
         case SpinTaylorF2:
         case TaylorF2RedSpin:
         case TaylorF2RedSpinTidal:
         case SpinTaylorT4Fourier:
             /* inspiral-only models: no merger time */
             tmerge = 0.0;
             break;
         default:
             /* IMR model: estimate plunge and merger time */
             /* sometimes these waveforms have phases that
              * cause them to wrap-around an amount equal to
              * the merger-ringodwn time, so we will undo
              * that here */
             s = XLALSimInspiralFinalBlackHoleSpinBound(S1z, S2z);
             tmerge = XLALSimInspiralMergeTimeBound(m1, m2) + XLALSimInspiralRingdownTimeBound(m1 + m2, s);
             break;
         }
  
         /* new lower frequency to start the waveform: add some extra early
          * part over which tapers may be applied, the extra amount being
          * a fixed fraction of the chirp time; add some additional padding
          * equal to a few extra cycles at the low frequency as well for
          * safety and for other routines to use */
         textra = extra_cycles / f_min;
         fstart = XLALSimInspiralChirpStartFrequencyBound((1.0 + extra_time_fraction) * tchirp, m1, m2);
  
         /* revise (over-)estimate of chirp from new start frequency */
         tchirp = XLALSimInspiralChirpTimeBound(fstart, m1, m2, S1z, S2z);
  
         /* need a long enough segment to hold a whole chirp with some padding */
         /* length of the chirp in samples */
         chirplen = round((tchirp + tmerge + 2.0 * textra) / deltaT);
         /* make chirplen next power of two */
         frexp(chirplen, &chirplen_exp);
         chirplen = ldexp(1.0, chirplen_exp);
         /* frequency resolution */
         if (deltaF == 0.0)
             deltaF = 1.0 / (chirplen * deltaT);
         else if (deltaF > 1.0 / (chirplen * deltaT))
             XLAL_PRINT_WARNING("Specified frequency interval of %g Hz is too large for a chirp of duration %g s", deltaF, chirplen * deltaT);
  
         /* generate the waveform in the frequency domain starting at fstart */
        
         printf("Zombie 4 | fstart: %f, f_ref %f, deltaT %f | \n", fstart, f_ref, deltaT);
         retval = cuInspiralChooseFDWaveform(hptilde, hctilde, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, distance, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, deltaF, fstart, f_max, f_ref, LALparams, approximant);
         printArrayE("First 10 4 Zombie", (*hptilde)->data->data, 1000);

         if (retval < 0)
             XLAL_ERROR(XLAL_EFUNC);
  
         /* taper frequencies between fstart and f_min */
         k0 = round(fstart / (*hptilde)->deltaF);
         k1 = round(f_min / (*hptilde)->deltaF);
         /* make sure it is zero below fstart */
         for (k = 0; k < k0; ++k) {
             (*hptilde)->data->data[k] = 0.0;
             (*hctilde)->data->data[k] = 0.0;
         }
         /* taper between fstart and f_min */
         for ( ; k < k1; ++k) {
             double w = 0.5 - 0.5 * cos(M_PI * (k - k0) / (double)(k1 - k0));
             (*hptilde)->data->data[k] *= w;
             (*hctilde)->data->data[k] *= w;
         }
         /* make sure Nyquist frequency is zero */
         (*hptilde)->data->data[(*hptilde)->data->length - 1] = 0.0;
         (*hctilde)->data->data[(*hctilde)->data->length - 1] = 0.0;
  
         /* we want to make sure that this waveform will give something
          * sensible if it is later transformed into the time domain:
          * to avoid the end of the waveform wrapping around to the beginning,
          * we shift waveform backwards in time and compensate for this
          * shift by adjusting the epoch */
         tshift = round(tmerge / deltaT) * deltaT; /* integer number of time samples */
         for (k = 0; k < (*hptilde)->data->length; ++k) {
             double complex phasefac = cexp(2.0 * M_PI * I * k * deltaF * tshift);
             (*hptilde)->data->data[k] *= phasefac;
             (*hctilde)->data->data[k] *= phasefac;
         }
         XLALGPSAdd(&(*hptilde)->epoch, tshift);
         XLALGPSAdd(&(*hctilde)->epoch, tshift);
  
     } else if (XLALSimInspiralImplementedTDApproximants(approximant)) {
  
         /* generate a conditioned waveform in the time domain and Fourier-transform it */
  
         REAL8TimeSeries *hplus = NULL;
         REAL8TimeSeries *hcross = NULL;
         void *plan;
  
         /* generate conditioned waveform in time domain */
         retval = XLALSimInspiralTD(&hplus, &hcross, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, distance, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, deltaT, f_min, f_ref, LALparams, approximant);
         if (retval < 0)
             XLAL_ERROR(XLAL_EFUNC);
  
         /* frequency resolution */
         if (deltaF == 0.0) {
             /* round length of time domain signal to next power of two */
             chirplen = hplus->data->length;
             frexp(chirplen, &chirplen_exp);
             chirplen = ldexp(1.0, chirplen_exp);
             deltaF = 1.0 / (chirplen * hplus->deltaT);
         } else {
             /* set chirp length using precomputed Nyquist */
                 chirplen = 2 * f_nyquist / deltaF;
             if (chirplen < hplus->data->length)
                 XLAL_PRINT_WARNING(
                     "Specified frequency interval of %g Hz is too large for a chirp of duration %g s with Nyquist frequency %g Hz. The inspiral will be truncated.",
                     deltaF, hplus->data->length * deltaT, f_nyquist
                 );
         }
  
         /* resize waveforms to the required length */
         XLALResizeREAL8TimeSeries(hplus, hplus->data->length - (size_t) chirplen, (size_t) chirplen);
         XLALResizeREAL8TimeSeries(hcross, hcross->data->length - (size_t) chirplen, (size_t) chirplen);
  
         /* put the waveform in the frequency domain */
         /* (the units will correct themselves) */
         *hptilde = XLALCreateCOMPLEX16FrequencySeries("FD H_PLUS", &hplus->epoch, 0.0, deltaF, &lalDimensionlessUnit, (size_t) chirplen / 2 + 1);
         *hctilde = XLALCreateCOMPLEX16FrequencySeries("FD H_CROSS", &hcross->epoch, 0.0, deltaF, &lalDimensionlessUnit, (size_t) chirplen / 2 + 1);
         plan = XLALCreateForwardREAL8FFTPlan((size_t) chirplen, 0);
         XLALREAL8TimeFreqFFT(*hctilde, hcross, plan);
         XLALREAL8TimeFreqFFT(*hptilde, hplus, plan);
  
         /* clean up */
         XLALDestroyREAL8FFTPlan(plan);
         XLALDestroyREAL8TimeSeries(hcross);
         XLALDestroyREAL8TimeSeries(hplus);
  
     } else /* error: neither a FD nor a TD approximant */
         XLAL_ERROR(XLAL_EINVAL, "Invalid approximant");
  
     return 0;
 }

static int ZombieInspiralTDFromFD(
     REAL8TimeSeries **hplus,                    /**< +-polarization waveform */
     REAL8TimeSeries **hcross,                   /**< x-polarization waveform */
     REAL8 m1,                                   /**< mass of companion 1 (kg) */
     REAL8 m2,                                   /**< mass of companion 2 (kg) */
     REAL8 S1x,                                  /**< x-component of the dimensionless spin of object 1 */
     REAL8 S1y,                                  /**< y-component of the dimensionless spin of object 1 */
     REAL8 S1z,                                  /**< z-component of the dimensionless spin of object 1 */
     REAL8 S2x,                                  /**< x-component of the dimensionless spin of object 2 */
     REAL8 S2y,                                  /**< y-component of the dimensionless spin of object 2 */
     REAL8 S2z,                                  /**< z-component of the dimensionless spin of object 2 */
     REAL8 distance,                             /**< distance of source (m) */
     REAL8 inclination,                          /**< inclination of source (rad) */
     REAL8 phiRef,                               /**< reference orbital phase (rad) */
     REAL8 longAscNodes,                         /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
     REAL8 eccentricity,                         /**< eccentrocity at reference epoch */
     REAL8 meanPerAno,                           /**< mean anomaly of periastron */
     REAL8 deltaT,                               /**< sampling interval (s) */
     REAL8 f_min,                                /**< starting GW frequency (Hz) */
     REAL8 f_ref,                                /**< reference GW frequency (Hz) */
     LALDict *LALparams,                         /**< LAL dictionary containing accessory parameters */
     Approximant approximant                     /**< post-Newtonian approximant to use for waveform production */
 )
 {
     COMPLEX16FrequencySeries *hptilde = NULL;
     COMPLEX16FrequencySeries *hctilde = NULL;
     void *plan;
     size_t chirplen, end, k;
     double tshift;
     const double extra_time_fraction = 0.1; /* fraction of waveform duration to add as extra time for tapering */
     const double extra_cycles = 3.0; /* more extra time measured in cycles at the starting frequency */
     double original_f_min = f_min; /* f_min might be overwritten below, so keep original value */
     double f_max = 0.5 / deltaT;
     double tchirp, tmerge, textra;
     double fisco, fstart;
     double s;
     int retval;
  
     if (!XLALSimInspiralImplementedFDApproximants(approximant))
         XLAL_ERROR(XLAL_EINVAL, "Invalid approximant: not a FD approximant");
  
     /* adjust the reference frequency for certain precessing approximants:
      * if that approximate interprets f_ref==0 to be f_min, set f_ref=f_min;
      * otherwise do nothing */
     f_ref = zombieFixReferenceFrequency(f_ref, f_min, approximant);
  
     /* apply redshift correction to dimensionful source-frame quantities */
     REAL8 z=XLALSimInspiralWaveformParamsLookupRedshift(LALparams);
     if (z != 0.0) {
         m1 *= (1.0 + z);
         m2 *= (1.0 + z);
         distance *= (1.0 + z);  /* change from comoving (transverse) distance to luminosity distance */
     }
     /* set redshift to zero so we don't accidentally apply it again later */
     z=0.;
     if (LALparams)
       XLALSimInspiralWaveformParamsInsertRedshift(LALparams,z);
  
     /* if the requested low frequency is below the lowest Kerr ISCO
      * frequency then change it to that frequency */
     fisco = 1.0 / (pow(9.0, 1.5) * LAL_PI * (m1 + m2) * LAL_MTSUN_SI / LAL_MSUN_SI);
     if (f_min > fisco)
         f_min = fisco;
  
     /* upper bound on the chirp time starting at f_min */
     tchirp = XLALSimInspiralChirpTimeBound(f_min, m1, m2, S1z, S2z);
    
     /* upper bound on the final black hole spin */
     s = XLALSimInspiralFinalBlackHoleSpinBound(S1z, S2z);
  
     /* upper bound on the final plunge, merger, and ringdown time */
     tmerge = XLALSimInspiralMergeTimeBound(m1, m2) + XLALSimInspiralRingdownTimeBound(m1 + m2, s);
  
     /* extra time to include for all waveforms to take care of situations
      * where the frequency is close to merger (and is sweeping rapidly):
      * this is a few cycles at the low frequency */
     textra = extra_cycles / f_min;
  
     /* generate the conditioned waveform in the frequency domain */
     /* note: redshift factor has already been applied above */
     /* set deltaF = 0 to get a small enough resolution */
     //printf("Zombie 3 | fstart: %f, f_ref %f, deltaT %f | \n", f_min, f_ref, deltaT);
    
    spin_t spin_1 = 
    {
        .x = S1x,
        .y = S1y,
        .z = S1z
    };
    spin_t spin_2 = 
    {
        .x = S2x,
        .y = S2y,
        .z = S2z
    };
    companion_s companion_1 = 
    {
        .mass              = initMassKilograms(m1),
        .spin              = spin_1,
        .quadrapole_moment = 0.0,
        .lambda            = 0.0
    };
    companion_s companion_2 = 
    {
        .mass              = initMassKilograms(m2),
        .spin              = spin_2,
        .quadrapole_moment = 0.0,
        .lambda            = 0.0
    };
    
     retval = cuInspiralFD(&hptilde, &hctilde, companion_1, companion_2, distance, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, 0.0, f_min, f_max, f_ref, LALparams, approximant);
     if (retval < 0)
         XLAL_ERROR(XLAL_EFUNC);
      
    
    // Init property structures:
    system_properties_s system_properties =
        initBinarySystem(
            companion_1,
            companion_2,
            initLengthMeters(distance),
            z,
            initAngleRadians(inclination),
            initAngleRadians(phiRef),
            longAscNodes,
            eccentricity, 
            meanPerAno
        );
    
    
    temporal_properties_s temporal_properties =
        initTemporalProperties(
            initTimeSeconds(deltaT),   // <-- Sampling interval (timeUnit_t).
            initFrequencyHertz(f_min),  // <-- Starting GW frequency (frequencyUnit_t).
            initFrequencyHertz(f_ref), // <-- Reference GW frequency (frequencyUnit_t).
            system_properties,
            approximant
        );
    
    /*
    // Rransform the waveform into the time domain:
    chirplen = 2 * (hptilde->data->length - 1);
    *hplus = XLALCreateREAL8TimeSeries("H_PLUS", &hptilde->epoch, 0.0, temporal_properties.sampling_interval.seconds, &lalStrainUnit, chirplen);
    *hcross = XLALCreateREAL8TimeSeries("H_CROSS", &hctilde->epoch, 0.0, temporal_properties.sampling_interval.seconds, &lalStrainUnit, chirplen);
    
    size_t num_waveform_samples = 2 * (hptilde->data->length - 1);
        performIRFFT(
            hptilde->data->data,
            hctilde->data->data,
            (*hplus)->data->data,
            (*hcross)->data->data,
            temporal_properties,
            initFrequencyHertz(hptilde->deltaF),
            (int32_t)num_waveform_samples
        );
    */
    
     // we want to make sure that this waveform will give something
    // sensible if it is later transformed into the time domain:
    // to avoid the end of the waveform wrapping around to the beginning,
    // we shift waveform backwards in time and compensate for this
    // shift by adjusting the epoch -- note that XLALSimInspiralFD
    // guarantees that there is extra padding to do this 
     tshift = round(textra / deltaT) * deltaT; // integer number of samples 
     for (k = 0; k < hptilde->data->length; ++k) {
         double complex phasefac = cexp(2.0 * M_PI * I * k * hptilde->deltaF * tshift);
         hptilde->data->data[k] *= phasefac;
         hctilde->data->data[k] *= phasefac;
     }
     XLALGPSAdd(&hptilde->epoch, tshift);
     XLALGPSAdd(&hctilde->epoch, tshift);
  
     // transform the waveform into the time domain 
     chirplen = 2 * (hptilde->data->length - 1);
     *hplus = XLALCreateREAL8TimeSeries("H_PLUS", &hptilde->epoch, 0.0, deltaT, &lalStrainUnit, chirplen);
     *hcross = XLALCreateREAL8TimeSeries("H_CROSS", &hctilde->epoch, 0.0, deltaT, &lalStrainUnit, chirplen);
     plan = XLALCreateReverseREAL8FFTPlan(chirplen, 0);
     if (!(*hplus) || !(*hcross) || !plan) {
         XLALDestroyCOMPLEX16FrequencySeries(hptilde);
         XLALDestroyCOMPLEX16FrequencySeries(hctilde);
         XLALDestroyREAL8TimeSeries(*hcross);
         XLALDestroyREAL8TimeSeries(*hplus);
         XLALDestroyREAL8FFTPlan(plan);
         XLAL_ERROR(XLAL_EFUNC);
     }
     XLALREAL8FreqTimeFFT(*hplus, hptilde, plan);
     XLALREAL8FreqTimeFFT(*hcross, hctilde, plan);
  
     /* compute how long a chirp we should have */
     /* revised estimate of chirp length from new start frequency */
     fstart = XLALSimInspiralChirpStartFrequencyBound((1.0 + extra_time_fraction) * tchirp, m1, m2);
    
     printf("fstart zomb %f \n", tchirp);
     tchirp = XLALSimInspiralChirpTimeBound(fstart, m1, m2, S1z, S2z);
  
     /* total expected chirp length includes merger */
     chirplen = round((tchirp + tmerge) / deltaT);
  
     /* amount to snip off at the end is tshift */
     end = (*hplus)->data->length - round(tshift / deltaT);
  
     /* snip off extra time at beginning and at the end */
     XLALResizeREAL8TimeSeries(*hplus, end - chirplen, chirplen);
     XLALResizeREAL8TimeSeries(*hcross, end - chirplen, chirplen);
  
     /* clean up */
     XLALDestroyREAL8FFTPlan(plan);
     XLALDestroyCOMPLEX16FrequencySeries(hptilde);
     XLALDestroyCOMPLEX16FrequencySeries(hctilde);
     
     return 0;
 }

 /* Internal utility macro to check transverse spins are zero
    returns 1 if x and y components of spins are zero, otherwise returns 0 */
 #define checkTransverseSpinsZero(s1x, s1y, s2x, s2y) \
     (((s1x) != 0. || (s1y) != 0. || (s2x) != 0. || (s2y) != 0. ) ? 0 : 1)

 /* Internal utility macro to check tidal parameters are zero
    returns 1 if both tidal parameters zero, otherwise returns 0 */
 #define checkTidesZero(lambda1, lambda2) \
     (((lambda1) != 0. || (lambda2) != 0. ) ? 0 : 1)

int ZombieInspiralChooseTDWaveform(
     REAL8TimeSeries **hplus,                    /**< +-polarization waveform */
     REAL8TimeSeries **hcross,                   /**< x-polarization waveform */
     const REAL8 m1,                             /**< mass of companion 1 (kg) */
     const REAL8 m2,                             /**< mass of companion 2 (kg) */
     const REAL8 S1x,                            /**< x-component of the dimensionless spin of object 1 */
     const REAL8 S1y,                            /**< y-component of the dimensionless spin of object 1 */
     const REAL8 S1z,                            /**< z-component of the dimensionless spin of object 1 */
     const REAL8 S2x,                            /**< x-component of the dimensionless spin of object 2 */
     const REAL8 S2y,                            /**< y-component of the dimensionless spin of object 2 */
     const REAL8 S2z,                            /**< z-component of the dimensionless spin of object 2 */
     const REAL8 distance,                       /**< distance of source (m) */
     const REAL8 inclination,                    /**< inclination of source (rad) */
     const REAL8 phiRef,                         /**< reference orbital phase (rad) */
     const REAL8 longAscNodes,                   /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
     const REAL8 eccentricity,                   /**< eccentrocity at reference epoch */
     const REAL8 meanPerAno,              /**< mean anomaly of periastron */
     const REAL8 deltaT,                         /**< sampling interval (s) */
     const REAL8 f_min,                          /**< starting GW frequency (Hz) */
     REAL8 f_ref,                                /**< reference GW frequency (Hz) */
     LALDict *LALparams,                         /**< LAL dictionary containing accessory parameters */
     const Approximant approximant               /**< post-Newtonian approximant to use for waveform production */
     )
 {
     REAL8 LNhatx, LNhaty, LNhatz, E1x, E1y, E1z;
     //REAL8 tmp1, tmp2;
     int ret;
     /* N.B. the quadrupole of a spinning compact body labeled by A is
      * Q_A = - quadparam_A chi_A^2 m_A^3 (see gr-qc/9709032)
      * where quadparam = 1 for BH ~= 4-8 for NS.
      * This affects the quadrupole-monopole interaction.
      */
     REAL8 v0 = 1.;
                 /* Note: approximant SEOBNRv2T/v4T will by default compute dQuadMon1, dQuadMon2 */
                 /* from TidalLambda1, TidalLambda2 using universal relations, */
                 /* or use the input value if it is present in the dictionary LALparams */
     REAL8 quadparam1 = 1.+XLALSimInspiralWaveformParamsLookupdQuadMon1(LALparams);
     REAL8 quadparam2 = 1.+XLALSimInspiralWaveformParamsLookupdQuadMon2(LALparams);
     REAL8 lambda1 = XLALSimInspiralWaveformParamsLookupTidalLambda1(LALparams);
     REAL8 lambda2 = XLALSimInspiralWaveformParamsLookupTidalLambda2(LALparams);
     int amplitudeO = XLALSimInspiralWaveformParamsLookupPNAmplitudeOrder(LALparams);
     int phaseO =XLALSimInspiralWaveformParamsLookupPNPhaseOrder(LALparams);
                 /* Tidal parameters to be computed, if required, by universal relations */
     REAL8 lambda3A_UR = 0.;
     REAL8 omega2TidalA_UR = 0.;
     REAL8 omega3TidalA_UR = 0.;
     REAL8 lambda3B_UR = 0.;
     REAL8 omega2TidalB_UR = 0.;
     REAL8 omega3TidalB_UR = 0.;
     REAL8 quadparam1_UR = 0.;
     REAL8 quadparam2_UR = 0.;
  
     /* General sanity checks that will abort
      *
      * If non-GR approximants are added, include them in
      * XLALSimInspiralApproximantAcceptTestGRParams()
      */
     if( !XLALSimInspiralWaveformParamsNonGRAreDefault(LALparams) && XLALSimInspiralApproximantAcceptTestGRParams(approximant) != LAL_SIM_INSPIRAL_TESTGR_PARAMS ) {
         XLALPrintError("XLAL Error - %s: Passed in non-NULL pointer to LALSimInspiralTestGRParam for an approximant that does not use LALSimInspiralTestGRParam\n", __func__);
         XLAL_ERROR(XLAL_EINVAL);
     }
     /* Support variables for precessing wfs*/
     REAL8 incl;
  
  
     /* SEOBNR flag for spin aligned model version. 1 for SEOBNRv1, 2 for SEOBNRv2 */
     UINT4 SpinAlignedEOBversion;
     REAL8 spin1x,spin1y,spin1z;
     REAL8 spin2x,spin2y,spin2z;
     REAL8 polariz=longAscNodes;
  
     /* SEOBNR flag for precessing model version. 3 for SEOBNRv3, 300 for SEOBNRv3_opt, 401 for SEOBNRv4P, 402 for SEOBNRv4PHM */
     UINT4 PrecEOBversion;
     REAL8 spin1[3], spin2[3];
  
     REAL8 maxamp = 0;
     INT4 loopi = 0;
     INT4 maxind = 0;
  
     //LIGOTimeGPS epoch = LIGOTIMEGPSZERO;
  
     /* General sanity check the input parameters - only give warnings! */
     if( deltaT > 1. )
         XLALPrintWarning("XLAL Warning - %s: Large value of deltaT = %e requested.\nPerhaps sample rate and time step size were swapped?\n", __func__, deltaT);
     if( deltaT < 1./16385. )
         XLALPrintWarning("XLAL Warning - %s: Small value of deltaT = %e requested.\nCheck for errors, this could create very large time series.\n", __func__, deltaT);
     if( m1 < 0.09 * LAL_MSUN_SI )
         XLALPrintWarning("XLAL Warning - %s: Small value of m1 = %e (kg) = %e (Msun) requested.\nPerhaps you have a unit conversion error?\n", __func__, m1, m1/LAL_MSUN_SI);
     if( m2 < 0.09 * LAL_MSUN_SI )
         XLALPrintWarning("XLAL Warning - %s: Small value of m2 = %e (kg) = %e (Msun) requested.\nPerhaps you have a unit conversion error?\n", __func__, m2, m2/LAL_MSUN_SI);
     if( m1 + m2 > 1000. * LAL_MSUN_SI )
         XLALPrintWarning("XLAL Warning - %s: Large value of total mass m1+m2 = %e (kg) = %e (Msun) requested.\nSignal not likely to be in band of ground-based detectors.\n", __func__, m1+m2, (m1+m2)/LAL_MSUN_SI);
     if( S1x*S1x + S1y*S1y + S1z*S1z > 1.000001 )
         XLALPrintWarning("XLAL Warning - %s: S1 = (%e,%e,%e) with norm > 1 requested.\nAre you sure you want to violate the Kerr bound?\n", __func__, S1x, S1y, S1z);
     if( S2x*S2x + S2y*S2y + S2z*S2z > 1.000001 )
         XLALPrintWarning("XLAL Warning - %s: S2 = (%e,%e,%e) with norm > 1 requested.\nAre you sure you want to violate the Kerr bound?\n", __func__, S2x, S2y, S2z);
     if( f_min < 1. )
         XLALPrintWarning("XLAL Warning - %s: Small value of fmin = %e requested.\nCheck for errors, this could create a very long waveform.\n", __func__, f_min);
     if( f_min > 40.000001 )
         XLALPrintWarning("XLAL Warning - %s: Large value of fmin = %e requested.\nCheck for errors, the signal will start in band.\n", __func__, f_min);
  
     /* adjust the reference frequency for certain precessing approximants:
      * if that approximate interprets f_ref==0 to be f_min, set f_ref=f_min;
      * otherwise do nothing */
     f_ref = zombieFixReferenceFrequency(f_ref, f_min, approximant);
  
     switch (approximant)
     {
         case IMRPhenomD:
             if( !XLALSimInspiralWaveformParamsFlagsAreDefault(LALparams) )
                     XLAL_ERROR(XLAL_EINVAL, "Non-default flags given, but this approximant does not support this case.");
             if( !checkTransverseSpinsZero(S1x, S1y, S2x, S2y) )
                     XLAL_ERROR(XLAL_EINVAL, "Non-zero transverse spins were given, but this is a non-precessing approximant.");
             if( !checkTidesZero(lambda1, lambda2) )
                     XLAL_ERROR(XLAL_EINVAL, "Non-zero tidal parameters were given, but this is approximant doe not have tidal corrections.");
             // generate TD waveforms with zero inclincation so that amplitude can be
             // calculated from hplus and hcross, apply inclination-dependent factors
             // in loop below
             
             //printf("Zombie 2 | fstart: %f, f_ref %f, deltaT %f | \n", f_min, f_ref, deltaT);
             ret = ZombieInspiralTDFromFD(hplus, hcross, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, distance, 0., phiRef, longAscNodes, eccentricity, meanPerAno, deltaT, f_min, f_ref, LALparams, approximant);
             //printArrayE("First 10 2 Zombie", (*hplus)->data->data, 100);

             maxamp=0;
         REAL8TimeSeries *hp = *hplus;
         REAL8TimeSeries *hc = *hcross;
         maxind=hp->data->length - 1;
             const REAL8 cfac=cos(inclination);
             const REAL8 pfac = 0.5 * (1. + cfac*cfac);
         for (loopi=hp->data->length - 1; loopi > -1; loopi--)
             {
                     REAL8 ampsqr = (hp->data->data[loopi])*(hp->data->data[loopi]) +
                            (hc->data->data[loopi])*(hc->data->data[loopi]);
                     if (ampsqr > maxamp)
                     {
                             maxind=loopi;
                             maxamp=ampsqr;
                     }
                     hp->data->data[loopi] *= pfac;
                     hc->data->data[loopi] *= cfac;
             }
             XLALGPSSetREAL8(&(hp->epoch), (-1.) * deltaT * maxind);
             XLALGPSSetREAL8(&(hc->epoch), (-1.) * deltaT * maxind);
         break;
             
         default:
             printf("Wrong!");
        break;
     }
     //R.C.: here's the reference explaining why we perform this rotation https://dcc.ligo.org/LIGO-G1900275
     if (polariz && (*hplus) && (*hcross) ) {
       REAL8 tmpP,tmpC;
       REAL8 cp=cos(2.*polariz);
       REAL8 sp=sin(2.*polariz);
       for (UINT4 idx=0;idx<(*hplus)->data->length;idx++) {
         tmpP=(*hplus)->data->data[idx];
         tmpC=(*hcross)->data->data[idx];
         (*hplus)->data->data[idx] =cp*tmpP+sp*tmpC;
         (*hcross)->data->data[idx]=cp*tmpC-sp*tmpP;
       }
     }
  
     if (ret == XLAL_FAILURE) XLAL_ERROR(XLAL_EFUNC);
  
     return ret;
 }

static int XLALSimInspiralTDFromTD(
     REAL8TimeSeries **hplus,                    /**< +-polarization waveform */
     REAL8TimeSeries **hcross,                   /**< x-polarization waveform */
     REAL8 m1,                                   /**< mass of companion 1 (kg) */
     REAL8 m2,                                   /**< mass of companion 2 (kg) */
     REAL8 S1x,                                  /**< x-component of the dimensionless spin of object 1 */
     REAL8 S1y,                                  /**< y-component of the dimensionless spin of object 1 */
     REAL8 S1z,                                  /**< z-component of the dimensionless spin of object 1 */
     REAL8 S2x,                                  /**< x-component of the dimensionless spin of object 2 */
     REAL8 S2y,                                  /**< y-component of the dimensionless spin of object 2 */
     REAL8 S2z,                                  /**< z-component of the dimensionless spin of object 2 */
     REAL8 distance,                             /**< distance of source (m) */
     REAL8 inclination,                          /**< inclination of source (rad) */
     REAL8 phiRef,                               /**< reference orbital phase (rad) */
     REAL8 longAscNodes,                         /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
     REAL8 eccentricity,                         /**< eccentrocity at reference epoch */
     REAL8 meanPerAno,                           /**< mean anomaly of periastron */
     REAL8 deltaT,                               /**< sampling interval (s) */
     REAL8 f_min,                                /**< starting GW frequency (Hz) */
     REAL8 f_ref,                                /**< reference GW frequency (Hz) */
     LALDict *LALparams,                         /**< LAL dictionary containing accessory parameters */
     Approximant approximant                     /**< post-Newtonian approximant to use for waveform production */
 )
 {
     const double extra_time_fraction = 0.1; /* fraction of waveform duration to add as extra time for tapering */
     const double extra_cycles = 3.0; /* more extra time measured in cycles at the starting frequency */
     double original_f_min = f_min; /* f_min might be overwritten below, so keep original value */
     double tchirp, tmerge, textra;
     double fisco, fstart;
     double s;
     int retval;
  
     if (!XLALSimInspiralImplementedTDApproximants(approximant))
         XLAL_ERROR(XLAL_EINVAL, "Invalid approximant: not a TD approximant");
  
     /* adjust the reference frequency for certain precessing approximants:
      * if that approximate interprets f_ref==0 to be f_min, set f_ref=f_min;
      * otherwise do nothing */
     f_ref = zombieFixReferenceFrequency(f_ref, f_min, approximant);
  
     /* apply redshift correction to dimensionful source-frame quantities */
     REAL8 z=XLALSimInspiralWaveformParamsLookupRedshift(LALparams);
     if (z != 0.0) {
         m1 *= (1.0 + z);
         m2 *= (1.0 + z);
         distance *= (1.0 + z);  /* change from comoving (transverse) distance to luminosity distance */
     }
     /* set redshift to zero so we don't accidentally apply it again later */
     z=0.;
     if (LALparams)
       XLALSimInspiralWaveformParamsInsertRedshift(LALparams,z);
  
     /* if the requested low frequency is below the lowest Kerr ISCO
      * frequency then change it to that frequency */
     fisco = 1.0 / (pow(9.0, 1.5) * LAL_PI * (m1 + m2) * LAL_MTSUN_SI / LAL_MSUN_SI);
     if (f_min > fisco)
         f_min = fisco;
  
     /* upper bound on the chirp time starting at f_min */
     tchirp = XLALSimInspiralChirpTimeBound(f_min, m1, m2, S1z, S2z);
  
     /* upper bound on the final black hole spin */
     s = XLALSimInspiralFinalBlackHoleSpinBound(S1z, S2z);
  
     /* upper bound on the final plunge, merger, and ringdown time */
     tmerge = XLALSimInspiralMergeTimeBound(m1, m2) + XLALSimInspiralRingdownTimeBound(m1 + m2, s);
  
     /* extra time to include for all waveforms to take care of situations
      * where the frequency is close to merger (and is sweeping rapidly):
      * this is a few cycles at the low frequency */
     textra = extra_cycles / f_min;
  
     /* time domain approximant: condition by generating a waveform
      * with a lower starting frequency and apply tapers in the
      * region between that lower frequency and the requested
      * frequency f_min; here compute a new lower frequency */
     fstart = XLALSimInspiralChirpStartFrequencyBound((1.0 + extra_time_fraction) * tchirp + tmerge + textra, m1, m2);
  
     /* generate the waveform in the time domain starting at fstart */
    
    
    
    
    spin_t spin_1 = 
    {
        .x = S1x,
        .y = S1y,
        .z = S1z
    };
    spin_t spin_2 = 
    {
        .x = S2x,
        .y = S2y,
        .z = S2z
    };
    companion_s companion_1 = 
    {
        .mass              = initMassKilograms(m1),
        .spin              = spin_1,
        .quadrapole_moment = 0.0,
        .lambda            = 0.0
    };
    companion_s companion_2 = 
    {
        .mass              = initMassKilograms(m2),
        .spin              = spin_2,
        .quadrapole_moment = 0.0,
        .lambda            = 0.0
    };
    // Init property structures:
    system_properties_s system_properties =
        initBinarySystem(
            companion_1,
            companion_2,
            initLengthMeters(distance),
            z,
            initAngleRadians(inclination),
            initAngleRadians(phiRef),
            longAscNodes,
            eccentricity, 
            meanPerAno
        );
    
    
    temporal_properties_s temporal_properties =
        initTemporalProperties(
            initTimeSeconds(deltaT),   // <-- Sampling interval (timeUnit_t).
            initFrequencyHertz(f_min),  // <-- Starting GW frequency (frequencyUnit_t).
            initFrequencyHertz(f_ref), // <-- Reference GW frequency (frequencyUnit_t).
            system_properties,
            approximant
        );
    
    
    
    //retval = cuInspiralTDFromFD(hplus, hcross, system_properties, temporal_properties,  LALparams, approximant);
    retval = ZombieInspiralChooseTDWaveform(hplus, hcross, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, distance, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, deltaT, fstart, f_ref, LALparams, approximant);

     //printf("Zombie 1 | fstart: %f, f_ref %f, deltaT %f | \n", fstart, f_ref, deltaT);
     //printArrayE("First 10 1 Zombie", (*hplus)->data->data, 100);
    
    
     if (retval < 0)
         XLAL_ERROR(XLAL_EFUNC);
  
     /* condition the time domain waveform by tapering in the extra time
         * at the beginning and high-pass filtering above original f_min */
    
     //XLALSimInspiralTDConditionStage1(*hplus, *hcross, extra_time_fraction * tchirp + textra, original_f_min);
  
     /* final tapering at the beginning and at the end to remove filter transients */
  
     /* waveform should terminate at a frequency >= Schwarzschild ISCO
         * so taper one cycle at this frequency at the end; should not make
         * any difference to IMR waveforms */
     //fisco = 1.0 / (pow(6.0, 1.5) * LAL_PI * (m1 + m2) * LAL_MTSUN_SI / LAL_MSUN_SI);
     //XLALSimInspiralTDConditionStage2(*hplus, *hcross, f_min, fisco);
  
     return 0;
 }

int ZombieInspiralTD(
     REAL8TimeSeries **hplus,                    /**< +-polarization waveform */
     REAL8TimeSeries **hcross,                   /**< x-polarization waveform */
     REAL8 m1,                                   /**< mass of companion 1 (kg) */
     REAL8 m2,                                   /**< mass of companion 2 (kg) */
     REAL8 S1x,                                  /**< x-component of the dimensionless spin of object 1 */
     REAL8 S1y,                                  /**< y-component of the dimensionless spin of object 1 */
     REAL8 S1z,                                  /**< z-component of the dimensionless spin of object 1 */
     REAL8 S2x,                                  /**< x-component of the dimensionless spin of object 2 */
     REAL8 S2y,                                  /**< y-component of the dimensionless spin of object 2 */
     REAL8 S2z,                                  /**< z-component of the dimensionless spin of object 2 */
     REAL8 distance,                             /**< distance of source (m) */
     REAL8 inclination,                          /**< inclination of source (rad) */
     REAL8 phiRef,                               /**< reference orbital phase (rad) */
     REAL8 longAscNodes,                         /**< longitude of ascending nodes, degenerate with the polarization angle, Omega in documentation */
     REAL8 eccentricity,                         /**< eccentrocity at reference epoch */
     REAL8 meanPerAno,                           /**< mean anomaly of periastron */
     REAL8 deltaT,                               /**< sampling interval (s) */
     REAL8 f_min,                                /**< starting GW frequency (Hz) */
     REAL8 f_ref,                                /**< reference GW frequency (Hz) */
     LALDict *LALparams,                         /**< LAL dictionary containing accessory parameters */
     Approximant approximant                     /**< post-Newtonian approximant to use for waveform production */
     )
 {
     /* call the appropriate helper routine */
     if (XLALSimInspiralImplementedTDApproximants(approximant)) {
         /* If using approximants for which reference frequency is the starting frequency
         * generate using XLALSimInspiralChooseTDWaveform and apply the
         * LAL Taper 'LAL_SIM_INSPIRAL_TAPER_START' instead of
         * XLALSimInspiralTDConditionStage1 and XLALSimInspiralTDConditionStage2
         * as is done in XLALSimInspiralTDFromTD.
         * This is because XLALSimInspiralTDFromTD modifies the start frequency
         * which is not always possible with NR_hdf5 waveforms.
         */
  
       // Check whether for the given approximant reference frequency is the starting frequency
       SpinFreq spin_freq_flag = XLALSimInspiralGetSpinFreqFromApproximant(approximant);
       if (spin_freq_flag == LAL_SIM_INSPIRAL_SPINS_CASEBYCASE || spin_freq_flag == LAL_SIM_INSPIRAL_SPINS_FLOW)
        {
             if (XLALSimInspiralChooseTDWaveform(hplus, hcross, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, distance, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, deltaT, f_min, f_ref, LALparams, approximant) <0)
                 XLAL_ERROR(XLAL_EFUNC);
  
             /* taper the waveforms */
             LALSimInspiralApplyTaper taper = LAL_SIM_INSPIRAL_TAPER_START;
             if (XLALSimInspiralREAL8WaveTaper((*hplus)->data, taper) == XLAL_FAILURE)
                 XLAL_ERROR(XLAL_EFUNC);
             if (XLALSimInspiralREAL8WaveTaper((*hcross)->data, taper) == XLAL_FAILURE)
                 XLAL_ERROR(XLAL_EFUNC);
         } else {
             if (XLALSimInspiralTDFromTD(hplus, hcross, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, distance, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, deltaT, f_min, f_ref, LALparams, approximant) < 0)
                 XLAL_ERROR(XLAL_EFUNC);
         }
     } else if (XLALSimInspiralImplementedFDApproximants(approximant)) {
         if (ZombieInspiralTDFromFD(hplus, hcross, m1, m2, S1x, S1y, S1z, S2x, S2y, S2z, distance, inclination, phiRef, longAscNodes, eccentricity, meanPerAno, deltaT, f_min, f_ref, LALparams, approximant) < 0)
             XLAL_ERROR(XLAL_EFUNC);
     } else
         XLAL_ERROR(XLAL_EINVAL, "Invalid approximant");
    
     return 0;
 }


void generatePhenomZombie(
	const Approximant       approximant,
    const massUnit_t        mass_1, 
    const massUnit_t        mass_2, 
    const frequencyUnit_t   sample_rate, 
    const timeUnit_t        duration, 
    const angularUnit_t     inclination, 
    const lengthUnit_t      distance, 
          float64_2_t     **ret_strain
    ) {
	
	const int32_t num_samples = 
		(int32_t)floor(sample_rate.hertz*duration.seconds);
	
	REAL8TimeSeries *hplus  = NULL;
	REAL8TimeSeries *hcross = NULL;
	
	REAL8 S1x          = 0.0;
	REAL8 S1y          = 0.0;
	REAL8 S1z          = 0.0;
	REAL8 S2x          = 0.0;
	REAL8 S2y          = 0.0;
	REAL8 S2z          = 0.0;
	
	REAL8 phiRef       = 0.0;
	REAL8 longAscNodes = 0.0;
	REAL8 eccentricity = 0.0;
	REAL8 meanPerAno   = 0.0;
	REAL8 deltaT       = 1.0/sample_rate.hertz;
	REAL8 f_min        = 
		calcMinimumFrequency(
			mass_1, 
			mass_2, 
			duration
		).hertz;
	
	REAL8        f_ref       = 0.0;
	LALDict     *extraParams = NULL;
	
	//Converting to SI:
		
	ZombieInspiralTD(
		&hplus,
		&hcross,
		mass_1.kilograms,
		mass_2.kilograms,
		S1x,
		S1y,
		S1z,
		S2x,
		S2y,
		S2z,
		distance.meters,
		inclination.radians,
		phiRef,
		longAscNodes,
		eccentricity,
		meanPerAno,
		deltaT,
		f_min,
		f_ref,
		extraParams,
		approximant
	);
	
	const int32_t waveform_num_samples = (int32_t)hplus->data->length;
	
	if (waveform_num_samples < num_samples) 
	{	
		fprintf(
			stderr, 
			"Warning! LAL Simulation not generating waveforms of desired "
			"num_samples.\n"
		);
	}
	
	size_t new_array_size = (size_t)num_samples * sizeof(float64_2_t);

	float64_2_t *strain = (float64_2_t*)malloc(new_array_size);
	int32_t new_waveform_index = 0;
    for (int32_t index = 0; index < num_samples; index++) 
	{	
		new_waveform_index = waveform_num_samples - num_samples - 1 + index;
		strain[index].x = (float64_t)hplus->data->data[new_waveform_index];
		strain[index].y = (float64_t)hcross->data->data[new_waveform_index];
    }
	
	free(hcross->data->data); free(hplus->data->data);

	*ret_strain = strain;
}

#endif