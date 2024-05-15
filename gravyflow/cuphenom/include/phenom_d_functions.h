#ifndef PHENOM_D_FUNCTIONS_H
#define PHENOM_D_FUNCTIONS_H

complex_waveform_axes_s initPhenomDWaveformAxes(
          temporal_properties_s *temporal_properties,
    const system_properties_s   *system_properties,
    const int32_t                num_waveforms
    );

int32_t sumPhenomDFrequencies(
    complex_waveform_axes_s waveform_axes_fd
    );

#endif