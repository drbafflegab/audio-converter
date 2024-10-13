// Dr. Baloney's Audio Converter
//
// Lightweight, open-source (CC0) C library designed for real-time sample rate
// conversion of audio streams. It is fast, efficient, and dependency-free,
// making it ideal for use in embedded systems or performance-critical
// applications. The library consists of a single header and source file, with
// no dependencies besides `memcpy` and `memset`.
//
// Works in both push and pull directions. In the push direction, the library
// converts incoming audio buffers and sends the resampled data to a consumer
// callback. In the pull direction, the library requests audio data from a
// producer callback, resamples it, and outputs the converted data to a
// user-provided target buffer.

#ifndef DRB_AUDIO_CONVERTER_H
#define DRB_AUDIO_CONVERTER_H

#include <stdbool.h> // For `bool`, `false`, and `true`.

#if defined(__cplusplus)
extern "C" {
#endif

// Semantic version for the library.
static struct
{
    short major, minor, patch;
}
const drb_audio_converter_version = { 0, 1, 0 };

// Supported sampling rates.
enum
{
    drb_audio_converter_sampling_rate_8000 = 8000,
    drb_audio_converter_sampling_rate_11025 = 11025,
    drb_audio_converter_sampling_rate_16000 = 16000,
    drb_audio_converter_sampling_rate_22050 = 22050,
    drb_audio_converter_sampling_rate_32000 = 32000,
    drb_audio_converter_sampling_rate_44100 = 44100,
    drb_audio_converter_sampling_rate_48000 = 48000,
    drb_audio_converter_sampling_rate_60000 = 60000,
    drb_audio_converter_sampling_rate_88200 = 88200,
    drb_audio_converter_sampling_rate_96000 = 96000,
    drb_audio_converter_sampling_rate_120000 = 120000,
    drb_audio_converter_sampling_rate_176400 = 176400,
    drb_audio_converter_sampling_rate_192000 = 192000,
    drb_audio_converter_sampling_rate_240000 = 240000
};

// Minimum and maximum supported channel count.
enum
{
    min_channel_count = 1,
    max_channel_count = 8
};

// Direction of the conversion.
typedef enum DrB_Audio_Converter_Direction
{
    drb_audio_converter_direction_push,
    drb_audio_converter_direction_pull
}
DrB_Audio_Converter_Direction;

// Quality of the resampling algorithm.
typedef enum DrB_Audio_Converter_Quality
{
    drb_audio_converter_quality_poor,
    drb_audio_converter_quality_fine,
    drb_audio_converter_quality_good,
    drb_audio_converter_quality_best
}
DrB_Audio_Converter_Quality;

// Represents a buffer of audio samples.
// - `samples`: Pointer to an array of audio samples.
typedef struct DrB_Audio_Converter_Buffer
{
    float * samples;
}
DrB_Audio_Converter_Buffer;

// Represents the callback interface for the converter. Depending on the
// conversion direction, the converter will either call the callback with new,
// converted samples or request input samples to convert.
typedef struct DrB_Audio_Converter_Data_Callback
{
    // Callback function that will be called when new samples need to be
    // consumed/produced, depending on the direction of the conversion.
    // - `user_data`: User-provided data that was passed to the callback.
    // - `latency`: Latency (in secs) that the conversion will induce.
    // - `buffers`: Array of sample buffers to be filled or processed.
    // - `frame_count`: Number of frames per buffer.
    void (* process)
    (
        void * user_data,
        double latency,
        DrB_Audio_Converter_Buffer const buffers [],
        int frame_count
    );

    // User-provided data that will be passed to the callback.
    void * user_data;
}
DrB_Audio_Converter_Data_Callback;

// Opaque structure representing a converter instance.
typedef struct DrB_Audio_Converter DrB_Audio_Converter;

// Computes the required memory alignment and size for constructing a converter
// instance. The alignment and size are stored in the `alignment` and `size`
// pointers, respectively.
// - `source_sampling_rate`: Sampling rate of the source stream.
// - `target_sampling_rate`: Sampling rate of the target stream.
// - `channel_count`: Shared channel count in the source and target streams.
// - `max_frame_count`: Maximum frame count used in the data callback.
// - `direction`: Direction of the conversion.
// - `resampling_quality`: Quality of the resampling algorithm.
// - `alignment`: Int-pointer in which the required alignment will be stored.
// - `size`: Int-pointer in which the required size will be stored.
extern bool drb_audio_converter_alignment_and_size
    (
        int source_sampling_rate,
        int target_sampling_rate,
        int channel_count,
        int max_frame_count,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Quality resampling_quality,
        long * alignment,
        long * size
    );

// Constructs a new converter. The `memory` pointer must meet the alignment and
// size requirements provided by `drb_audio_converter_alignment_and_size`.
// The resulting `DrB_Audio_Converter` pointer will point to the same address as
// the `memory` pointer.
// - `memory`: Pointer to memory that will be used to construct the converter.
// - `source_sampling_rate`: Sampling rate of the source stream.
// - `target_sampling_rate`: Sampling rate of the target stream.
// - `channel_count`: Shared channel count in the source and target streams.
// - `max_frame_count`: Maximum frame count used in the data callback.
// - `direction`: Direction of the conversion (push or pull).
// - `resampling_quality`: Quality of the resampling algorithm.
// - `data_callback`: Callback that will be invoked before/after the conversion.
extern DrB_Audio_Converter * drb_audio_converter_construct
    (
        void * memory,
        int source_sampling_rate,
        int target_sampling_rate,
        int channel_count,
        int max_frame_count,
        DrB_Audio_Converter_Direction direction,
        DrB_Audio_Converter_Quality resampling_quality,
        DrB_Audio_Converter_Data_Callback data_callback
    );

// Computes the required memory alignment and size for the work memory that the
// converter needs while converting. The work memory is transient and only used
// during the conversion; afterward, it can be reused for other purposes.
// - `converter`: Converter for which to compute the memory requirements.
// - `alignment`: Int-pointer in which the required alignment will be stored.
// - `size`: Int-pointer in which the required size will be stored.
extern void drb_audio_converter_work_memory_alignment_and_size
    (
        DrB_Audio_Converter * converter,
        long * alignment,
        long * size
    );

// Converts a batch of samples with the converter. Depending on the conversion
// direction, the converter will either expect the buffers in the `buffers`
// argument to be filled with samples or will fill them with converted samples.
// The number of buffers in the `buffers` array must match the channel count
// provided during construction.
// - `converter`: Converter instance used for the conversion.
// - `work_memory`: Pointer to transient memory used during conversion.
// - `buffers`: List of buffers used for conversion.
// - `frame_count`: Number of samples per buffer.
extern void drb_audio_converter_process
    (
        DrB_Audio_Converter * converter,
        void * work_memory,
        DrB_Audio_Converter_Buffer const buffers [],
        int frame_count
    );

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // DRB_AUDIO_CONVERTER_H
