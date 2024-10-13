// Example: Pushing Audio Frames Through Dr. Baloney's Audio Converter
//
// This example demonstrates how to use Dr. Baloney's Audio Converter library to
// convert audio from a source sample rate at 44.1 kHz to a target sample rate
// at 48 kHz using the push workflow.
//
// In this workflow, audio frames are pushed into the converter, and the output
// is consumed via a callback function (`consume_frames`), which receives the
// resampled audio data. The example simulates processing audio in chunks
// (slices) and prints the resampled frames to the console.
//
// This example demonstrates the core usage of the library, including memory
// management, resampling between standard sample rates, and handling
// multi-channel audio (stereo).

#include "drb-audio-converter.h"

#include <assert.h> // For `assert`.
#include <stdlib.h> // For `EXIT_SUCCESS`, `aligned_alloc`, and `free`.
#include <stdio.h> // For `printf`.

// Enum values that define the conversion parameters.
enum { source_sampling_rate = drb_audio_converter_sampling_rate_44100 };
enum { target_sampling_rate = drb_audio_converter_sampling_rate_48000 };
enum { channel_count = 2 }; // Stereo (2 channels)
enum { max_frame_count = 256 }; // Maximum number of frames in `consume_frames`.
enum { quality = drb_audio_converter_quality_good }; // Resampling quality.
enum { total_frame_count = 400 }; // Total amount of frames to be processed.

// Defines how many frames will be pushed to the converter in each call.
static int const slices [] = { 13, 17, 4, 7, 5, 4, 21, 29, 300 };

// Calculate the number of slices.
enum { slice_count = sizeof(slices) / sizeof(int) };

// Struct to keep track of how many frames that have been processed.
typedef struct Counter
{
    int count;
}
Counter;

// This function will be called after incomming frames have been converted.
// `user_data`: User-provided data, in this case, a `Counter` instance.
// `latency`: Latency caused by the resampling (in seconds).
// `buffers`: Array of audio buffers; one for each channel.
// `frame_count`: Number of frames in each buffer.
static void consume_frames
    (
        void * user_data,
        double latency,
        DrB_Audio_Converter_Buffer const buffers [],
        int frame_count
    )
{
    // Access the Counter struct to track the total amount of processed frames.

    Counter * const counter = user_data;

    // Print the latency.

    printf("`consume_frames` (latency: %lf)\n", latency);

    // Iterate over the frames and print their values.

    for (int frame = 0; frame < frame_count; frame++)
    {
        printf("%3d", counter->count++); // Increment the counter.

        for (int channel = 0; channel < channel_count; channel++)
        {
            printf(", %8.3f", buffers[channel].samples[frame]);  // Print the sample value (formatted)
        }

        printf("\n");
    }
}

extern int main (int const argc, char const * const argv [const])
{
    (void)argc, (void)argv; // Suppress unused parameter warnings.

    long alignment, size;

    // Calculate the alignment and size required for the converter.

    assert(drb_audio_converter_alignment_and_size
    (
        source_sampling_rate,
        target_sampling_rate,
        channel_count,
        max_frame_count,
        drb_audio_converter_direction_push,
        quality,
        &alignment,
        &size
    ));

    // Allocate memory for the converter with the correct alignment.

    void * const converter_memory = aligned_alloc(alignment, size);

    // Ensure allocation was successful.

    assert(converter_memory);

    // Initialize the frame counter.

    Counter counter = { .count = 0 };

    // Construct the converter.

    DrB_Audio_Converter * const converter = drb_audio_converter_construct
    (
        converter_memory,
        source_sampling_rate,
        target_sampling_rate,
        channel_count,
        max_frame_count,
        drb_audio_converter_direction_push,
        quality,
        (DrB_Audio_Converter_Data_Callback){ consume_frames, &counter }
    );

    // Ensure the converter was created successfully.

    assert(converter);

    // Calculate the memory required for the work buffer (temporary buffer used
    // during processing).

    drb_audio_converter_work_memory_alignment_and_size
    (
        converter,
        &alignment,
        &size
    );

    // Allocate the work memory.

    void * const work_memory = aligned_alloc(alignment, size);

    // Ensure allocation was successful.

    assert(work_memory);

    // Create and populate the input data.

    float samples [channel_count][total_frame_count];

    for (int channel = 0; channel < channel_count; channel++)
    {
        for (int frame = 0; frame < total_frame_count; frame++)
        {
            samples[channel][frame] = frame + 100 * channel;
        }
    }

    // Process each "slice" (batch of frames) as defined by the `slices` array.

    for (int index = 0, offset = 0; index < slice_count; index++)
    {
        // Declare the buffers and make them point to the corresponding part of
        // the input data.

        DrB_Audio_Converter_Buffer buffers [channel_count];

        for (int channel = 0; channel < channel_count; channel++)
        {
            buffers[channel].samples = samples[channel] + offset;
        }

        // Process the slice.

        drb_audio_converter_process
        (
            converter,
            work_memory,
            buffers,
            slices[index] // Number of frames to process in this slice.
        );

        // Update the offset for the next slice of frames.

        offset += slices[index];

        // Ensure we don't exceed the total number of frames.

        assert(offset <= total_frame_count);
    }

    // Free the converter and work memory.

    free(work_memory);
    free(converter_memory);

    // Done!

    return EXIT_SUCCESS;
}
