# Dr. Baloney's Audio Converter

*Dr. Baloney's Audio Converter* is a lightweight, open-source ([CC0](https://creativecommons.org/publicdomain/zero/1.0/)) C library for real-time [sample rate conversion](https://en.wikipedia.org/wiki/Sample-rate_conversion) of audio streams, designed to be fast, easy to use, and dependency-free, making it ideal for use in embedded systems or performance-critical applications.

## Features

*Push and Pull Workflows*:
- *Pushing* converts incoming audio buffers of a specified length and sends the resampled data to a consumer callback.
- *Pulling* requests audio buffers from a producer callback, resamples them, and outputs the converted data via user-provided buffers of a specified length.
- Ideal for synchronized setups, such as routing a 44.1 kHz microphone through a 48 kHz processor and back to a 44.1 kHz speaker. The library ensures matching frame counts between the push/pull callbacks when provided with compatible buffers.

*C17 Complient Code with No External Dependencies:*
- Comprises a single header and source file.
- No dependencies besides `memcpy` and `memset`.
- Fully compliant with the [C17](https://en.wikipedia.org/wiki/C17_(C_standard_revision)) standard.

*Custom Allocators:*
- Supports custom allocators and allows pre-allocation of all necessary memory.
- No internal calls to `malloc`.

*Optimized for SIMD Architectures:*
- Optional support for **SIMD** architectures with **NEON** and **SSE** code paths.

*Lightweight*:
- The compiled static library is approximately 50 to 100 KB (on Windows/macOS/iOS/Android), including precomputed sinc tables and FIR filter taps.

*Multi-channel Support:*
- Supports audio streams with up to 8 channels, making it suitable for stereo, surround, or other multi-channel audio processing.

*Subsample Accurate Latency:*
- Reports accurate latency with subsample precision for each conversion call.

## Supported Sampling Rate Conversions

The library supports conversions between the following standard sampling rates:

- 8 kHz
- 11.025 kHz
- 16 kHz
- 22.05 kHz
- 32 kHz
- 44.1 kHz
- 48 kHz
- 60 kHz
- 88.2 kHz
- 96 kHz
- 120 kHz
- 176.4 kHz
- 192 kHz
- 240 kHz

## How to Use It?

To get started, copy [`drb-audio-converter.h`](drb-audio-converter.h) and [`drb-audio-converter.c`](drb-audio-converter.c) into your project.

The header file exposes a few enums, structs, and functions:

- `DrB_Audio_Converter_Direction`: Enum that is used to specify the direction of the conversion.
- `DrB_Audio_Converter_Quality`: Enum that is used to specify the quality of the resampling algorithm that will be used in the conversion.
- `DrB_Audio_Converter_Buffer`: Struct that represents a buffer of audio samples.
- `DrB_Audio_Converter_Data_Callback`: Struct that defines the callback interface for the converter.
- `DrB_Audio_Converter`: Opaque struct that represents an instance of a converter.
- `drb_audio_converter_alignment_and_size`: Function for computing the required memory alignment and size for constructing a converter. 
- `drb_audio_converter_construct`: Function for constructing a new converter.
- `drb_audio_converter_work_memory_alignment_and_size`: Function for computing the required memory alignment and size for the work memory that the converter needs while converting a batch of buffers.
- `drb_audio_converter_process`: Function for converting the next batch of audio buffers.

In this example, we'll consider a stereo audio generator producing audio at 44.1 kHz. Our goal is to convert this output to 48 kHz. Since we're pulling audio samples from the generator, the converter will request the necessary data using a producer callback.

First, let's implement the producer callback, which the converter will use to pull audio samples from the generator. This callback will be invoked whenever the converter needs more input data.

    static void pull
        (
            void * user_data,
            double latency,
            DrB_Audio_Converter_Buffer const buffers [],
            int frame_count
        )
    {
        Generator * const generator = user_data;

        // Use the generator to write samples into the buffers:

        ...
    }

In this callback:

- `user_data` refers to the audio generator instance.
- `latency` specifies the latency (in seconds) that the conversion will induce.
- `buffers` is an array of `DrB_Audio_Converter_Buffer` structs, one for each channel (stereo in this case).
- `frame_count` specifies how many frames we need to pull from the generator.

Before we can create the converter, we need to determine how much memory is required to store the converter instance. This memory will include space for internal state and buffers.

    long alignment, size;
    
    drb_audio_converter_alignment_and_size
    (
        drb_audio_converter_sampling_rate_44100,
        drb_audio_converter_sampling_rate_48000,
        2, // <- Channel count, 2 for stereo
        256, // <- Maximum frame count that we can process in the data callback.
        drb_audio_converter_direction_pull,
        drb_audio_converter_quality_good,
        &alignment,
        &size
    );

We specify input and output sample rates of 44.1 kHz and 48 kHz, respectively. The channel count is set to 2 (stereo), and the converter will process up to 256 frames per callback invocation. The conversion direction is set to *pull*, and the resampling quality is set to *good*.

Now we can allocate the required memory, ensuring the memory alignment matches the calculated requirements:

    void * const converter_memory = aligned_alloc(alignment, size);

With the allocated memory, we construct the converter instance by passing the desired sampling rates, channel count, maximum frame count, conversion direction, and the callback function.

    DrB_Audio_Converter * const converter = drb_audio_converter_construct
    (
        converter_memory,
        drb_audio_converter_sampling_rate_44100,
        drb_audio_converter_sampling_rate_48000,
        2, // <- Channel count, 2 for stereo
        256, // <- Maximum frame count that we can process in the data callback.
        drb_audio_converter_direction_pull,
        drb_audio_converter_quality_good,
        (DrB_Audio_Converter_Data_Callback){ .process = pull, .user_data = generator }
    );

The callback (`pull`) will be invoked whenever the converter needs audio data. We pass the generator instance via the `user_data` pointer, so it can be accessed within the callback. Note that the pointer to the converter will share the exact same address as the pointer to the raw memory (`converter_memory`).

The converter also requires work memory during the conversion process. This memory is transient and can be shared between converters or reused for other purposes between conversions:

    drb_audio_converter_work_memory_alignment_and_size
    (
        converter,
        &alignment,
        &size
    );

    void * const work_memory = aligned_alloc(alignment, size);

Finally, we can now process audio buffers by invoking the `drb_audio_converter_process` function, which converts audio from 44.1 kHz to 48 kHz:

    drb_audio_converter_process
    (
        converter_memory,
        work_memory,
        buffers,
        frame_count
    );

Here, the `buffers` array will contain the audio data after conversion. The `frame_count` specifies how many frames the converter shall produce. The converter will internally manage buffer sizes, ensuring the correct number of frames is processed without exceeding the specified limits.

Complete examples can be found in [`example-push.c`](example-push.c) and [`example-pull.c`](example-pull.c).

## How Does it Work?

More details coming soon.

## Quality Measurements

More details coming soon.

## Testing

More details coming soon.

## Changelog

## License

[CC0 1.0 UNIVERSAL](https://creativecommons.org/publicdomain/zero/1.0/). See [LICENSE.txt](LICENSE.txt).
