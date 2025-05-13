# Documentation Update Plan

**Goal:** Update `README.md` and `projectSpecs.md` to accurately reflect the current state of the Video to Japanese ASCII Art Converter project, including new features, arguments, implementation details, and documentation for the test suite and utility functions.

**Steps:**

1.  **Update `README.md`:**
    *   **Overview:**
        *   Mention the parallel processing capability and its benefits more prominently.
        *   Briefly explain the aspect ratio preservation during downscaling.
    *   **Usage:**
        *   Add the new command-line arguments: `--mode` (sequential/parallel), `--scale` (scaling factor for render resolution), and `--profile` (performance profiling).
        *   Update the description of `--width` and `--height` to clarify they are *maximum* dimensions and aspect ratio is preserved.
        *   Refine the description of `--processes` and `--batch-size` based on the code's implementation (defaulting `--processes` to CPU count, their role in parallel processing and memory management).
    *   **Examples:**
        *   Add examples demonstrating the use of the new arguments (`--mode`, `--scale`, `--profile`).
        *   Update existing examples to reflect the clarified behavior of `--width` and `--height`.
        *   Review and potentially revise the "Recommended Configurations" section to provide more general guidance on `--processes` and `--batch-size` based on system resources rather than specific numbers, aligning with the code's default behavior.
    *   **Troubleshooting:**
        *   Add a point about potential issues with font loading if a suitable Japanese font is not found, referencing the font loading logic in [`renderer.py`](renderer.py).
    *   **Technical Implementation Details:**
        *   Expand on the parallel processing implementation, referencing the use of `multiprocessing` and `ProcessPoolExecutor` in [`video_processor.py`](video_processor.py), [`ascii_converter.py`](ascii_converter.py), and [`renderer.py`](renderer.py).
        *   Explain the role of the static method `render_ascii_frame_to_image_static` and the worker initializer in [`renderer.py`](renderer.py) for parallel rendering.
        *   Briefly mention the aspect ratio preservation logic in [`video_processor.py`](video_processor.py) and [`ascii_converter.py`](ascii_converter.py).
        *   Add a section on the comparison video creation process, referencing [`video_processor.py`](video_processor.py) and the resizing/stacking using ffmpeg.

2.  **Update `projectSpecs.md`:**
    *   **Project Overview:**
        *   Update the overview to explicitly mention parallel processing and aspect ratio preservation.
    *   **System Requirements:**
        *   Update the list of required Python packages to accurately reflect [`requirements.txt`](requirements.txt) (`opencv-python`, `numpy`, `Pillow`, `tqdm`). Mention `subprocess`, `argparse`, and `os` as standard library imports used.
    *   **Project Structure:**
        *   Update the project structure diagram to include all files present in the repository: [`create_test_video.py`](create_test_video.py), [`monitor_resources.py`](monitor_resources.py), [`test_requirements.txt`](test_requirements.txt), [`test_suite.py`](test_suite.py), and [`utils.py`](utils.py), as well as the directories (`custom_comparison_*`, `debug_output`, `test_output`, `test_videos`).
    *   **Detailed Component Design:**
        *   **Command-line Interface (`main.py`):**
            *   Update the `parse_arguments` section to include the new arguments (`--mode`, `--scale`, `--profile`) and the updated descriptions for `--width`, `--height`, `--processes`, and `--batch-size`.
            *   Describe the logic for selecting sequential or parallel mode and initializing components accordingly.
            *   Explain the profiling feature enabled by `--profile`.
            *   Detail the comparison video creation logic when `--compare` is used, referencing the `create_comparison_video` method in [`video_processor.py`](video_processor.py).
        *   **Video Processor (`video_processor.py`):**
            *   Detail the aspect ratio preservation logic in `downscale_video`.
            *   Describe the parallel frame extraction using `ProcessPoolExecutor` in `extract_frames`.
            *   Document the `create_comparison_video` method, explaining its purpose and how it uses ffmpeg to combine the original and ASCII videos.
        *   **ASCII Converter (`ascii_converter.py`):**
            *   Detail the parallel conversion process using `ProcessPoolExecutor` and batch processing.
            *   Explain the aspect ratio preservation logic in `convert_video_to_ascii`.
            *   Mention the use of NumPy vectorization in `convert_frame_to_ascii`.
        *   **Character Mapper (`character_mapper.py`):**
            *   Ensure the description accurately reflects the character mapping logic and the use of `black_threshold` and `brightness_step`.
        *   **Renderer (`renderer.py`):**
            *   Detail the parallel rendering process using `ProcessPoolExecutor`, the worker initializer (`_worker_init`), and the static rendering method (`render_ascii_frame_to_image_static`).
            *   Explain the font loading mechanism and the list of fonts attempted.
            *   Describe the `_create_video_from_frames` method and the use of `mp4v` codec for Windows compatibility.
        *   **Utility Functions (`utils.py`):**
            *   Add a new section for [`utils.py`](utils.py).
            *   Document each function: `check_ffmpeg_installed`, `create_directory_if_not_exists`, `get_video_info`, and `is_admin`, explaining their purpose.
        *   **Test Suite (`test_suite.py`):**
            *   Add a new section for the test suite.
            *   Describe the purpose of [`test_suite.py`](test_suite.py) and the types of tests included (comparison, error handling, performance, profiling).
            *   Briefly explain the helper functions like `run_command`, `corrupt_video`, `run_conversion`, `compare_images`, and `extract_frames`.
            *   Mention the use of `unittest` for structuring the tests.
        *   **Test Video Creation (`create_test_video.py`):**
            *   Add a new section for [`create_test_video.py`](create_test_video.py).
            *   Describe its purpose in generating test video files.
            *   Explain the `create_test_video` function and its parameters.
        *   **Resource Monitoring (`monitor_resources.py`):**
            *   Add a new section for [`monitor_resources.py`](monitor_resources.py).
            *   Describe its purpose in monitoring resource usage during command execution.
            *   Explain the `monitor_resources` and `plot_resource_usage` functions and their arguments.
            *   Mention the use of `psutil` and `matplotlib`.

3.  **Review and Refine:**
    *   Read through the updated documentation files to ensure clarity, accuracy, and consistency.
    *   Check for any remaining discrepancies between the code and documentation.
    *   Ensure the language and tone are appropriate for the intended audience (users and developers).

**Mermaid Diagram for Project Structure:**

```mermaid
graph TD
    A[Video to Japanese ASCII Art Converter] --> B(main.py)
    A --> C(video_processor.py)
    A --> D(ascii_converter.py)
    A --> E(character_mapper.py)
    A --> F(renderer.py)
    A --> G(utils.py)
    A --> H(requirements.txt)
    A --> I(README.md)
    A --> J(projectSpecs.md)
    A --> K(test_suite.py)
    A --> L(create_test_video.py)
    A --> M(monitor_resources.py)
    A --> N(test_requirements.txt)
    A --> O(custom_comparison_*/)
    A --> P(debug_output/)
    A --> Q(test_output/)
    A --> R(test_videos/)

    B --> C
    B --> D
    B --> F
    B --> G

    C --> G

    D --> E

    F --> G

    K --> C
    K --> G
    K --> L
    K --> M
    K --> N

    M --> G