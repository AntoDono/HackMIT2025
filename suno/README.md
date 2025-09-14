# Suno Music Generator

A Python script that generates music using the Suno API. Create custom music tracks with just a few command-line arguments!

## Features

- Generate music from text prompts
- Customize track title, tags, and style
- Generate instrumental tracks (no vocals) by default
- Save tracks to a specified output directory
- Monitor generation progress with real-time status updates
- Simple programmatic API for integration with other Python code

## Prerequisites

- Python 3.6+
- `requests` package
- Suno API key

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/suno-music-generator.git
   cd suno-music-generator
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python suno_music_generator.py --prompt "A description of the music you want to generate"
```

### All Options

```bash
python suno_music_generator.py \
  --prompt "A beautiful piano piece" \
  --title "My Composition" \
  --tags "piano,classical,relaxing" \
  --instrumental \
  --output-dir "output" \
  --max-attempts 30 \
  --poll-interval 10
```

### Arguments

| Argument            | Required | Default          | Description                                                                  |
| ------------------- | -------- | ---------------- | ---------------------------------------------------------------------------- |
| `--prompt`          | Yes      | -                | Description of the song you want to generate                                 |
| `--api-key`         | No       | Your default key | Your Suno API key                                                            |
| `--title`           | No       | "Generated Song" | Title of the generated song                                                  |
| `--tags`            | No       | ""               | Comma-separated tags for the song (e.g., "pop,happy,summer")                 |
| `--no-instrumental` | No       | False            | Include this flag to generate music with vocals (default: instrumental only) |
| `--output-dir`      | No       | "output"         | Directory to save the generated audio file                                   |
| `--max-attempts`    | No       | 30               | Maximum number of status check attempts                                      |
| `--poll-interval`   | No       | 10               | Seconds to wait between status checks                                        |

## Usage

### Command Line Interface

#### Generate a Relaxing Piano Track
```bash
python suno_music_generator.py \
  --prompt "A calming piano composition with soft strings" \
  --title "Peaceful Moments" \
  --tags "piano,relaxing,instrumental"
```

#### Generate an Upbeat Pop Song with Vocals
```bash
python suno_music_generator.py \
  --prompt "An upbeat pop song with catchy melodies and vocals" \
  --title "Summer Vibes" \
  --tags "pop,summer,vocals" \
  --no-instrumental
```

### Python API

```python
from suno_music_generator import SunoMusicGenerator

# Initialize with your API key
generator = SunoMusicGenerator(api_key="your_api_key_here")

# Generate and download music
result = generator.generate_and_download(
    prompt="An epic orchestral piece with powerful brass and strings",
    title="Epic Journey",
    tags="orchestral,epic,soundtrack",
    make_instrumental=True,
    output_dir="my_music"
)

if result['success']:
    print(f"🎵 Music saved to: {result['filepath']}")
else:
    print(f"❌ Error: {result.get('message')}")
```

### Generate an Instrumental Electronic Track
```bash
python suno_music_generator.py \
  --prompt "An upbeat pop song with catchy melodies and electronic elements" \
  --title "Summer Vibes" \
  --tags "pop,summer,happy" \
  --output-dir "my_music"
```

## Output

Generated audio files will be saved in the specified output directory (default: `output/`) with filenames based on the prompt.

## Error Handling

The script provides detailed error messages for common issues:
- Invalid API key
- Network errors
- Generation failures
- Timeout errors

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Suno API](https://suno.com) for the music generation service
- Python `requests` library for HTTP requests
