import os
import json
import time
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List


class SunoMusicGenerator:
    BASE_URL = "https://studio-api.prod.suno.com/api/v2/external/hackmit"

    def __init__(self, api_key: str):
        """Initialize the SunoMusicGenerator with your API key.

        Args:
            api_key: Your Suno API key
        """
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def generate_music(self,
                       prompt: str,
                       title: str = "Generated Song",
                       tags: str = "",
                       make_instrumental: bool = True) -> Dict[str, Any]:
        """
        Generate music using Suno API.

        Args:
            prompt: Description of the song you want to generate
            title: Title of the song (default: "Generated Song")
            tags: Comma-separated tags for the song (e.g., "pop, happy, summer")
            make_instrumental: Whether to generate instrumental music (default: True)

        Returns:
            Dictionary containing the API response with clip information
        """
        url = f"{self.BASE_URL}/generate"
        payload = {
            "prompt": prompt,
            "title": title,
            "tags": tags,
            "instrumental": make_instrumental,
            "model": "V4_5"
        }

        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()

            # The API returns a single clip object, but we'll wrap it in a list for consistency
            result = response.json()
            if isinstance(result, dict) and 'id' in result:
                return {"success": True, "clips": [result]}
            return result

        except requests.exceptions.RequestException as e:
            print(f"Error generating music: {e}")
            return {"success": False, "error": str(e)}

    def check_status(self, clip_ids: List[str]) -> Dict[str, Any]:
        """
        Check the status of one or more audio clips.

        Args:
            clip_ids: List of clip IDs to check

        Returns:
            Dictionary containing the status of the clips
        """
        url = f"{self.BASE_URL}/clips"
        params = {"ids": ",".join(clip_ids)}

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return {"success": True, "clips": response.json()}
        except requests.exceptions.RequestException as e:
            print(f"Error checking status: {e}")
            return {"success": False, "error": str(e)}

    def get_audio(self, audio_id: str) -> Optional[bytes]:
        """
        Download the generated audio file.

        Args:
            audio_id: The ID of the generated audio

        Returns:
            Audio data as bytes if successful, None otherwise
        """
        # First check the status to get the audio URL
        status = self.check_status([audio_id])

        if not status.get("success") or not status.get("clips"):
            print(
                f"Error getting audio status: {status.get('error', 'Unknown error')}")
            return None

        clip = status["clips"][0]  # Get the first clip

        if clip.get("status") != "complete":
            print(f"Audio is not ready yet. Status: {clip.get('status')}")
            return None

        audio_url = clip.get("audio_url")
        if not audio_url:
            print("No audio URL found in the response")
            return None

        try:
            response = requests.get(audio_url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error downloading audio: {e}")
            return None

    def save_audio(self, audio_data: bytes, filename: str, output_dir: str = "output") -> str:
        """
        Save audio data to a file.

        Args:
            audio_data: Audio data as bytes
            filename: Name of the output file (with extension)
            output_dir: Directory to save the file (default: "output")

        Returns:
            Path to the saved file or empty string if failed
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            filepath = os.path.join(output_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(audio_data)

            return os.path.abspath(filepath)

        except Exception as e:
            print(f"Error saving audio file: {e}")
            return ""

    def generate_and_download(self,
                              prompt: str,
                              title: str = "Generated Song",
                              tags: str = "",
                              make_instrumental: bool = True,
                              output_dir: str = "output",
                              max_attempts: int = 30,
                              poll_interval: int = 10) -> Dict[str, Any]:
        """
        Generate and download a music track in one method call.

        Args:
            prompt: Description of the song you want to generate
            title: Title of the generated song
            tags: Comma-separated tags for the song
            make_instrumental: Whether to generate instrumental music
            output_dir: Directory to save the audio file
            max_attempts: Maximum number of status check attempts
            poll_interval: Seconds to wait between status checks

        Returns:
            Dictionary containing the result with 'success' status and 'filepath' if successful
        """
        result = {
            'success': False,
            'message': '',
            'filepath': None,
            'clip_id': None
        }

        try:
            # Generate the music
            gen_result = self.generate_music(
                prompt=prompt,
                title=title,
                tags=tags,
                make_instrumental=make_instrumental
            )

            if not gen_result.get("success"):
                result['message'] = f"Failed to generate music: {gen_result.get('error', 'Unknown error')}"
                return result

            if not gen_result.get("clips"):
                result['message'] = "No clip information in the response"
                return result

            clip = gen_result["clips"][0]
            clip_id = clip.get("id")
            result['clip_id'] = clip_id

            if not clip_id:
                result['message'] = "No clip ID in the response"
                return result

            # Poll for completion
            for attempt in range(max_attempts):
                status = self.check_status([clip_id])

                if not status.get("success") or not status.get("clips"):
                    result['message'] = f"Error checking status: {status.get('error', 'Unknown error')}"
                    return result

                clip_status = status["clips"][0].get("status")
                print(
                    f"\r🔄 Status: {clip_status.capitalize()} (Attempt {attempt + 1}/{max_attempts}) ", end="", flush=True)

                if clip_status == "complete":
                    print("\n✅ Audio generation complete!")
                    break
                elif clip_status == "failed":
                    result['message'] = "Audio generation failed"
                    return result

                time.sleep(poll_interval)
            else:
                result['message'] = "Timed out waiting for audio generation"
                return result

            # Download the audio
            print("\n⬇️  Downloading audio...")
            audio_data = self.get_audio(clip_id)

            if audio_data:
                # Use the title from the status if available, otherwise use the provided title
                clip_title = status["clips"][0].get("title", title)
                filename = f"{clip_title.replace(' ', '_').lower()}.mp3"
                saved_path = self.save_audio(audio_data, filename, output_dir)

                if saved_path:
                    result.update({
                        'success': True,
                        'message': 'Audio generated and saved successfully',
                        'filepath': saved_path
                    })
                    return result

            result['message'] = "Failed to download audio"
            return result

        except Exception as e:
            result['message'] = f"An error occurred: {str(e)}"
            return result


def main():
    """Command-line interface for the Suno Music Generator."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate music using Suno API')

    # Required arguments
    parser.add_argument('--prompt', type=str, required=True,
                        help='Description of the song you want to generate')

    # Optional arguments
    parser.add_argument('--api-key', type=str, default="ddf6ac923c954382b0700a5d5368b434",
                        help='Your Suno API key')
    parser.add_argument('--title', type=str, default="Generated Song",
                        help='Title of the generated song')
    parser.add_argument('--tags', type=str, default="",
                        help='Comma-separated tags for the song (e.g., "pop,happy,summer")')
    parser.add_argument('--no-instrumental', dest='instrumental', action='store_false',
                        help='Generate music with vocals (default: instrumental only)')
    parser.set_defaults(instrumental=True)
    parser.add_argument('--output-dir', type=str, default="output",
                        help='Directory to save the generated audio file')
    parser.add_argument('--max-attempts', type=int, default=30,
                        help='Maximum number of status check attempts')
    parser.add_argument('--poll-interval', type=int, default=10,
                        help='Seconds to wait between status checks')

    args = parser.parse_args()

    # Initialize the generator
    generator = SunoMusicGenerator(args.api_key)

    # Generate and download the music
    result = generator.generate_and_download(
        prompt=args.prompt,
        title=args.title,
        tags=args.tags,
        make_instrumental=args.instrumental,
        output_dir=args.output_dir,
        max_attempts=args.max_attempts,
        poll_interval=args.poll_interval
    )

    # Print results
    if result['success']:
        print(f"\n🎉 {result['message']}")
        print(f"📁 Saved to: {result['filepath']}")
        return 0
    else:
        print(f"\n❌ {result.get('message', 'Unknown error occurred')}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
