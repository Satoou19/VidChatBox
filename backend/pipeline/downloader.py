import os
import re
import yt_dlp

class VideoDownloader:
    def __init__(self, data_dir="./backend/data"):
        self.data_dir = data_dir
        self.audio_dir = os.path.join(data_dir, "audio")
        os.makedirs(self.audio_dir, exist_ok=True)

    def _get_cookiefile_path(self):
        """Checks for cookies.txt file or writes YOUTUBE_COOKIES env var to a temp file if available.
        
        Returns the path to the cookies file, or None.
        """
        env_cookies = os.getenv("YOUTUBE_COOKIES")
        if env_cookies:
            cookies_path = os.path.join(self.data_dir, "cookies.txt")
            try:
                normalized_cookies = env_cookies.replace("\\n", "\n")
                with open(cookies_path, "w", encoding="utf-8") as f:
                    f.write(normalized_cookies)
                return cookies_path
            except Exception as e:
                print(f"Warning: Failed to write YOUTUBE_COOKIES to file: {e}")
                
        local_path = "./cookies.txt"
        if os.path.exists(local_path):
            return local_path
            
        data_path = os.path.join(self.data_dir, "cookies.txt")
        if os.path.exists(data_path):
            return data_path
            
        return None

    def download_subtitles(self, url, progress_callback=None):
        """Downloads subtitles/auto-captions from YouTube.
        
        Returns info dictionary containing segments parsed from subtitles.
        """
        # First extract metadata
        info = self.extract_info(url)
        video_id = info["id"]
        safe_video_id = re.sub(r'[^\w\-_]', '', video_id)
        
        # Temp directory for downloading subtitles
        subs_dir = os.path.join(self.data_dir, "subtitles")
        os.makedirs(subs_dir, exist_ok=True)
        
        # We template the filename: subs_dir/subs_<video_id>
        output_template = os.path.join(subs_dir, f"subs_{safe_video_id}.%(ext)s")
        
        # Clean any old subtitle files for this video
        if os.path.exists(subs_dir):
            for f in os.listdir(subs_dir):
                if f.startswith(f"subs_{safe_video_id}"):
                    try:
                        os.remove(os.path.join(subs_dir, f))
                    except:
                        pass
                    
        # Extract available languages from info
        subtitles = info.get("subtitles", {}) or {}
        auto_captions = info.get("automatic_captions", {}) or {}
        
        pref_langs = ['vi', 'en', 'ja', 'zh-Hans', 'zh-Hant', 'ko', 'fr', 'de', 'es']
        selected_lang = None
        matched_key = None
        is_auto = False
        
        # 1. Try manual subtitles first (safe from 429 translation rates)
        for lang in pref_langs:
            if lang in subtitles:
                selected_lang = lang
                matched_key = lang
                break
                
        # 2. Try automatic captions in the video's original language (prevents on-the-fly translation 429s)
        if not selected_lang:
            orig_lang = None
            for key in auto_captions.keys():
                if key.endswith("-orig"):
                    orig_lang = key.split("-")[0]
                    break
                    
            if orig_lang and orig_lang in auto_captions:
                selected_lang = orig_lang
                matched_key = orig_lang
                is_auto = True
            else:
                # Fallbacks
                if 'en' in auto_captions:
                    selected_lang = 'en'
                    matched_key = 'en'
                    is_auto = True
                elif auto_captions:
                    matched_key = list(auto_captions.keys())[0]
                    selected_lang = matched_key.split("-")[0]
                    is_auto = True
                    
        if not selected_lang or not matched_key:
            raise FileNotFoundError("Video does not have any manual subtitles or auto-captions available.")
            
        # Find the VTT URL
        target_dict = auto_captions if is_auto else subtitles
        lang_subs = target_dict.get(matched_key, [])
        vtt_url = None
        
        # We prefer vtt, but will fallback to json3 if necessary (though our parser requires vtt)
        for sub in lang_subs:
            if sub.get('ext') == 'vtt':
                vtt_url = sub.get('url')
                break
                
        if not vtt_url:
            raise FileNotFoundError(f"VTT subtitle format not available for language '{selected_lang}'.")
            
        try:
            if progress_callback:
                progress_callback("extracting_subtitles", 30)
                
            # Download the VTT file directly
            import urllib.request
            vtt_path = os.path.join(subs_dir, f"subs_{safe_video_id}.{selected_lang}.vtt")
            
            req = urllib.request.Request(vtt_url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(vtt_path, 'wb') as out_file:
                out_file.write(response.read())
                
            if not os.path.exists(vtt_path):
                raise FileNotFoundError("No subtitle files downloaded successfully.")
                
            # Parse the VTT file
            segments = self._parse_vtt(vtt_path)
            
            # Clean up local subtitle files and temp directory if empty
            try:
                os.remove(vtt_path)
            except:
                pass
                
            info["segments"] = segments
            return info
        except Exception as e:
                raise Exception(f"Failed to fetch subtitles: {str(e)}")

    def _parse_vtt(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        block_pattern = re.compile(
            r'(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3})[^\n]*\n(.*?)(?=\n\n|\n\d{2}:|\Z)',
            re.DOTALL
        )
        
        def parse_time(time_str):
            time_str = time_str.replace(',', '.')
            parts = time_str.split(':')
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds_parts = parts[2].split('.')
            seconds = int(seconds_parts[0])
            ms = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
            return hours * 3600 + minutes * 60 + seconds + ms / 1000.0

        # Remove HTML/style tags from VTT (e.g. <c> tags)
        clean_content = re.sub(r'<[^>]+>', '', content)
        clean_content = clean_content.replace('&nbsp;', ' ')
        
        segments = []
        for match in block_pattern.finditer(clean_content):
            start_str, end_str, text = match.groups()
            start = parse_time(start_str)
            end = parse_time(end_str)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                segments.append({
                    "start": start,
                    "end": end,
                    "text": text
                })
                
        # Deduplicate consecutive prefix subtitles (common in YouTube auto captions)
        deduped = []
        for seg in segments:
            text = seg["text"]
            if deduped:
                last = deduped[-1]
                if last["text"] == text:
                    last["end"] = max(last["end"], seg["end"])
                    continue
                if text.startswith(last["text"]) and len(text) > len(last["text"]):
                    last["text"] = text
                    last["end"] = max(last["end"], seg["end"])
                    continue
            deduped.append(seg)
            
        return deduped

    def extract_info(self, url):
        """Extracts video metadata using yt-dlp without downloading."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            # ignore_no_formats_error: if format selection finds nothing, warn
            # instead of raising. This lets process_video_result() complete and
            # populate automatic_captions/subtitles even when no video formats
            # are available (e.g. restricted formats, certain player clients).
            'ignore_no_formats_error': True,
            'skip_download': True,
            'youtube_include_dash_manifest': False,
            'youtube_include_hls_manifest': False,
        }
        cookie_file = self._get_cookiefile_path()
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
            
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                return {
                    "id": info.get("id"),
                    "title": info.get("title"),
                    "duration": info.get("duration"),
                    "uploader": info.get("uploader"),
                    "webpage_url": info.get("webpage_url"),
                    "extractor": info.get("extractor"),
                    "subtitles": info.get("subtitles", {}),
                    "automatic_captions": info.get("automatic_captions", {}),
                }
            except Exception as e:
                raise Exception(f"Failed to extract video info: {str(e)}")

    def download_audio(self, url, progress_callback=None):
        """Downloads audio track from YouTube/Twitch URL.
        
        Saves as raw m4a/webm to bypass ffmpeg requirement if missing.
        """
        # First get info to determine video ID
        info = self.extract_info(url)
        video_id = info["id"]
        
        # Clean video_id for filename safety
        video_id = re.sub(r'[^\w\-_]', '', video_id)
        
        # Output template: d:\PythonProject\VidChatBox\backend\data\audio\<video_id>
        output_template = os.path.join(self.audio_dir, f"audio_{video_id}.%(ext)s")
        
        # Progress hook for tracking percentage
        def ydl_hook(d):
            if d['status'] == 'downloading':
                # Extract percentage
                total = d.get('total_bytes') or d.get('total_bytes_estimate') or 1
                downloaded = d.get('downloaded_bytes', 0)
                percent = (downloaded / total) * 100
                if progress_callback:
                    progress_callback("downloading", percent)
            elif d['status'] == 'finished':
                if progress_callback:
                    progress_callback("finished", 100)

        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': output_template,
            'progress_hooks': [ydl_hook],
            'quiet': True,
            'no_warnings': True,
        }
        
        cookie_file = self._get_cookiefile_path()
        if cookie_file:
            ydl_opts['cookiefile'] = cookie_file
            
        # Note: If FFmpeg was available we could do postprocessors, but we intentionally avoid
        # forcing FFmpeg so that raw m4a/webm works fine.

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                download_info = ydl.extract_info(url, download=True)
                # Find the actual written filename
                filename = ydl.prepare_filename(download_info)
                # Ensure the file exists (extension might differ slightly based on bestaudio format downloaded)
                # e.g., if .webm or .m4a was downloaded, prepare_filename is correct
                
                # Double check actual file path on disk
                base_path, _ = os.path.splitext(filename)
                actual_path = None
                for ext in ['m4a', 'webm', 'mp3', 'opus', 'ogg', 'wav']:
                    test_path = f"{base_path}.{ext}"
                    if os.path.exists(test_path):
                        actual_path = test_path
                        break
                        
                if not actual_path and os.path.exists(filename):
                    actual_path = filename
                
                if not actual_path:
                    # Let's search in directory
                    for f in os.listdir(self.audio_dir):
                        if f.startswith(f"audio_{video_id}"):
                            actual_path = os.path.join(self.audio_dir, f)
                            break
                            
                if not actual_path or not os.path.exists(actual_path):
                    raise FileNotFoundError("Downloaded audio file not found on disk.")
                    
                info["audio_path"] = actual_path
                return info
            except Exception as e:
                raise Exception(f"Failed to download audio: {str(e)}")

# Quick manual testing block
if __name__ == "__main__":
    import sys
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
    
    print(f"Testing downloader with URL: {test_url}")
    downloader = VideoDownloader()
    
    def cb(status, percent):
        print(f"Status: {status} | Progress: {percent:.2f}%")
        
    try:
        res = downloader.download_audio(test_url, progress_callback=cb)
        print("Success! Download details:")
        print(res)
    except Exception as e:
        print("Error during download:", e)
