import os
import subprocess

genres_dict = {
    'blues': '1920s 1930s delta blues acoustic solo, traditional country blues guitar, Robert Johnson style blues audio',
    'classical': 'classical orchestra symphony',
    'country': 'country music song',
    'disco': 'disco 70s dance music',
    'hiphop': 'hiphop rap audio',
    'jazz': 'jazz saxophone trumpet instrumental',
    'metal': 'heavy metal rock audio',
    'pop': 'pop music hits audio',
    'reggae': 'reggae music rhythm',
    'rock': 'classic rock song'
}

def download_music():
    for genre, query in genres_dict.items():
        folder_path = f"test_data_1/{genre}"
        os.makedirs(folder_path, exist_ok=True)
        
        print(f"loading {genre.upper()}")
        search_query = f"ytsearch20:{query} -live -mix -album"
        
        command = [
            'yt-dlp',
            '-x', '--audio-format', 'wav',
            '--output', f'{folder_path}/%(title)s.%(ext)s',
            '--no-playlist',
            '--ignore-errors',              
            '--max-downloads', '10',       
            '--match-filter', 'duration > 60 & duration < 420',
            '--postprocessor-args', 'ffmpeg:-ar 22050 -ac 1',
            search_query
        ]
        subprocess.run(command)
if __name__ == "__main__":
    download_music()
    print("Music download completed.")


