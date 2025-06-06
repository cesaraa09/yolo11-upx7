import yt_dlp

url = "https://www.youtube.com/watch?v=ByED80IKdIU"

ydl_opts = {
    'quiet': True,
    'no_warnings': True,
    'format': 'best',
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=False)
    stream_url = info['url']
    print("URL direta:", stream_url)
