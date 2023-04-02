import os
import re
import csv
import cv2
import glob
import json
import requests
import pandas as pd
import streamlit as st
from pytube import YouTube
from bs4 import BeautifulSoup

invalid_chars = '<>:"/\\|?*'
download_dir = 'data'

def format_bytes(byte_str):
    selected_unit = bytes
    byte_count = int(byte_str)
    for unit in ['KB', 'MB', 'GB']:
        if byte_count >= 1024:
            byte_count /= 1024
            selected_unit = unit
    
    return f"{byte_count:.2f} {selected_unit}"

def progress_function(chunk, file_handle, byte_remainings):
    current = ((file_size - byte_remainings)/file_size)*100
    progress_bar.progress(int(current))

def get_heatmap(video:str):
    soup = BeautifulSoup(requests.get(video).text, "html.parser")
    data = re.search(r"var ytInitialData = ({.*?});", soup.prettify()).group(1)
    data = json.loads(data)
    data = data['playerOverlays']['playerOverlayRenderer']['decoratedPlayerBarRenderer']['decoratedPlayerBarRenderer']['playerBar']['multiMarkersPlayerBarRenderer']['markersMap']
    data = data[len(data)-1]['value']['heatmap']['heatmapRenderer']['heatMarkers']
    heatmap = {}
    for id, item in enumerate(data):
        heatmap[id] = item['heatMarkerRenderer']
    df = pd.DataFrame(heatmap).T
    return df

def download(url:str):
    video = YouTube(url, on_progress_callback=progress_function)
    video_type = video.streams.filter(progressive = True, file_extension = "mp4").get_highest_resolution()
    global file_size
    file_size = video_type.filesize
    title = video.title
    
    for char in invalid_chars:
        title = title.replace(char, "")

    st.write(title)
    st.write(f"file_size {format_bytes(file_size)}")
    global progress_bar
    progress_bar = st.progress(0)
    video_type.download(download_dir, filename=f"{title}.mp4")
    try:
        heatmap = get_heatmap(video=url)
        heatmap.to_csv(f"{download_dir}/{title}.csv", index=False)
    except Exception as e:
        st.write(f"Heatmap is not available for this video.")

def transfrom_data(path:str):
    video_path = f"data/{path}.mp4"
    
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    heat_map = pd.read_csv(f"data/{path}.csv")

    frame_num = 0
    results = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        time_ms = frame_num * (1000 / frame_rate)
        rows = heat_map[(heat_map['timeRangeStartMillis'] <= time_ms) & (time_ms < heat_map['timeRangeStartMillis'] + heat_map['markerDurationMillis'])]
        avg_heat = rows['heatMarkerIntensityScoreNormalized'].mean()
        results.append([frame_num, avg_heat])
        frame_num += 1

    results_df = pd.DataFrame(results, columns=['frame_num', 'heat'])
    results_df.to_csv(f"data/{path}_heatmap_final.csv",index=False)

    st.write(path," Frame rate: ", frame_rate)


def main():
    st.title("Get Youtube's Most Replayed Heatmap")

    video_id = st.text_input("Enter a video url or id:")

    file = st.file_uploader("Upload file", type=["csv", "txt"])
    url_list = None
    if file is not None:
        csv_reader = list(csv.reader(file.read().decode('utf-8').splitlines()))
        url_list = [row[0] for row in csv_reader]
        st.write(url_list)

    if st.button("Download"):
        if url_list is None:
            url = video_id if "https://www.youtube.com/watch?v=" in video_id else "https://www.youtube.com/watch?v=" + video_id
            download(url)
        else:
            total_progress_bar = st.progress(0)
            progress_status = st.text(f"Downloading...  0/{len(url_list)} Done")
            for idx, url in enumerate(url_list):
                
                with st.container():
                    st.markdown("""---""")
                    st.write(url)
                    download(url)
                    st.markdown("""---""")
                total_progress_bar.progress(int((idx+1)*100/len(url_list)))
                progress_status.text(f"Downloading...  {idx+1}/{len(url_list)} Done")

    if st.button("Transform"):
        csv_files = glob.glob('data/*.csv')
        files = []

        for file_path in csv_files:
            file = os.path.basename(file_path)
            file = os.path.splitext(file)[0]
            transfrom_data(file)
            files.append(file)

        st.write(files)
        



if __name__ == '__main__':
    main()
