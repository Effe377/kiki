import streamlit as st
import os
import re
import tempfile
from moviepy.editor import VideoFileClip

# Directory where sign language videos are stored
SIGN_VIDEO_DIR = 'C:/Users/akogo/Desktop/Sign- Link/videos'  # Update this with your actual path

# Function to find video corresponding to a phrase
def get_video_for_phrase(phrase):
    # Replace spaces with underscores to match file naming convention
    phrase_cleaned = phrase.replace(' ', '_').lower()  # Match the video file format
    # Ensure forward slashes in path to avoid issues with backslashes
    video_path = os.path.join(SIGN_VIDEO_DIR, f"{phrase_cleaned}.mp4").replace("\\", "/")
    
    # Debugging print statement to check the constructed path
    print(f"Looking for video at: {video_path}")
    
    if os.path.exists(video_path):
        print(f"Found video: {video_path}")
        return video_path
    else:
        print(f"Video not found: {video_path}")
        return None

# Function to get the video for a single phrase
def get_sign_language_video(text):
    phrase = text.strip()  # Clean the phrase (remove any leading/trailing spaces)
    phrase_with_underscores = phrase.replace(' ', '_')  # Convert spaces to underscores
    video_path = get_video_for_phrase(phrase_with_underscores)

    if video_path:
        return VideoFileClip(video_path)
    else:
        st.warning(f"No sign language video found for phrase: '{phrase}'")
        return None

# Streamlit app layout
st.title("Text-to-Sign Language Converter")

# Text input for user to enter a single phrase
text_input = st.text_input("Enter a phrase to convert to sign language:")

if text_input:
    st.write(f"Converting: {text_input}")
    
    # Convert the text to a single sign language video
    video_clip = get_sign_language_video(text_input)
    
    if video_clip:
        # Save the video to a temporary file and display it
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        video_clip.write_videofile(temp_file.name, codec="libx264")
        st.video(temp_file.name)
    else:
        st.error("No video found for the given input.")
