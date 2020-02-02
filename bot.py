import telebot
import os
from scipy.io import wavfile
import scipy.signal as sps
import subprocess
import numpy as np
import cv2
import io
from PIL import Image

API_TOKEN = '951549634:AAGK2LCjOCC8Ksm6cb7n3B564tY6liGkyOA'
# New sampling rate
NEW_RATE = 16000
# Path to data
SRC_DATA = r'./data'

# Dictionary {user_id : [audio_message_0, audio_message_1, ..], ..}
storage = {}
# Create bot
bot = telebot.TeleBot(API_TOKEN)

""" Convert audio message's frequency with NEW_RATE

    :parameter
    user_id: current used ID
    audio_dir: directory to audio message which needs to be converted
    
    Result:
    Save converted audio message
"""
def convert_audio_frequency(user_id, audio_dir):
    current_src, file_count = create_dir(user_id, identification='audio_after_frequency')
    new_file_name = current_src + '/' f'audio_message_{file_count}.wav'

    process = subprocess.run(['ffmpeg', '-i', audio_dir, new_file_name])
    if process.returncode != 0:
        raise Exception("Something went wrong")

    sampling_rate, data = wavfile.read(new_file_name)
    # Resample data
    num_of_samples = round(len(data) * float(NEW_RATE) / sampling_rate)
    data = sps.resample(data, num_of_samples)

    wavfile.write(new_file_name, sampling_rate, data)
    print("[INFO] Saved audio message after converting frequency")


""" Create a directory for user if it doesn't exists

    :parameter
    user_id: current user ID

    :returns 
    new_dir: directory for current user
    file_count: number of files in current directory
"""
def create_dir(user_id, identification):
    new_dir = SRC_DATA + '/' + identification + '/' + str(user_id)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    # Count files in directory
    path, dirs, files = next(os.walk(new_dir))
    file_count = len(files)

    return new_dir, file_count


@bot.message_handler(content_types=['voice'])
def voice_processing(message):
    # User ID identification
    user_id = message.from_user.id
    if user_id not in storage:
        storage[user_id] = []

    current_src, file_count = create_dir(user_id, identification='audio_before_frequency')

    # Get voice message
    file_info = bot.get_file(message.voice.file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    # Update audio dictionary
    storage[user_id].append(f'audio_message_{file_count}')

    # Save downloaded voice file to dir in .wav format
    downloaded_file_name = current_src + '/' f'audio_message_{file_count}.wav'
    with open(downloaded_file_name, 'wb') as write_audio:
        write_audio.write(downloaded_file)
        print("[INFO] Saved audio message")

    convert_audio_frequency(user_id, downloaded_file_name)


@bot.message_handler(content_types=['photo'])
def photo_processing(message):
    # User ID identification
    user_id = message.from_user.id

    # Get photo
    file_info = bot.get_file(message.photo[-1].file_id)
    downloaded_file = bot.download_file(file_info.file_path)

    photo = Image.open(io.BytesIO(downloaded_file))

    # Convert to grey color
    image = np.array(photo)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(cv2.haarcascades +
                                        "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    print("[INFO] Found {0} face(-s)".format(len(faces)))

    # Save photo if detected face(-s)
    if len(faces) > 0:
        current_src, file_count = create_dir(user_id, identification='images')
        photo_name = current_src + '/' + f'photo_with_faces_{file_count}.jpg'
        cv2.imwrite(photo_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        print("[INFO] Saved photo".format(len(faces)))


if __name__ == "__main__":
    bot.polling(none_stop=True)
