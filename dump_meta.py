from PIL import Image
from PIL.ExifTags import TAGS
import os

UPLOAD_DIR = "/Users/devanshpatel/Devansh/project/Deepfake AI/uploads"
filename = "AWS Training & Certification.png"
file_path = os.path.join(UPLOAD_DIR, filename)

img = Image.open(file_path)
print(f"--- Info for {filename} ---")
print(f"Format: {img.format}")
print(f"Info: {img.info}")

exif = img._getexif()
if exif:
    print("\n--- EXIF ---")
    for tag, value in exif.items():
        tag_name = TAGS.get(tag, tag)
        print(f"{tag_name}: {value}")
else:
    print("\nNo EXIF found.")
