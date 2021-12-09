from rembg.bg import remove
import numpy as np
import io
from PIL import Image
from PIL import ImageFile

input_path = '13d.jpg'
output_path = 'out.png'

# Uncomment the following line if working with trucated image formats (ex. JPEG / JPG)
ImageFile.LOAD_TRUNCATED_IMAGES = True

f = np.fromfile(input_path)
result = remove(f)
img = Image.open(io.BytesIO(result)).convert("RGBA")
img.save(output_path)