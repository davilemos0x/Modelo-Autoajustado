import glob
from PIL import Image

# filepaths
fp_in = r"C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\Resultados\*.png"
fp_out = r"C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\Resultados\Resultado.gif"

img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=1100, loop=1)