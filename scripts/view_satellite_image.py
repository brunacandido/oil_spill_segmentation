import rasterio
import matplotlib.pyplot as plt
import numpy as np

# Caminho da sua imagem (altere para o caminho real)
image_path = "C:/Users/Bruna/Downloads/S1A_IW_GRDH_1SDV_20170810T024714_20170810T024738_017855_01DEF7_F48C.SAFE/S1A_IW_GRDH_1SDV_20170810T024714_20170810T024738_017855_01DEF7_F48C.SAFE/measurement/s1a-iw-grd-vv-20170810t024714-20170810t024738-017855-01def7-001.tiff"

# Abrir a imagem com rasterio
with rasterio.open(image_path) as src:
    image = src.read(1)  # lê a primeira banda
    profile = src.profile
    print("CRS:", profile["crs"])
    print("Tamanho:", image.shape)

# Plotar a imagem com contraste ajustado
plt.figure(figsize=(10, 10))
plt.imshow(np.log1p(image), cmap='gray')  # log para realçar contraste SAR
plt.title("Sentinel-1 SAR - Oil Spill")
plt.axis("off")
plt.show()
