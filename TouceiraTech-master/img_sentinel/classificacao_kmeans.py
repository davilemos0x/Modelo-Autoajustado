import rasterio as rio
from rasterio.plot import show
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mc


img = rio.open(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\imgs\Abril2020\05042020\b4_0504_recorte_areatotal.tif')

img.meta['dtype'] = "int16"
print(img.meta)
'''
with rio.open(img) as src:
    img_data = src.read()
    img_meta = src.profile
'''   

'''
img_raster = img.read() # read the opened image
vmin, vmax = np.nanpercentile(img_raster, (0,100))  # 5-95% contrast stretch

plt.figure(figsize=[20,20])
show(img, cmap='gray', vmin=vmin, vmax=vmax)
plt.show()
'''
# create an empty array with same dimension and data type
imgxyb = np.empty((img.height, img.width, img.count), 'uint16')
# loop through the raster's bands to fill the empty array
for band in range(imgxyb.shape[2]):
    imgxyb[:,:,band] = img.read(band+1)
    
print(imgxyb.shape)

# convert to 1d array
img1d=imgxyb[:,:,:3].reshape((imgxyb.shape[0]*imgxyb.shape[1],imgxyb.shape[2]))

print(img1d.shape)

cl = cluster.KMeans(n_clusters=8) # create an object of the classifier
param = cl.fit(img1d) # train it
img_cl = cl.labels_ # get the labels of the classes
img_cl = img_cl.reshape(imgxyb[:,:,0].shape) # reshape labels to a 3d array (one band only)

# Create a custom color map to represent our different 4 classes
cmap = mc.LinearSegmentedColormap.from_list("", ["black", "red","green","yellow"])
# Show the resulting array and save it as jpg image
plt.figure(figsize=[20,20])
plt.imshow(img_cl, cmap=cmap)
plt.axis('off')
#plt.savefig("elhas_clustered.jpg", bbox_inches='tight')
plt.show()

#########################################################################################################################################################


# open the raster image
img2 = rio.open(r'C:\Users\Ânderson Fischoeder\Desktop\img_sentinel\imgs\Janeiro2020\16012020\b4_1601_recorte_areatotal.tif')
# create an empty array with same dimensions and data type 
imgxyb2 = np.empty((img2.height, img2.width, img2.count), 'uint16')
# loop through the raster bands and fill the empty array in x-y-bands order
for band in range(imgxyb2.shape[2]):
    imgxyb2[:,:,band] = img2.read(band+1)
# convert to 1d array
img1d2 = imgxyb2[:,:,:3].reshape(imgxyb2.shape[0]*imgxyb2.shape[1], imgxyb2.shape[2])
# predict the clusters in the image 

pred = cl.predict(img1d2)

# reshape the 1d array predictions to x-y-bands shape order (only one band)
cul = pred
cul = cul.reshape(imgxyb2[:,:,0].shape)

img2arr = img2.read() # Read the image
vmin, vmax = np.nanpercentile(img2arr, (0,100)) # 5–95% contrast stretch
# show the original and predicted image
fig, (ax1,ax2) = plt.subplots(figsize=[15,15], nrows=1,ncols=2, sharey=False,)
show(img2, cmap='gray', vmin=vmin, vmax=vmax, ax=ax1)
show(cul, cmap=cmap, ax=ax2)
ax1.set_axis_off()
ax2.set_axis_off()
#fig.savefig("pred.png", bbox_inches='tight')
plt.show()
