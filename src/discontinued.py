#[] organize the libs importeds
from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import sys
#import scikit-learn
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import localization

car_image = imread(sys.argv[1], as_gray=True)
# it should be a 2 dimensional array
#print(car_image.shape)

# a grey scale pixel in skimage ranges between 0 & 1.
# multiplying it with 255 will make it range between 0 & 255
gray_car_image = car_image * 255
#fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1.imshow(gray_car_image, cmap="gray")

threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
#ax2.imshow(binary_car_image, cmap="gray")
#plt.show()


P=localization.Project(mode='2D', solver='CCA')
P.add_anchor('binary_car_image', binary_car_image)
P.add_anchor('gray_car_image', gray_car_image)

t,label=P.add_target()

t.add_measure('binary_car_image', 50)
t.add_measure('gray_car_image', 50)

print(t)

print(label)

P.solve()

'''
# this gets all the connected regions and groups them together
#label_image = measure.label(localization.binary_car_image)
fig, (ax1) = plt.subplots(1)
#ax1.imshow(localization.gray_car_image, cmap="gray")

# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label):
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue

    # the bounding box coordinates
    minRow, minCol, maxRow, maxCol = region.bbox
    rectBorder = patches.Rectangle((minCol, minRow), maxCol-minCol, maxRow-minRow, edgecolor="red", linewidth=2, fill=False)
    ax1.add_patch(rectBorder)
    # let's draw a red rectangle over those regions

plt.show()
'''