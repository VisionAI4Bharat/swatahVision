import swatahvision as sv
import cv2

# Get image path using your asset system
path = sv.Assets.Image("CAR")

print("Image Path:", path)

# Read image
img = cv2.imread(path)

img = cv2.resize(img, (800,500))

# Show image
cv2.imshow("Car Image", img)

# Wait until key is pressed
cv2.waitKey(0)

# Close window
cv2.destroyAllWindows()