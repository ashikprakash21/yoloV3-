import detect_obj_2
import cv2


image = detect_obj_2.image_BGR

cv2.namedWindow("original image", cv2.WINDOW_NORMAL) 
cv2.imshow("original image",image)
# cv2.waitKey(0)

#blob from image
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416,416), crop=False, swapRB=True)

print("#"*100)
print("OG image shape = {} and blob image shape = {}".format(image.shape, blob.shape))


#transposing the blob  after slicing to make channel in last position
blob_to_show = blob[0,:,:,:].transpose(1,2,0)
print("&"*100)
print(blob_to_show.shape)

# cv2.namedWindow("Blob image", cv2.WINDOW_NORMAL)
# cv2.imshow("Blob image", cv2.cvtColor(blob_to_show, cv2.COLOR_RGB2BGR))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


