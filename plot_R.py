import cv2
image_name = '49.jpg'
cov_size = 20
image = cv2.imread(image_name)
image = cv2.resize(image, (80,324))
b, g, r = cv2.split(image)
row = r.shape[0]
cow = r.shape[1]
g = r
b = r
for i in range(1, row):
    for j in range(1, cow):
        if i%cov_size == 0:
            r[i][j] = 25
            g[i][j] = 155
            b[i][j] = 155
        if j%cov_size == 0:
            r[i][j] = 25
            g[i][j] = 155
            b[i][j] = 155
img2 = cv2.merge([r, g, b])
cv2.imwrite(image_name.split('.')[0] + '_R.jpg', img2)
