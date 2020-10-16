from skimage.exposure import cumulative_distribution
import matplotlib.pylab as plt
import numpy as np
import cv2
from PIL import Image



img=cv2.imread("data/img1.jpg",0)
img_ref=cv2.imread("data/img2.jpg",0)




def applyGamma(img):
    gamma_value=[0.2,0.4,0.8,2.4,2.8,3.2,3.6,4.0]
    for i in range(len(gamma_value)):
        image=np.array(255*(img/255)**gamma_value[i],dtype='uint8')
        nw_img=Image.fromarray(np.uint8(image),'L')
        filename="img"+str(i)+".jpg"
        nw_img.save(filename)





def freq(img):
    mp=[0]*256
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            mp[int(img[i][j])]=mp[int(img[i][j])]+1
    return mp



def pdf(hist):
    pb=[0]*256
    for i in range(len(pb)):
        pb[i]=hist[i]/(img.shape[0]*img.shape[1])
    
    return pb





def cdf(pdf):
    cdf=[0]*256
    sum=0
    for i in range(len(cdf)):
        sum=sum+pdf[i]
        cdf[i]=round(sum*255)
    
    return cdf




def helper(p1,p2):
    dif=p1-p2
    mask=np.ma.less_equal(dif,-1)
    # print(mask)
    if np.all(mask):
        tmp=np.abs(diff).argmin()
        return tmp
    
    masked_diff=np.ma.masked_array(dif,mask)
    return masked_diff.argmin()



img_shape=img.shape
img_ref_shape=img_ref.shape

flat_img=img.ravel()
flat_img_ref=img_ref.ravel()

img_values,img_idx,img_counts=np.unique(flat_img,return_inverse=True,return_counts=True)
# print("--------------------------------------")
# print(img_values)
# print("--------------------------------------")
# print(img_idx)
# print("--------------------------------------")
# print(img_counts)
# print("--------------------------------------")


imgref_values,imgref_counts=np.unique(flat_img_ref,return_counts=True)

img_cdf=np.cumsum(img_counts).astype(np.float64)
img_cdf=img_cdf/img_cdf[-1]


imgref_cdf=np.cumsum(imgref_counts).astype(np.float64)
imgref_cdf=imgref_cdf/imgref_cdf[-1]


img1=np.round(img_cdf*255)
img2=np.round(imgref_cdf*255)

mp=[]

for i in img1[:]:
    mp.append(helper(img2,i))

mp=np.array(mp,dtype='uint8')
# print(mp)

# nw_img=mp[img_idx].reshape(img_shape)
nw_img=imgref_values[mp[img_idx]].reshape(img_shape)
# print(nw_img)
# print(img)

# cv2.imshow('nw_img',np.array(nw_img,dtype='uint8'))
# cv2.imwrite("nw_img.jpg",nw_img)
# cv2.imshow('img',img)
# cv2.imwrite("img.jpg",img)
# cv2.imshow('ref',img_ref)
# cv2.imwrite("ref.jpg",img_ref)


given_img_hist=freq(img)
given_img_pdf=pdf(given_img_hist)
given_img_cdf=cdf(given_img_pdf)




# if you want to see the given img cdf plot uncomment the below code

plt.bar(range(256),given_img_cdf)
plt.savefig("given_img_cdf.png")
plt.show()
plt.clf()


refimg_hist=freq(img_ref)
refimg_pdf=pdf(refimg_hist)
refimg_cdf=cdf(refimg_pdf)

# if you want to see the reference img cdf uncomment the below code

plt.bar(range(256),refimg_cdf)
plt.savefig("refimg_cdf.png")
plt.show()
plt.clf()



nw_img_hist=freq(nw_img)
nw_img_pdf=pdf(nw_img_hist)
nw_img_cdf=cdf(nw_img_pdf)


# if you want to see the new img cdf uncomment the below code


plt.bar(range(256),nw_img_cdf)
plt.savefig("nwimg_cdf.png")
plt.show()
plt.clf()


applyGamma(img)

