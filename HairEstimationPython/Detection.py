#! /usr/bin/env python3

import cv2
import numpy as np
from matplotlib import pyplot as plt


def show(title, image):
    resize = True
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))
    if resize:
        image = cv2.resize(image, (int(width * 0.2), int(height * 0.2)))
    cv2.imshow(title, image)


def found(img):
    # red_channel = img[:,:,1]
    sum = np.count_nonzero(img == 255)
    all = np.size(img)

    print('hairpixels:', sum)
    print('imagepixels:', all)
    print('percentage:', (sum / all) * 100)

    return sum

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy() # don't clobber original
    skel = img.copy()

    skel[:,:] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp  = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:,:] = eroded[:,:]
        if cv2.countNonZero(img) == 0:
            break

    return skel

def intensity(orig, gray, edges):

    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    hairPixels = img_dilation #edges

    #skel = skeletonize(img_dilation)
    hairPixelMask = np.ones(orig.shape[:2], dtype="uint8")
    hairPixelMask[:,:] = (hairPixels != 0) # 0 or 1 depending on wehter it is ==0
    hairOnBlack = gray.copy()
    hairOnBlack = hairPixelMask * gray

    hairOnWhite = gray.copy()
    #no hair*255 + hair * hair On Black(gray)
    #if no hair found, make it white. if hair found copy color from grayscale original
    hairOnWhite = (1-hairPixelMask)*255+hairPixelMask * gray

    show('orig', orig)
    show('hairOnWhite', hairOnWhite)
    #darker is more intense in this case

    # average color of background
    imediateBackground = cv2.dilate(edges, kernel, iterations=4)
    imediateBackground = imediateBackground-hairPixels
    imediateBackground = (1-imediateBackground)*0+imediateBackground * gray
    #count colors that are ligher than black
    backGroundPixels = np.count_nonzero(imediateBackground > 0)
    #print(backGroundPixels)
    #average color = sum of colors / pixels
    averageBackgroundColor = np.sum(imediateBackground)/backGroundPixels
    #print('backgroundcolor', averageBackgroundColor)

    #average color of everything
    avgColor = np.sum(gray)/np.size(gray)

    hairOnAverage = gray.copy()
    hairOnAverage = (1-hairPixelMask)*int(avgColor)+hairPixelMask * gray
    show('hair on Average background color', hairOnAverage)

    #average hair color
    sum = np.sum(hairOnBlack)
    averageHairColor = sum/np.count_nonzero(hairPixels == 255)

    #print('average hair color ', averageHairColor)

    #if average hair color is lighter than the background, flip hair on average
    #brither is more intense
    intensity = hairOnWhite.copy()
    intensity = 255-intensity
    #hmmmm scale is so that brightest hair is white and darkest hair stays the same

    uniquesortedValues = np.unique(intensity)
    newhigh = 255
    oldhigh = uniquesortedValues[uniquesortedValues.size-1]
    newlow = oldlow = uniquesortedValues[1]
    #print(np.unique(intensity))
    #if hair scale
    #hmm maybe we shoudnt scale it...
    #intensity = ( newlow + ((newhigh - newlow) / (oldhigh - oldlow)) * (intensity-oldlow))
    #reapply hairPixelMask
    #intensity = (1-hairPixelMask)*0 +(hairPixelMask * intensity) # no hair found = black
    intensitySum = np.sum(intensity)
    print('intensitySum:',intensitySum)
    show('Intensity', intensity)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(gray, 40, 200, kernel)

    show('original', img)
    show('canny', edges)
    # show('dilation', img_dilation)
    # show('erode', img_erosion)
    # show('inversion', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img, gray, edges

def detect(path):
    print(path)
    croped = scaleDots(path)
    orig,gray, edges = process(croped)
    found(edges)
    intensity(orig, gray, edges)
    return orig,edges

def processMatchResult(img_rgb,res,threshold,templatew, templateh):
    loc = np.where(res >= threshold)
    cnt = 0
    actualpoints = np.array([])
    prevpt= (0,0)
    differentPointthreshhold = 20
    rectangleImage = img_rgb.copy()
    for pt in zip(*loc[::-1]):
         if np.abs((pt[0]+pt[1]) - (prevpt[0]+prevpt[1])) >  differentPointthreshhold:
             actualpoints = np.append(actualpoints, pt)
             cnt = cnt + 1
         prevpt = pt
         cv2.rectangle(rectangleImage, pt, (pt[0] + templatew, pt[1] + templateh), (0, 0, 255), 2)
    show('foundDots', rectangleImage)
    #print('actual points', actualpoints)
    return cnt,actualpoints
def scaleDots(path):
    img_rgb = cv2.imread(path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('TemplateDot.jpg',0)
    templatew, templateh = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    show('img',img_rgb)
    show('match res', res)

    threshold = 0.8
    cnt, actualpoints = processMatchResult(img_rgb, res, threshold, templatew, templateh)
    if cnt == 0:
        print('no dots found, not cropping image')

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return img_rgb
    tries = 0
    retry = True
    retImg = img_rgb.copy()
    while (retry):
        tries = tries + 1
        if tries > 30:
            print('gave up. Best found:', cnt)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            retry = False
        if cnt ==4:
            # y1: the larger one of the lowerst 2 y
            ypoints = np.sort(actualpoints[1::2])  # every odd item
            xpoints = np.sort(actualpoints[::2])  # every even item
            y1 = int(ypoints[1] + (templateh / 2))  # the largest one of the loweset 2y
            y2 = int(ypoints[2] - (templateh / 2))
            x1 = int(xpoints[1] + (templatew / 2))
            x2 = int(xpoints[2] - (templatew / 2))
            # x1,y1 is top left vertex
            # x2,y2 is bottom right vertex
            #print(x1, y1, x2, y2)
            crop_img = img_rgb[y1:y2, x1:x2].copy()
            show('crop image', crop_img)
            retImg = crop_img
            retry = False
        if cnt >4:
            #found to many... make threshhold higher
            threshold = threshold+0.01
            cnt, actualpoints = processMatchResult(img_rgb, res, threshold, templatew, templateh)
        if cnt< 4:
            #found to few dots.. make threshhold lower to let more pass
            threshold = threshold-0.01
            cnt, actualpoints = processMatchResult(img_rgb, res, threshold, templatew, templateh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return retImg



if __name__ == "__main__":
    detect('Dots.jpg')
    detect('Dot_Mummel_1.jpg')
    detect('Dot_Mummel_1_3.jpg')
    detect('Dot_Mummel_1_4.jpg')
    detect('Dot_Mummel_4.jpg')
    #orig, edges = detect('Mummel_1.jpg')
    #orig, edges = detect('Mummel_6_long.jpg')
    #orig, edges = detect('Mummel_12_long_dense.jpg')
    #orig, edges = detect('Mummel_12_long_loose.jpg')
