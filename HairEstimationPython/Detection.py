#! /usr/bin/env python3

import cv2
import numpy as np
import scipy
from scipy.ndimage import label
from matplotlib import pyplot as plt
#region done
def show(title, image):
    resize = True
    height = image.shape[0]
    width = image.shape[1]
    # image = cv2.resize(image, (width, height))
    if resize:
        image = cv2.resize(image, (int(width * 0.2), int(height * 0.2)))
    cv2.imshow(title, image)
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
    #show('foundDots', rectangleImage)
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
        #trying out inverted image. for blond hair on black background
        revImg = 255 - img_gray
        res = cv2.matchTemplate(revImg, template, cv2.TM_CCOEFF_NORMED)
        show('revRes', res)
        cnt, actualpoints = processMatchResult(img_rgb, res, threshold, templatew, templateh)
        if cnt == 0:
            #if still nothing found assume there arent any points
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
            print(ypoints)
            print(xpoints)
            y1 = int(ypoints[1] + (templateh / 2))  # the largest one of the loweset 2y
            y2 = int(ypoints[2] - (templateh / 2))
            x1 = int(xpoints[1] + (templatew / 2))
            x2 = int(xpoints[2] - (templatew / 2))
            # x1,y1 is top left vertex
            # x2,y2 is bottom right vertex
            print(x1, y1, x2, y2)
            crop_img = img_rgb[y1:y2, x1:x2].copy()
            print(crop_img)
            #show('crop image', crop_img)
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

def hairPixelPercentage(img):
    sum = np.count_nonzero(img > 0)
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
def removeSmallRegions(intensity, img):
    #https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    removalThreshold = 600
    #show('intenisty', intensity)
    ret, markers = cv2.connectedComponents(intensity*255)
    #print('markers',np.unique(markers))
    markers = markers+1
    labels = cv2.watershed(img, markers)
    returnImage = intensity.copy()
    for label in np.unique(labels):
        mask = np.zeros(intensity.shape, dtype="uint8")
        mask[labels == label] = 1
        #print(np.sum(mask))
        image = intensity.copy()
        image = mask*255+(1-mask)*0
        #show('label '+str(label),image)
        if np.sum(mask) < removalThreshold:
            #remove the region
            returnImage = mask*0+(1-mask)*returnImage
    #show('small regions removed',returnImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return returnImage
def hairPixelIntensity(orig, gray, edges):

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

    #show('orig', orig)
    #show('hairOnWhite', hairOnWhite)
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
    #show('hair on Average background color', hairOnAverage)

    #average hair color
    sum = np.sum(hairOnBlack)
    averageHairColor = sum/np.count_nonzero(hairPixels == 255)

    #print('average hair color ', averageHairColor)

    #if average hair color is lighter than the background, flip hair on average
    #brither is more intense
    intensity = hairOnWhite.copy()
    intensity = 255-intensity
    intensity = removeSmallRegions(intensity,orig)
    intensitySum = np.sum(intensity)
    print('intensitySum:',intensitySum)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return intensity
def edgeProcess(orig):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(gray, 40, 200, kernel)
    intensity = hairPixelIntensity(orig, gray, edges)
    hairPixelPercentage(intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges,intensity
#endregion

def detect(path):
    print(path)
    croped = scaleDots(path)
    edges,intensity = edgeProcess(croped)
    backgroundRegions(intensity)
def backgroundRegions(intensity):
    region(intensity)

def region(intensity):
    show('input intensity', intensity)
    img = cv2.cvtColor(intensity,cv2.COLOR_GRAY2RGB)
    gray = intensity
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    #thresh = 255-thresh
    show('thresh',thresh)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    show('opening', opening)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    show('sure_bg',sure_bg)
    # Finding sure foreground area
    sure_fg = opening.copy()
    #dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    #show('dist_transform',dist_transform)
    #ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    show('sure_fg', sure_fg)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    show('unknown',unknown)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    #ok that looks like it worked. hair(black) is now 2. and all the white parts(background) are labelded 1+
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    backgroundSum = 0

    print(np.unique(markers))
    print(' backgorund sections',)
    allPixelSum = img.shape[0]*img.shape[1]
    print('all pixels', allPixelSum)
    sectionNum = np.unique(markers).size
    print('number of section:', sectionNum)
    innerSectionNum = sectionNum - 2
    print('number of section inclosed:', innerSectionNum)#-1 is space between and 1 is hair
    #finding size of outermost section
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == 2] = 1 # set pixels marked with 2 to 1. rest is 0.
    outerSectionSum = np.sum(mask)
    print('outerSectionSum', outerSectionSum)
    innerSectionSum = allPixelSum- outerSectionSum
    print('innerSectionSum', innerSectionSum)
    print('innserSectionAvgSize', innerSectionSum/innerSectionNum)


    for marker in np.unique(markers):
        #if(marker > 1): # skipping -1 and 1. -1 is space between, and 1 is hair(background).
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == marker] = 1
            backgroundSum = backgroundSum+ np.sum(mask)
            #print(np.sum(mask))
            image = gray.copy()
            image = mask * 255 + (1 - mask) * 0
            show('marker ' + str(marker), image)
    #img[markers == -1] = [255, 0, 0]
    #print('backgorund pixels:', backgroundSum / (np.size(markers)-2))
    #show('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
if __name__ == "__main__":
    #detect('Dot_Felina_ 4_1.jpg')
    detect('Dot_Felina_ 4_2.jpg')

    #detect('testRG.png')
    #detect('Dot_Mummel_1.jpg')
    #detect('Dot_Mummel_1_3.jpg')
    #detect('Dot_Mummel_1_4.jpg')
    #detect('Dot_Mummel_4.jpg')
    #detect('Dot_Mummel_medium.jpg')
