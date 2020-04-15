#! /usr/bin/env python
import datetime
import shutil

import cv2
import os
import numpy as np
import re
import matplotlib.dates as mdates
import argparse

import scipy
from scipy import stats
from scipy import interpolate
from matplotlib import pyplot as plt
import math
from scipy.optimize import curve_fit


debugstate = False
# region detect
def h_add(data, keys, key, value):
    data = np.append(data, value)
    keys = np.append(keys, key)
    return data, keys

def h_show(title, image):
    if debugstate:
        resize = True
        height = image.shape[0]
        width = image.shape[1]
        # image = cv2.resize(image, (width, height))
        if resize:
            image = cv2.resize(image, (int(width * 0.2), int(height * 0.2)))
        cv2.imshow(title, image)

# region cropDots
def h_processMatchResult(img_rgb, res, threshold, templatew, templateh):
    loc = np.where(res >= threshold)
    cnt = 0
    actualpoints = np.array([])
    differentPointthreshhold = (min(templateh, templateh) / 2) ** 2  # adaptive threshhold
    # print(differentPointthreshhold)
    rectangleImage = img_rgb.copy()
    for pt in zip(*loc[::-1]):
        if h_fuzzycontains(actualpoints, pt, differentPointthreshhold):
            actualpoints = np.append(actualpoints, pt)
            cnt = cnt + 1
        cv2.rectangle(rectangleImage, pt, (pt[0] + templatew, pt[1] + templateh), (0, 0, 255), 2)
    h_show('foundDots', rectangleImage) #show found dots with red rectangle around them
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print('actual points', actualpoints)
    return cnt, actualpoints, rectangleImage


def h_fuzzycontains(actualpoints, pt, differentPointthreshhold):
    xpoints = actualpoints[::2]
    ypoints = actualpoints[1::2]
    for actpoints in zip(xpoints, ypoints):
        if (pt[0] - actpoints[0]) ** 2 + (pt[1] - actpoints[1]) ** 2 < differentPointthreshhold:
            return False
    return True


def cropDots(img_rgb):
    print('cropping image to TemplateDot using PatternMatching')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    print(activeUserDirectory)
    template = cv2.imread(activeUserDirectory+'/TemplateDot.jpg', 0)
    templatew, templateh = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    #h_show('img',img_rgb)
    h_show('match res', res)

    threshold = 0.8
    cnt, actualpoints, rectImg = h_processMatchResult(img_rgb, res, threshold, templatew, templateh)
    if cnt == 0:
        print('fliping image cuz it is blond')
        # trying out inverted image. for blond hair on black background
        revImg = 255 - img_gray
        res = cv2.matchTemplate(revImg, template, cv2.TM_CCOEFF_NORMED)
        h_show('revRes', res)
        cnt, actualpoints, rectImg = h_processMatchResult(img_rgb, res, threshold, templatew, templateh)
        if cnt == 0:
            # if still nothing found assume there arent any points
            print('no dots found, not cropping image')
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            return img_rgb
    tries = 0
    retry = True
    retImg = img_rgb.copy()
    while retry:
        #print(cnt)
        #print(threshold)
        tries = tries + 1
        if tries > 30:
            print('gave up. Best found:', cnt)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            retry = False
        elif (cnt ==3) & (tries > 20):

            retImg = h_cropWith3Points(img_rgb, res, threshold, templatew, templateh, actualpoints, rectImg)
            retry = False
        elif cnt == 4:
            # y1: the larger one of the lowerst 2 y
            ypoints = np.sort(actualpoints[1::2])  # every odd item
            xpoints = np.sort(actualpoints[::2])  # every even item
            # print(xpoints)
            # print(ypoints)
            y1 = int(ypoints[1] + (templateh / 2))  # the largest one of the loweset 2y
            y2 = int(ypoints[2] - (templateh / 2))
            x1 = int(xpoints[1] + (templatew / 2))
            x2 = int(xpoints[2] - (templatew / 2))
            # x1,y1 is top left vertex
            # x2,y2 is bottom right vertex
            # print(x1, y1, x2, y2)

            crop_img = img_rgb[y1:y2, x1:x2].copy()
            retImg = crop_img
            h_show('rectImg', rectImg)
            h_show('crop image', crop_img)
            retry = False
        elif cnt > 4:
            # found to many... make threshhold higher
            threshold = threshold + 0.01
            cnt, actualpoints, rectImg = h_processMatchResult(img_rgb, res, threshold, templatew, templateh)

        elif cnt < 4:
            # found to few dots.. make threshhold lower to let more pass
            threshold = threshold - 0.01
            cnt, actualpoints, rectImg = h_processMatchResult(img_rgb, res, threshold, templatew, templateh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return retImg
def h_3pointsCropPoints(points, templatelength):
    if abs(points[0] - points[1]) > abs(points[1] - points[2]):
        # middle point is large [ 367. 3856. 3873.]
        choosenlowerPoint = int(points[0] + (templatelength / 2))
        choosenhigherPoint = int(points[1] - (templatelength / 2))
    else:
        # middle point is small [ 367. 375. 3873.]
        choosenlowerPoint = int(points[1] + (templatelength / 2))
        choosenhigherPoint = int(points[2] - (templatelength / 2))
    return choosenlowerPoint,choosenhigherPoint
def h_cropWith3Points(img_rgb, res, threshold, templatew, templateh, actualpoints, rectImg):
    print('crop with 3 points gets used')
    #if if keep jumping between 3 and 5 take the lower number and crop based on that.
    ypoints = np.sort(actualpoints[1::2])  # every odd item
    xpoints = np.sort(actualpoints[::2])  # every even item
    print(xpoints,ypoints)
    y1,y2 = h_3pointsCropPoints(ypoints,templateh)
    x1,x2 = h_3pointsCropPoints(xpoints,templatew)
    print(x1,x2,y1,y2)

    crop_img = img_rgb[y1:y2, x1:x2].copy()

    retImg = crop_img
    h_show('rectImg', rectImg)
    h_show('crop image', crop_img)
    return retImg
# endregion
def hairPixelPercentage(data, keys, img):
    sum = np.count_nonzero(img > 0)
    all = np.size(img)
    intensityShare = data[0] / sum
    data, keys = h_add(data, keys, 'intensityShare', intensityShare)
    data, keys = h_add(data, keys, 'hairpixels', sum)
    data, keys = h_add(data, keys, 'imagepixels', all)
    percentage = (sum / all) * 100
    data, keys = h_add(data, keys, 'percentage', percentage)

    return data, keys

def skeletonize(img):
    """ OpenCV function to return a skeletonized version of img, a Mat object"""

    #  hat tip to http://felix.abecassis.me/2011/09/opencv-morphological-skeleton/

    img = img.copy()  # don't clobber original
    skel = img.copy()

    skel[:, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img[:, :] = eroded[:, :]
        if cv2.countNonZero(img) == 0:
            break

    return skel


def removeSmallRegions(intensity, img):
    # https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html
    print('removing smaller Regions...')
    removalThreshold = img.shape[0]*img.shape[1]*0.00009
    ret, markers = cv2.connectedComponents(intensity * 255)
    # print('markers',np.unique(markers))
    markers = markers + 1
    labels = cv2.watershed(img, markers)
    returnImage = intensity.copy()
    for label in np.unique(labels):
        mask = np.zeros(intensity.shape, dtype="uint8")
        mask[labels == label] = 1
        # print(np.sum(mask))
        image = intensity.copy()
        image = mask * 255 + (1 - mask) * 0
        # show('removeSmallRegions label '+str(label),image)
        if np.sum(mask) < removalThreshold:
            # remove region if it is to small
            returnImage = mask * 0 + (1 - mask) * returnImage
    h_show('small regions removed', returnImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('done')
    return returnImage


def hairPixelIntensity(data, keys, orig, gray, edges):
    kernel = np.ones((3, 3), np.uint8)
    img_dilation = cv2.dilate(edges, kernel, iterations=1)
    hairPixels = img_dilation  # edges

    hairPixelMask = np.ones(orig.shape[:2], dtype="uint8")
    hairPixelMask[:, :] = (hairPixels != 0)  # 0 or 1 depending on wehter it is ==0

    # average color of everything

    # average color of background
    imediateBackground = cv2.dilate(edges, kernel, iterations=20)
    imediateBackground = imediateBackground - hairPixels
    imediateBackgroundPixelMask = np.ones(orig.shape[:2],dtype="uint8")
    imediateBackgroundPixelMask[:, :] = (imediateBackground != 0)  # 0 or 1 depending on wehter it is ==0
    imediateBackground = (1 - imediateBackgroundPixelMask) * 0 + imediateBackgroundPixelMask * gray
    h_show('imediateBackground',imediateBackground)
    # count colors that are ligher than black
    backGroundPixels = np.count_nonzero(imediateBackground > 0)
    # print(backGroundPixels)
    # average color = sum of colors / pixels
    averageBackgroundColor = np.sum(imediateBackground) / backGroundPixels
    print('backgroundcolor', averageBackgroundColor)

    sureHairPixelMask = np.ones(orig.shape[:2], dtype="uint8")
    sureHairPixelMask[:,:] = (cv2.erode(img_dilation, kernel, iterations=1) != 0)
    sum = np.sum(sureHairPixelMask * gray)
    h_show('sure hair',sureHairPixelMask * gray)
    # average hair color
    averageHairColor = sum / np.count_nonzero(sureHairPixelMask > 0)

    avgColor = np.sum(gray) / np.size(gray)
    #inverte hair if it's blond aka haircolor ligher than background.
    hairOnWhite = gray.copy()
    if averageHairColor>averageBackgroundColor:
        gray = 255-gray
    h_show('gray',gray)
    h_show('inverted gray', 255-gray)
    hairOnWhite = (1 - hairPixelMask) * 255 + hairPixelMask * gray

    # brither is more intense
    intensity = hairOnWhite.copy()
    intensity = 255 - intensity

    h_show('intenstiy', intensity)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    intensity = removeSmallRegions(intensity, orig)
    h_showMissed(intensity, gray, orig)

    intensitySum = np.sum(intensity)
    data, keys = h_add(data, keys, 'intensitySum:', intensitySum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return data, keys, intensity


def h_showMissed(intensity, gray, orig):
    # show how much hair is missed
    hairPixelMask = np.ones(gray.shape[:2], dtype="uint8")
    hairPixelMask[:, :] = (intensity != 0)  # 0 or 1 depending on wehter it is ==0
    ret = gray.copy()
    ret = hairPixelMask * 255 + (1 - hairPixelMask) * gray
    h_show('missed hair', ret)
    h_show('orig', orig)


def edgeProcess(data,gray, keys, blur):
    print('detecting Hair via edge detection')
    if blur:
        gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(gray, 40, 200, kernel)
    h_show('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return data, keys, edges


def backgroundRegions(data, keys, intensity):  # only uses intensity image. background black, hair white
    print('processing background regions...')
    h_show('input intensity', intensity)
    img = cv2.cvtColor(intensity, cv2.COLOR_GRAY2RGB)
    gray = intensity
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    sure_fg = opening.copy() # Finding sure foreground area
    sure_fg = np.uint8(sure_fg) # Finding unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg) # Marker labelling
    # ok that looks like it worked. hair(black) is now 2. and all the white parts(background) are labelded 1+
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)

    allPixelSum = img.shape[0] * img.shape[1]
    data, keys = h_add(data, keys, 'all pixels', allPixelSum)
    sectionNum = np.unique(markers).size
    data, keys = h_add(data, keys, 'number of section', sectionNum)
    innerSectionNum = sectionNum - 2
    data, keys = h_add(data, keys, 'number of section inclosed', innerSectionNum)  # -1 is space between and 1 is hair
    # finding size of outermost section
    maskoutr = np.zeros(gray.shape, dtype="uint8")
    maskoutr[markers == 2] = 1  # set pixels marked with 2 to 1. rest is 0.

    image = intensity.copy()
    image = maskoutr * 255 + (1 - maskoutr) * 0
    h_show('outer section', image)
    outerSectionSum = np.sum(maskoutr)
    data, keys = h_add(data, keys, 'outerSectionSum', outerSectionSum)
    data, keys = h_add(data, keys, 'outerSectionPercentage', outerSectionSum / allPixelSum)
    innerSectionSum = allPixelSum - outerSectionSum
    data, keys = h_add(data, keys, 'innerSectionSum', innerSectionSum)
    innerSectionAvg = innerSectionSum / innerSectionNum
    data, keys = h_add(data, keys, 'innserSectionAvgSize', innerSectionAvg)
    data, keys = h_add(data, keys, 'innerSectionAvgSize Percentage', innerSectionAvg / allPixelSum)
    print('innerSectionNum',innerSectionNum)
    sizes = np.array([])
    for marker in np.unique(markers):
        if marker > 2:
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == marker] = 1
            sizes = np.append(sizes, np.sum(mask))
    innerSectionSizeVariance = np.var(sizes)
    data, keys = h_add(data, keys, 'innerSectionSizeVariance', innerSectionSizeVariance)
    data, keys = h_add(data, keys, 'std', np.std(sizes))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print('done')
    return data, keys,maskoutr


def h_showstats(data, keys):
    for i in range(np.size(data)):
        print(keys[i], data[i])
    print()

def denseAndLoosePerc(data,keys, intensity,maskoutr,densemaskoutr):
    hairPixelMask = np.ones(intensity.shape[:2], dtype="uint8")
    hairPixelMask[:, :] = (intensity != 0)  #1 for hair
    densemaskoutr = 1-densemaskoutr # 1 for inside
    maskoutr = 1-maskoutr # 1 for inside
    looseMask = (maskoutr-densemaskoutr)
    densehair = hairPixelMask * (densemaskoutr)
    loosehair = hairPixelMask * (looseMask)
    denseHairSum= np.count_nonzero(densehair > 0)
    looseHairSum = np.count_nonzero(loosehair> 0)
    data, keys = h_add(data, keys, 'denseHairSum', denseHairSum)
    data, keys = h_add(data, keys, 'looseHairSum', looseHairSum)
    # intensity per section
    looseintensity = looseMask * intensity
    denseintensity = densemaskoutr * intensity
    data, keys = h_add(data, keys, 'IntensitySum in Loose Section', np.sum(looseintensity))
    data,keys = h_add(data,keys, 'IntensitySum in Dense Section',np.sum(denseintensity))
    return data,keys

def detect(path):
    print('image path',path)
    # loading image
    blur = False
    orig = cv2.imread(path)
    data = np.array([])
    keys = np.array([])
    if orig is None:
        print('no image with that path found')
        return data,keys

    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)

    croped = cropDots(orig)
    data, keys, edges         = edgeProcess(data, keys, croped, blur)
    data, keys, intensity     = hairPixelIntensity(data, keys, orig, gray, edges)
    data, keys                = hairPixelPercentage(data, keys, intensity)
    data, keys, maskoutr      = backgroundRegions(data, keys, intensity)
    data, keys, densemaskoutr = backgroundRegions(data,keys,skeletonize(intensity))
    data, keys                = denseAndLoosePerc(data,keys, intensity,maskoutr,densemaskoutr)
    # showstats( data,keys)
    return data, keys


def testEdgeDetection(path):
    orig = cv2.imread(path)

    blur = cv2.blur(orig, (5, 5))
    h_show('blur', blur)
    blur = cv2.GaussianBlur(orig, (5, 5), 0)
    h_show('gaussianBlur', blur)
    # blur = cv2.medianBlur(orig, 3)
    h_show('meidanblur', blur)
    # blur = orig
    h_show('orig', orig)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # show('thresh', thresh)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(orig, 30, 200, kernel)
    h_show('orig edges', edges)
    edges2 = cv2.Canny(blur, 30, 200, kernel)
    h_show('blur canny', edges2)
    h_show('diff', edges - edges2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# endregion
# region calibration
useOldFunc = False
def calibration():
    #find every image in the folder and use them to calibrate with
    #retrieve names of the images
    paths = os.listdir(calibrationImagesDirectorypath)
    #add the image names to the relative path
    paths = [calibrationImagesDirectorypath+'/' + path for path in paths]
    print(paths)
    calibrateProcessImages(paths)

def calibrateProcessImages(calibrationPaths):
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.5f}'.format})
    alldata = np.array([])
    hairAmount = np.array([])
    i=0
    if len(calibrationPaths) == 0:
        print('No Images for calibration found. '+calibrationImagesDirectorypath+' is empty.')
        return
    for path in calibrationPaths:
        numbers = re.findall(r'\d+', path)
        if(np.size(numbers)==0):
            continue
        hairAmount = np.append(hairAmount, numbers[0])
        data, keys = detect(path)
        print('hair amount:', hairAmount[i], 'hair percent', data[4], 'outersectionSize:', data[9],'innersectionNum', data[7])
        i= i+1
        alldata = np.append(alldata, data)
    # f = open("calibrationImageStats.", "w+")
    print('save', alldata)
    np.save(calibrationResultDatapath, alldata)
    np.save(keyDatapath, keys)
    np.save(hairAmountDatapath, hairAmount)
    if useOldFunc==False:
        ProcessBestFunctions(alldata, keys, hairAmount)
    # f.write(alldata)
    # f.close()
    # save alldata and stats in a file
def addCalibrationImage(path,amount):
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.5f}'.format})
    alldata = np.array([])
    hairAmount = np.array([])
    try:
        alldata = np.load(calibrationResultDatapath + '.npy')
        hairAmount = np.load(hairAmountDatapath + '.npy')
    except FileNotFoundError:
        pass
    data, keys = detect(path)
    if(np.size(data) == 0):
        return
    alldata = np.append(alldata, data)
    hairAmount = np.append(hairAmount, amount)
    np.save(calibrationResultDatapath, alldata)
    np.save(hairAmountDatapath, hairAmount)
    np.save(keyDatapath,keys)
#endregion
#region guess
def manualGuess(datestr,numstr):
    save('',int(float(numstr)),datestr)
def guessTest():
    debugg(True)
    debugg(True)
    alldata = np.load(calibrationResultDatapath + '.npy')
    keys = np.load(keyDatapath + '.npy')
    hairAmount = np.load(hairAmountDatapath + '.npy')
    imgdata = alldata[:np.size(keys)]
    guessProcess(alldata,keys,hairAmount,imgdata)
def h_ymd(path):
    numbers = re.findall(r'\d+', path)
    numbers = numbers[0]
    year = numbers[0:4:]
    month = numbers[4:6:]
    day = numbers[6:8:]
    ymd = year + '-' + month + '-' + day
    return ymd
def readFilesAndGuess(path):
    # read alldata and stats from file.
    try:
        alldata = np.load(calibrationResultDatapath + '.npy')
        keys = np.load(keyDatapath + '.npy')
        hairAmount = np.load(hairAmountDatapath + '.npy')
    except:
        print('cant estimate hair without calibrating first. run <calibrate>')
        return
    if (np.size(alldata) == 0) | (np.size(keys) == 0) | (np.size(hairAmount) == 0):
        print('cant estimate hair without calibrating first. run <calibrate>')
        return
    print('guessing the amount of hair in the picture')
    imgdata, _ = detect(path)
    return guessProcess(alldata, keys, hairAmount, imgdata)
def guess(path):
    estimationResult = readFilesAndGuess(path)
    ymd = h_ymd(path)
    save(path, estimationResult,ymd)
def guessWithDaysAndTag(path,days,tag):
    estimationResult = readFilesAndGuess(path)
    saveDays(path,days,estimationResult)
    ymd = h_ymd(path)
    saveTag(tag,ymd)
def guessWithDaysOrTag(path,arg):
    if arg.isdigit():
        guessWithDays(path,arg)
    else:
        guessWithTag(path,arg)
def guessWithDays(path,days):
    estimationResult = readFilesAndGuess(path)
    saveDays(path,days,estimationResult)
def guessWithTag(path,tag):
    guess(path)
    ymd = h_ymd(path)
    saveTag(tag,ymd)
def automaticGuessFunctions():
    try:
        alldata = np.load(calibrationResultDatapath + '.npy')
        keys = np.load(keyDatapath + '.npy')
        hairAmount = np.load(hairAmountDatapath + '.npy')
    except:
        print('cant estimate hair without calibrating first. run <calibrate>')
        return
    if(np.size(alldata)==0) | (np.size(keys)== 0) | (np.size(hairAmount)==0) :
        print('cant estimate hair without calibrating first. run <calibrate>')
        return
    print('automatically finding guess Functions')
    ProcessBestFunctions(alldata, keys, hairAmount)
def guessProcess(alldata, keys, hairAmount, imgdata):
    #['0 intensitySum:' '1 intensityShare' '2 hairpixels' '3 imagepixels' '4 percentage'
    # '5 all pixels' '6 number of section' '7 number of section inclosed'
    # '8 outerSectionSum' '9 outerSectionPercentage' '10 innerSectionSum'
    # '11 innserSectionAvgSize' '12 innerSectionAvgSize Percentage'
    # '13 innerSectionSizeVariance' '14 std' '15 all pixels' '16 number of section'
    # '17 number of section inclosed' '18 outerSectionSum' '19 outerSectionPercentage'
    # '20 innerSectionSum' '21 innserSectionAvgSize' '22 innerSectionAvgSize Percentage'
    # '23 innerSectionSizeVariance' '24 std', 25 densehairSum, 26 loosehair sum]
    functions = loadFunctions()


    hairAmount = np.array(list(map(int, hairAmount)))
    allPixels = alldata[3::np.size(keys)]
    hairPerc = alldata[4::np.size(keys)]
    hairpixels = alldata[2::np.size(keys)]
    hairSectionSize = alldata[10::np.size(keys)]

    denseHairSum = alldata[25::np.size(keys)]
    denseSectionperc = alldata[19::np.size(keys)]
    denseInnerSectionSize = alldata[20::np.size(keys)]
    denseIntensity = alldata[28::np.size(keys)]

    outerSectionPerc = alldata[9::np.size(keys)]
    secn = alldata[7::np.size(keys)]  # number of inner sections
    looseHairSum = alldata[26::np.size(keys)]
    backgroundSectionNum = alldata[7::np.size(keys)]
    denseSectionNum = alldata[17::np.size(keys)]
    intensitySum= alldata[0::np.size(keys)]
    intensityShare = alldata[1::np.size(keys)]
    denseSectionAVGSize = alldata[21::np.size(keys)]
    looseSectionAVGSize =alldata[11::np.size(keys)]

    i_intensitySum= imgdata[0]
    i_intensityShare = imgdata[1]
    i_allPixels = imgdata[3]
    i_origperc = imgdata[4]
    i_hairpixels = imgdata[2]
    i_hairSectionSize = imgdata[10]

    i_denseHairSum = imgdata[25]
    i_denseSectionperc = imgdata[19]
    i_denseinnerSectionSize = imgdata[20]
    i_denseIntensity = imgdata[28]

    i_outerSectionPerc = imgdata[9]
    i_secn = imgdata[7]  # number of inner sections
    i_looseHairSum = imgdata[26]
    i_backgroundSectionNum = imgdata[7]
    i_denseSectionNum = imgdata[17]
    i_denseSectionAVGSize = imgdata[21]
    i_looseSectionAVGSize =imgdata[11]

    denseDensity = (((denseHairSum / denseInnerSectionSize) * (1 - denseSectionperc)))
    looseDensity = ((hairpixels - denseHairSum) / (hairSectionSize - denseInnerSectionSize)) * (
            (1 - outerSectionPerc) - (1 - denseSectionperc))
    i_denseDensity = (((i_denseHairSum / i_denseinnerSectionSize) * (1 - i_denseSectionperc)))
    i_looseDensity = ((i_hairpixels - i_denseHairSum) / (i_hairSectionSize - i_denseinnerSectionSize)) * (
            (1 - i_outerSectionPerc) - (1 - i_denseSectionperc))

    estimation = np.array([])
    estimation = np.append(estimation,
                           model(hairPerc,hairAmount,i_origperc,'hairpercent',False,functions[0]))
    estimation = np.append(estimation,
                           model((hairpixels / hairSectionSize) * (1 - outerSectionPerc), hairAmount,
                                 (i_hairpixels / i_hairSectionSize) * (1 - i_outerSectionPerc),
                                 'density * hairsection size',False,functions[1]))
    estimation = np.append(estimation,
                           model((hairpixels/hairSectionSize)*(1-outerSectionPerc)*hairPerc,hairAmount,
                                 (i_hairpixels/i_hairSectionSize)*(1-i_outerSectionPerc)*i_origperc,
                                 'density per sectionsize per hairperc',False,functions[2]))
    estimation = np.append(estimation,
                           model(denseDensity,hairAmount,
                                 i_denseDensity,'density of dense section in realtion to section size',False,functions[3]))
    estimation = np.append(estimation,
                           model(denseDensity / looseDensity, hairAmount,
                                 i_denseDensity/i_looseDensity,'dense Density to looseDensity ratio',False,functions[4]))
    estimation = np.append(estimation,
                            model(backgroundSectionNum*(1-outerSectionPerc),hairAmount,
                                  i_backgroundSectionNum*(
                                          1-i_outerSectionPerc),
                                  'density by background sections peeking through',False,functions[5]))
    estimation = np.append(estimation,
                           model(denseSectionNum*(1-outerSectionPerc),hairAmount,
                                 i_denseSectionNum*(1-i_outerSectionPerc),
                                 'dense Density by background sections peeking through',False,functions[6]))
    estimation = np.append(estimation,
                           model(intensitySum*hairpixels*(1-outerSectionPerc),hairAmount,
                                 i_intensitySum*i_hairpixels*(1-i_outerSectionPerc),
                                 'intensity in relation to hairPixels and hairsection size',True,functions[7]))
    estimation = np.append(estimation,
                           model(denseSectionAVGSize/(1-denseSectionperc),hairAmount,
                                i_denseSectionAVGSize/(1-i_denseSectionperc),
                           'average backgorund section sizes in dense section',True,functions[8]))
    estimation = np.append(estimation,
                           model((denseSectionAVGSize/looseSectionAVGSize)/(1-outerSectionPerc),hairAmount,
                                (i_denseSectionAVGSize/i_looseSectionAVGSize)/(1-i_outerSectionPerc),
                                'dense average background section sizes compared to looseaveragebackgorund sections size',True,functions[9]))
    estimation = np.append(estimation,
                           model((denseIntensity/intensitySum)*(1-denseSectionperc),hairAmount,
                                (i_denseIntensity/i_intensitySum)*(1-i_denseSectionperc),'denseIntensity/intensitySum * denseSectionSize',False,functions[10]))
    estimation = np.append(estimation,
                           model((denseIntensity/denseHairSum)*(1-denseSectionperc),hairAmount,
                                 (i_denseIntensity/i_denseHairSum)*(1-i_denseSectionperc),'denseIntensity/denseHairSum* denseSectionSize',False,functions[11]))
    cleanedEstimation = np.array([])
    for e in estimation:
        if e > 0:
            cleanedEstimation = np.append(cleanedEstimation, e)
    print(cleanedEstimation)
    cleanedEstimation = PGPremoveOutlier(cleanedEstimation,0.75,0.1)
    print('outliers Removed', cleanedEstimation)
    mean = np.mean(cleanedEstimation)
    res = round(mean, 0)
    print('mean', mean, 'res', res, 'median', np.median(cleanedEstimation))
    return res
def func_exp(x, a, b, c):
    return a * np.exp(b * x) + c
def func_divx(x, a, b):
    return a * (1 / x) + b
def func_log(x, a, b, c):
    return a * np.log(abs(b) * x) + c
def func_lin(x,a,b):
    return a * x + b
def func_empty(x,a):
    return -1*x*0.0000000000000000001*a
def findBestFunction(x, y):
    func = func_exp
    possiblefunc = np.array([func_divx,func_exp,func_lin,func_log])
    minChisq = 5000
    bestFunc = func_lin
    for func in possiblefunc:
        try:
            popt, pcov = curve_fit(func, x, y)
        except:
            continue
        #plt.plot(x, y, 'ko', label="Original Noised Data")
        #plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
        _,chisq = scipy.stats.chisquare(y, func(x, *popt), ddof = 2,axis = 0)
        print(chisq)
        if np.isnan(chisq):
            continue
        if chisq < minChisq:
            print('found smaller')
            minChisq = chisq
            bestFunc = func
    print(bestFunc)
    return bestFunc

def ProcessBestFunctions(alldata, keys, hairAmount):
    hairAmount = np.array(list(map(int, hairAmount)))
    allPixels = alldata[3::np.size(keys)]
    hairPerc = alldata[4::np.size(keys)]
    hairpixels = alldata[2::np.size(keys)]
    hairSectionSize = alldata[10::np.size(keys)]

    denseHairSum = alldata[25::np.size(keys)]
    denseSectionperc = alldata[19::np.size(keys)]
    denseInnerSectionSize = alldata[20::np.size(keys)]
    denseIntensity = alldata[28::np.size(keys)]
    outerSectionPerc = alldata[9::np.size(keys)]
    secn = alldata[7::np.size(keys)]  # number of inner sections
    looseHairSum = alldata[26::np.size(keys)]
    backgroundSectionNum = alldata[7::np.size(keys)]
    denseSectionNum = alldata[17::np.size(keys)]
    intensitySum = alldata[0::np.size(keys)]
    intensityShare = alldata[1::np.size(keys)]
    denseSectionAVGSize = alldata[21::np.size(keys)]
    looseSectionAVGSize = alldata[11::np.size(keys)]

    denseDensity = (((denseHairSum / denseInnerSectionSize) * (1 - denseSectionperc)))
    looseDensity = ((hairpixels - denseHairSum) / (hairSectionSize - denseInnerSectionSize)) * (
            (1 - outerSectionPerc) - (1 - denseSectionperc))

    functions = np.array([])
    functions = np.append(functions,findBestFunction(hairPerc,hairAmount))
    functions = np.append(functions,findBestFunction((hairpixels / hairSectionSize) * (1 - outerSectionPerc), hairAmount))
    functions = np.append(functions,findBestFunction((hairpixels / hairSectionSize) * (1 - outerSectionPerc) * hairPerc, hairAmount))
    functions = np.append(functions,findBestFunction(denseDensity, hairAmount))
    functions = np.append(functions,findBestFunction(denseDensity / looseDensity, hairAmount))
    functions = np.append(functions,findBestFunction(backgroundSectionNum * (1 - outerSectionPerc), hairAmount))
    functions = np.append(functions,findBestFunction(denseSectionNum * (1 - outerSectionPerc), hairAmount))
    functions = np.append(functions,findBestFunction(intensitySum * hairpixels * (1 - outerSectionPerc), hairAmount))
    functions = np.append(functions,findBestFunction(denseSectionAVGSize / (1 - denseSectionperc), hairAmount))
    functions = np.append(functions,findBestFunction((denseSectionAVGSize / looseSectionAVGSize) / (1 - outerSectionPerc), hairAmount))
    functions = np.append(functions,findBestFunction((denseIntensity/intensitySum)*(1-denseSectionperc),hairAmount))
    functions = np.append(functions,findBestFunction((denseIntensity/denseHairSum)*(1-denseSectionperc),hairAmount))
    saveFunctions(functions)
def saveFunctions(functions):
    str = ''
    for f in functions:
        if f == func_divx:
            str = str+'_divx'
        if f == func_exp:
            str = str+'_exp'
        if f == func_lin:
            str = str+'_lin'
        if f == func_log:
            str = str+'_log'
    setFunctions(str)
def loadFunctions():
    functions = np.array([])
    if os.path.exists(guessfunctionDatapath):
        fp = open(guessfunctionDatapath, 'r+')
        data = fp.read()
        fp.close()
        data = data.split("_")
        for e in data:
            if e == 'exp':
                functions = np.append(functions, func_exp)
            if e == 'log':
                functions = np.append(functions, func_log)
            if e == 'divx':
                functions = np.append(functions, func_divx)
            if e == 'lin':
                functions = np.append(functions, func_lin)
    #fill the rest up with func_empty
    i = np.size(functions)
    for i in range(15):
        functions = np.append(functions,func_empty)
    return functions
def loadTags():
    if os.path.exists(tagsDatapath):
        fp = open(tagsDatapath,'r+')
        content = fp.read().splitlines()
        return content
    return np.array([])
def saveTag(tag, date):
    #loadTags
    taglists = loadTags()
    #find our tag
    #updatedTaglists = np.array([])
    found = False
    for i,tagstring in enumerate(taglists):
        content = tagstring.split(' ')
        if content[0] == tag:
            #add date to the taglist
            found = True
            taglists[i] = taglists[i]+' '+date
            # does taglits get changed as well?
    if found == False:
        #Tag doesnt exist jet.
        print('creating new tag', tag)
        taglists = np.append(taglists,  tag +' '+date)
    saveAllTags(taglists)
    #save updated tags
def saveAllTags(taglists):
    fp = open(tagsDatapath,'w+')
    for singletaglist in taglists:
        fp.write(singletaglist)
        fp.write('\n')
    fp.close()
def setFunctions(str):
    fp = open(guessfunctionDatapath, 'w+')
    print('saving new guess functions to', guessfunctionDatapath)
    fp.write(str)
    fp.close()
def PGPremoveOutlier(PData, Pp1, Pp2):
    # phase 1
    mu1 = PData.mean()
    sigma1 = math.sqrt(PData.var())  # std = standardabweichung (wie stats.binom.std()).
    k = 1 / math.sqrt(Pp1)
    odv1U = mu1 + k * sigma1
    odv1L = mu1 - k * sigma1
    # print('mu1 ',mu1, 'sigma: ',sigma1, 'k ', k, odv1U, odv1L)
    NewData = np.array([i for i in PData if i <= odv1U])
    NewData = np.array([i for i in NewData if i >= odv1L])

    # phase2
    mu2 = NewData.mean()
    sigma2 = math.sqrt(PData.var())
    k = 1 / math.sqrt(Pp2)
    odv2U = mu2 + k * sigma2
    odv2L = mu2 - k * sigma2
    NewData = np.array([i for i in NewData if i <= odv2U])
    NewData = np.array([i for i in NewData if i >= odv2L])

    return NewData

def lin(a,b,x):
    return a*x+b
def linregress(x,y):
    a = np.cov(x, y)[0, 1] / np.cov(x, y)[0, 0]  # slope
    b = y.mean() - a * x.mean()  # intersept
    return a,b




def model(x,y,datapoint,description,ignorelin,func):
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})
    inds = x.argsort()
    x = x[inds]
    y = y[inds]
    a, b = linregress(x, y)
    spl = interpolate.InterpolatedUnivariateSpline(x, y, k=1)
    popt, pcov = curve_fit(func, x, y)

    linguess = lin(a, b, datapoint)
    splguess = spl(datapoint)
    funcguess = func(datapoint, *popt)
    if debugstate:
        fitx = np.linspace(x.min(), x.max(), 100)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fitx, spl(fitx),label ='splinefunktion')
        ax.plot(fitx, lin(a, b, fitx),label ='linearRegression')
        plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
        ax.scatter(x, y, label='calibrationData')
        ax.set_title('Guess by ' + description + ' funktion')
        ax.set_xlabel(description)
        ax.set_ylabel('amount of hair')
        if ignorelin==False:
            ax.scatter(datapoint,linguess,label='linear Guess')
        ax.scatter(datapoint,splguess, label ='spline Guess')
        ax.scatter(datapoint,funcguess, label ='func Guess')
        ax.legend(loc='best')
        plt.show()
    if ignorelin:
        print('Guess by', description, '[ spl', "%.2f" % splguess, ';func', "%.2f" % funcguess, ']')
        return splguess,funcguess
    print('Guess by', description, '[ spl',"%.2f" % splguess,';func',"%.2f" % funcguess, ';linguess', "%.2f" % linguess,']')
    return linguess, splguess, funcguess
def findBestFunction(x, y):
    func = func_exp
    possiblefunc = np.array([func_divx,func_exp,func_lin,func_log])
    minChisq = 5000
    bestFunc = func_lin
    for func in possiblefunc:
        try:
            popt, pcov = curve_fit(func, x, y)
        except:
            continue
        #plt.plot(x, y, 'ko', label="Original Noised Data")
        #plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
        _,chisq = scipy.stats.chisquare(y, func(x, *popt), ddof = 2,axis = 0)
        print(chisq)
        if np.isnan(chisq):
            continue
        if chisq < minChisq:
            print('found smaller')
            minChisq = chisq
            bestFunc = func
    return bestFunc

def guessFolder(folder):
    #estimate every image in the folder
    if folder == '':
        folder = 'NewEstimationInput'
    folder = activeUserDirectory +'/'+folder
    files = os.listdir(folder)
    paths = [folder+'/' + file for file in files]
    for path,file in zip(paths,files):
        guess(path)
        shutil.move(path, activeUserDirectory+"/estimationInput/"+file)

duplicateHandelingMode = 'r'
def saveDays(path,days,estRes):
    #save for the day in path and the number of days before that date.
    ymd = h_ymd(path)
    days =  int(days)
    res = estRes / (days+0.0)
    save(path,estRes/days,ymd)
    days = days-1
    DAY = datetime.timedelta(1)
    ymd = datetime.datetime.strptime(ymd, "%Y-%m-%d").date()
    for i in range(days):
        ymd = ymd - DAY
        dayStr = ymd.strftime("%Y-%m-%d")
        save(path,res,dayStr)

def save(path, estRes,ymd):

    #save image data to a file with all the previous estimations by date
    estimationResult = np.array([ymd, estRes])
    oldData = np.array([])
    if os.path.exists(estimationResultDatapath+'.npy'):
        oldData = np.load(estimationResultDatapath+'.npy')
        oldDates = oldData[::2] #date est
        if ymd in oldDates:
            print('found duplicate date. Using handeling mode:',duplicateHandelingMode)
            #three handeling modi. replace, mean,ignoreNew
            if duplicateHandelingMode =='i':
                print('ignoring new estimation')
                return
            index = (np.where(oldData==ymd))[0]
            index = index + 1  # index of est result.
            if duplicateHandelingMode =='r':
                #replace old date.
                oldData[index] = estRes
                np.save(estimationResultDatapath, oldData)
                print('replacing old datapoint with', ymd, estRes, 'ImagePath:', path)
                return
            if duplicateHandelingMode =='m':
                oldest = oldData[index]
                oldest = float(oldest)
                oldest = int(oldest)
                mean = np.mean(np.array([oldest,estRes]))
                oldData[index] = mean
                np.save(estimationResultDatapath, oldData)
                print('using mean between old and new data', ymd, mean, 'ImagePath:', path)
                return
            if duplicateHandelingMode =='k':
                print('keeping duplicates')
            if duplicateHandelingMode =='a':
                oldest = oldData[index]
                oldest = float(oldest)
                oldest = int(oldest)
                n = oldest + estRes
                oldData[index] = n
                np.save(estimationResultDatapath,oldData)
                print('adding new datapoint to old datapoint',ymd, n, 'ImagePath:',path)
                return
    newData = np.append(oldData, estimationResult)
    print('saving data point', ymd, estRes, 'ImagePath:', path)
    print()
    np.save(estimationResultDatapath ,newData)
#endregion
# region showanddebugg
def printResult():
    try:
        data = np.load(estimationResultDatapath + '.npy')
    except:
        print('no data found')
        return
    print(data)
def onlyGuessTest(str):
    clearSave()
    global duplicateHandelingMode
    duplicateHandelingMode = 'k'
    guessFolder('estimationInput')
    checkCalibration()
    calculateError(str)
def fullTest(str):
    clearSave()
    calibration()
    global duplicateHandelingMode
    duplicateHandelingMode = 'k'
    guessFolder('estimationInput')
    checkCalibration()
    calculateError(str)
def checkCalibration():
    try:
        hairAmount = np.load(hairAmountDatapath + '.npy')
    except:
        print('no calibration data found')
    print(np.sort(hairAmount))
    if debugstate:
        guessTest()
def calculateError(numberstring):
    try:
        data = np.load(estimationResultDatapath + '.npy')
    except:
        print('no data found')
        return
    numbers = re.findall(r'\d+', numberstring)
    numbers = np.array(list(map(int, numbers)))
    if (np.size(numbers) == 0):
        print('no actual hairAmount given in second positional argument')
        return

    estimatedHair = data[1::2]
    estimatedHair = np.array(list(map(float,estimatedHair)))
    estimatedHair = np.array(list(map(int, estimatedHair)))
    print(estimatedHair)
    if np.size(estimatedHair)>np.size(numbers):
        estimatedHair = estimatedHair[np.size(estimatedHair)-np.size(numbers)::]
    print('(estimated, actual)')
    print(list(zip(estimatedHair,numbers)))
    estimatedHair = np.array(estimatedHair)
    error= estimatedHair-numbers
    print('error per estimation', error)
    error = abs(error)
    meanerror = np.mean(error)
    print('mean error', meanerror)

def removeLastSaveImg():
    try:
        estimationData = np.load(estimationResultDatapath+'.npy')
    except:
        print('no data found')
        return
    #data has one pair per img. hairAmounts has one entry per img
    print(estimationData)
    estimationData = estimationData[:-2]
    print(estimationData)
    np.save(estimationResultDatapath, estimationData)
def debugg(state):
    global debugstate
    debugstate = state

def plotEstimationResult():
    try:
        data = np.load(estimationResultDatapath+'.npy')
    except:
        print('no data to show. use command <guess>')
        return
    if np.size(data)== 0:
        print('no data to show')
        return
    dates = data[::2]
    dates = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in dates]
    hairAmounts = data[1::2]
    hairAmounts = np.array(list(map(float, hairAmounts)))
    ax = plt.gca()
    formatter = mdates.DateFormatter("%Y-%m-%d")
    ax.xaxis.set_major_formatter(formatter)
    locator = mdates.DayLocator(interval = 4)
    ax.xaxis.set_major_locator(locator)
    tags = loadTags()
    for tag in tags:
        tag = tag.split(' ')
        tagdates = tag[1::]
        tagdates = [datetime.datetime.strptime(d, "%Y-%m-%d").date() for d in tagdates]
        plt.bar(tagdates,hairAmounts.max(),0.2,label=tag[0])
    plt.scatter(dates, hairAmounts, color ='blue', label ='estimated Hair')
    plt.plot(dates,hairAmounts, color ='blue')
    ax.legend(loc='best')
    plt.gcf().autofmt_xdate()
    ax.grid()
    plt.show()
def clearSave():
    np.save(estimationResultDatapath,np.array([]))
    print('erased saved estimation results')
def printFile(path):
    data = np.load(path)
    print(data)
# endregion
# region User filesystem handeling
activeUserDirectory = ''
calibrationImagesDirectorypath = ''
dataDirectorypath = ''
calibrationResultDatapath = ''
keyDatapath =''
estimationResultDatapath=''
activeUserDatapath = 'activeUser.txt'
hairAmountDatapath = ''
usersDirectory ='Users'
guessfunctionDatapath = ''
tagsDatapath = ''
def buildPaths(user):
    global calibrationImagesDirectorypath
    global dataDirectorypath
    global calibrationResultDatapath
    global keyDatapath
    global estimationResultDatapath
    global hairAmountDatapath
    global activeUserDirectory
    global guessfunctionDatapath
    global tagsDatapath
    activeUserDirectory = usersDirectory +'/'+user
    calibrationImagesDirectorypath = usersDirectory + '/' + user + '/calibrationImages'
    dataDirectorypath = usersDirectory+'/'+user + '/data'
    calibrationResultDatapath = dataDirectorypath + '/calibrationData.out'
    keyDatapath  = dataDirectorypath+'/calibkeyData.out'
    estimationResultDatapath = dataDirectorypath+'/result.out'
    hairAmountDatapath = dataDirectorypath+'/calibhairAmount.out'
    guessfunctionDatapath = dataDirectorypath+'/guessfunctions.txt'
    tagsDatapath = dataDirectorypath+'/tags.txt'
def loadUser():
    if os.path.exists(activeUserDatapath):
        fp = open(activeUserDatapath, 'r+')
        currentUser = fp.read()
        fp.close()
        if (len(currentUser) == 0):
            print('no aktive user found')
            createUser('default')
        else:#user found
            buildPaths(currentUser)
            print('active user: '+currentUser)
            print()
    else:
        print('no users exits yet')
        createUser('default')
def switchUser(user):
    if False == os.path.exists(usersDirectory):
        createUser(user)
        return
    if user in h_getAllUsers():
        #user exits
        buildPaths(user)
        fp = open(activeUserDatapath, 'w+')
        fp.write(user)
        fp.close()
        print('switching to user '+user)
    else:
        print('user '+user+' does not exist. Command <create '+user+'> to create.')
        print('maybe you are looking for: ', h_getAllUsers())
def createUser(user):
    fp = open(activeUserDatapath, 'w+')
    fp.write(user)
    fp.close()
    buildPaths(user)
    if False == os.path.exists(usersDirectory):
        print('ceating Users directory')
        os.mkdir(usersDirectory)
    try:
        print('adding user '+user)
        os.mkdir(usersDirectory+'/'+user)
        os.mkdir(calibrationImagesDirectorypath)
        os.mkdir(dataDirectorypath)
    except FileExistsError:
        print('User directory exists already')
    switchUser(user)
def printActiveUser():
    if os.path.exists(activeUserDatapath):
        fp = open(activeUserDatapath, 'r+')
        currentUser = fp.read()
        fp.close()
        if (len(currentUser) == 0):
            print('no aktive user found')
        else:
            print('active User '+currentUser)
    else:
        print('no users exits yet. use command <create> to create')
def h_getAllUsers():
    if os.path.exists(usersDirectory):
        folder = os.path.dirname(os.path.abspath(__file__))
        folder = folder + '/' + usersDirectory
        return [f.name for f in os.scandir(folder) if f.is_dir()]
    else:
        return []
def printAllUsers():
    print(h_getAllUsers())
def deleteUser(user):
    allUsers = h_getAllUsers()
    if user in allUsers:
        fp = open(activeUserDatapath, 'r+')
        currentUser = fp.read()
        fp.close()
        if (len(currentUser) == 0) | (user == currentUser):
            #dont switch
            print('error: can not delete the currently active user. Create or switch to a different user before deleting.')
        else:
            print('deleting user '+user)
            buildPaths(user)
            shutil.rmtree(usersDirectory + '/' + user, ignore_errors=True)
            #os.rmdir(usersDirectory + '/' + user)
            switchUser(currentUser)
    else:#catch the case that the user doesnt exist.
        print('user '+user+' doesnt exits. There is nothing to delete')
def repairUser(user):
    #repair filesystem for user
    buildPaths(user)
    #check if every directory and file exits. if its not there create it
    if os.path.exists(calibrationImagesDirectorypath)==False:
        os.mkdir(calibrationImagesDirectorypath)
    if os.path.exists(dataDirectorypath)==False:
        os.mkdir(dataDirectorypath)

    #np.save(keyDatapath, np.array([]))
    #np.save(estimationResultDatapath, np.array([]))
    #np.save(hairAmountDatapath, np.array([]))
    #np.save(calibrationResultDatapath, np.array([]))
def repairUsers():
    print('attempting to repair users filesystem')
    users = h_getAllUsers()
    if len(users) == 0:
        if os.path.exists(activeUserDatapath): # no directory but active user exits
            fp = open(activeUserDatapath, 'r+')
            currentUser = fp.read()
            fp.close()
            if (len(currentUser) > 0):
                createUser(currentUser)
        else:
            print('no users found. Create new user with <create>')
            print('every user folder needs to be in the subdirectory Users')
        return
    currentUser = ''
    if os.path.exists(activeUserDatapath):
        fp = open(activeUserDatapath, 'r+')
        currentUser = fp.read()
        fp.close()
        if (len(currentUser) == 0):
            currentUser = users[0]
        if currentUser not in users:
            createUser(currentUser)
    else:
        fp = open(activeUserDatapath, 'w+')
        currentUser = users[0]

    for user in users:
        repairUser(user)
    switchUser(currentUser)
# endregion

def commandlinehandeling():
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})
    FUNCTION_MAP = {'calibrate': calibration,
                    'guess': guess, # args
                    'plot': plotEstimationResult,
                    'clearSave': clearSave,
                    'addCalibrationImage': addCalibrationImage, #args
                    'guessFolder': guessFolder,
                    'setGuessFunctions':setFunctions,
                    'automaticGuessFunctions':automaticGuessFunctions,
                    'saveTag': saveTag,
                    'manualGuess':manualGuess,
                    #checking result of calibration and debugging
                    'calibrationResult': guessTest,
                    'printFile': printFile,
                    'calculateError': calculateError,
                    'checkCalibration': checkCalibration,
                    'printResult': printResult,
                    #handeling users
                    'allUsers': printAllUsers,
                    'user': printActiveUser,
                    'delete': deleteUser, #args
                    'create': createUser, # args
                    'switch': switchUser, # args
                    'repairUsers': repairUsers,
                    'removeLastGuess': removeLastSaveImg,
                    'fullTest': fullTest,
                    'onlyGuessTest': onlyGuessTest,
                    'testEdgeDetection': testEdgeDetection
                    }
    parser = argparse.ArgumentParser(description='Estimate shed hair')
    parser.add_argument("command", choices=FUNCTION_MAP.keys(), help="command to be executed by the programm")
    parser.add_argument("-d", "--debug", help="show Images and extracted Image Data", action="store_true")
    parser.add_argument("arg1", nargs='?',help="depending on command: imagepath or user", type=str)
    parser.add_argument("arg2", nargs='?', help="for command addCalibrationImage: hair Amount")
    parser.add_argument("arg3", nargs='?' )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-r", action='store_true')#replace
    group.add_argument("-m",action = 'store_true')#use mean
    group.add_argument("-i", action = 'store_true')#ignore second duplicate
    group.add_argument("-k",action ='store_true')#keep duplicate
    group.add_argument("-a", action = 'store_true')
    parser.add_argument("-o","--useOldFunc", help="dont find new functions during calibration",action="store_true")
    args = parser.parse_args()
    global useOldFunc
    if args.useOldFunc:
        useOldFunc = True
    global duplicateHandelingMode
    if args.r:

        duplicateHandelingMode = 'r'
    if args.m:
        duplicateHandelingMode = 'm'
    if args.i:
        duplicateHandelingMode = 'i'
    if args.k:
        duplicateHandelingMode ='k'
    if args.a:
        duplicateHandelingMode='a'
    if args.debug:
        debugg(True)
    if args.debug == False:
        debugg(False)
    func = FUNCTION_MAP[args.command]
    if func == guess :
        if args.arg1 is None :
            print('Error: guess needs a path to an image as second positional argument.')
            return
        else:
            if args.arg2 is None:
                if args.arg3 is None:
                    func(args.arg1)
                else:
                    guessWithDaysAndTag(args.arg1,args.arg2,args.arg3)
            else:
                guessWithDaysOrTag(args.arg1,args.arg2) #second argument is a tag
    elif func==guessFolder:
        if args.arg1 is None :
            func('')
        else:
            func(args.arg1)
    elif func == addCalibrationImage:
        if args.arg1 is None :
            print('Error: addCalibrationImage needs a path to an image as second positional argument.')
            return
        else:
            func(args.arg1,args.arg2)
    elif func == saveTag:
        if (args.arg1 is None) | (args.arg2 is None):
            print('Error: saveTag needs 2 positional arguments. tag date')
        else:
            func(args.arg1, args.arg2)
    elif func == manualGuess:
        func(args.arg1,args.arg2)
    elif (func == deleteUser) | (func == createUser) | (func == switchUser) | (func == printFile)\
            |(func == calculateError) |(func == fullTest) |(func == onlyGuessTest) |(func == setFunctions) | (func == testEdgeDetection):
        if args.arg1 is None :
            print('Error: command needs second postional argument')
            return
        else:

            func(args.arg1)
    else:
        func()
if __name__ == "__main__":
    loadUser()
    commandlinehandeling()

    #debugg(True)
    #detect('Users/Bina/estimationImages/IMG_20200315_093236_10.jpg')
    #guessTest()
    #guess('D:/Eigene Dateien/Dokumente/GitHub/HairEstimation/HairEstimationPython/Users/Mummel/calibrationImages/30.jpg')
    #plotEstimationResult()
