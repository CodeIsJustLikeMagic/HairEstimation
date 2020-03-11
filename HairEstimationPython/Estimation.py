#! /usr/bin/env python
import datetime
import cv2
import os
import numpy as np
import re
import matplotlib.dates as mdates
import argparse
from scipy import interpolate
from matplotlib import pyplot as plt


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
    # show('foundDots', rectangleImage) #show found dots with red rectangle around them
    # print('actual points', actualpoints)
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
    template = cv2.imread('TemplateDot.jpg', 0)
    templatew, templateh = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # show('img',img_rgb)
    # show('match res', res)

    threshold = 0.8
    cnt, actualpoints, rectImg = h_processMatchResult(img_rgb, res, threshold, templatew, templateh)
    if cnt == 0:
        # trying out inverted image. for blond hair on black background
        revImg = 255 - img_gray
        res = cv2.matchTemplate(revImg, template, cv2.TM_CCOEFF_NORMED)
        # show('revRes', res)
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
        # print(threshold)
        tries = tries + 1
        if tries > 30:
            print('gave up. Best found:', cnt)

            cv2.waitKey(0)
            cv2.destroyAllWindows()
            retry = False
        if cnt == 4:
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
        if cnt > 4:
            # found to many... make threshhold higher
            threshold = threshold + 0.01
            cnt, actualpoints, rectImg = h_processMatchResult(img_rgb, res, threshold, templatew, templateh)
        if cnt < 4:
            # found to few dots.. make threshhold lower to let more pass
            threshold = threshold - 0.01
            cnt, actualpoints, rectImg = h_processMatchResult(img_rgb, res, threshold, templatew, templateh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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

#not used
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
    img_erosion = cv2.erode(img_dilation, kernel, iterations=1)
    hairPixels = img_dilation  # edges

    # skel = skeletonize(img_dilation)
    hairPixelMask = np.ones(orig.shape[:2], dtype="uint8")
    hairPixelMask[:, :] = (hairPixels != 0)  # 0 or 1 depending on wehter it is ==0
    hairOnBlack = gray.copy()
    hairOnBlack = hairPixelMask * gray

    hairOnWhite = gray.copy()
    # no hair*255 + hair * hair On Black(gray)
    # if no hair found, make it white. if hair found copy color from grayscale original
    hairOnWhite = (1 - hairPixelMask) * 255 + hairPixelMask * gray

    # show('orig', orig)
    # show('hairOnWhite', hairOnWhite)
    # darker is more intense in this case

    # average color of background
    imediateBackground = cv2.dilate(edges, kernel, iterations=4)
    imediateBackground = imediateBackground - hairPixels
    imediateBackground = (1 - imediateBackground) * 0 + imediateBackground * gray
    # count colors that are ligher than black
    backGroundPixels = np.count_nonzero(imediateBackground > 0)
    # print(backGroundPixels)
    # average color = sum of colors / pixels
    averageBackgroundColor = np.sum(imediateBackground) / backGroundPixels
    # print('backgroundcolor', averageBackgroundColor)

    # average color of everything
    avgColor = np.sum(gray) / np.size(gray)

    hairOnAverage = gray.copy()
    hairOnAverage = (1 - hairPixelMask) * int(avgColor) + hairPixelMask * gray
    # show('hair on Average background color', hairOnAverage)

    # average hair color
    sum = np.sum(hairOnBlack)
    averageHairColor = sum / np.count_nonzero(hairPixels == 255)

    # print('average hair color ', averageHairColor)

    # if average hair color is lighter than the background, flip hair on average
    # brither is more intense
    intensity = hairOnWhite.copy()
    intensity = 255 - intensity
    h_show('intenstiy', intensity)
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


def edgeProcess(data, keys, orig, blur):
    print('detecting Hair via edge detection')
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(gray, 40, 200, kernel)
    h_show('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    data, keys, intensity = hairPixelIntensity(data, keys, orig, gray, edges)
    data, keys = hairPixelPercentage(data, keys, intensity)
    return data, keys, edges, intensity


def backgroundRegions(data, keys, intensity):  # only uses intensity image. background black, hair white
    print('processing background regions...')
    # show('input intensity', intensity)
    img = cv2.cvtColor(intensity, cv2.COLOR_GRAY2RGB)
    gray = intensity
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # thresh = 255-thresh
    # show('thresh',thresh)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # show('opening', opening)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=1)
    # show('sure_bg',sure_bg)
    # Finding sure foreground area
    sure_fg = opening.copy()
    # dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    # show('dist_transform',dist_transform)
    # ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    # show('sure_fg', sure_fg)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # show('unknown',unknown)
    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # ok that looks like it worked. hair(black) is now 2. and all the white parts(background) are labelded 1+
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0
    markers = cv2.watershed(img, markers)
    backgroundSum = 0

    allPixelSum = img.shape[0] * img.shape[1]
    data, keys = h_add(data, keys, 'all pixels', allPixelSum)
    sectionNum = np.unique(markers).size
    data, keys = h_add(data, keys, 'number of section', sectionNum)
    innerSectionNum = sectionNum - 2
    data, keys = h_add(data, keys, 'number of section inclosed', innerSectionNum)  # -1 is space between and 1 is hair
    # finding size of outermost section
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == 2] = 1  # set pixels marked with 2 to 1. rest is 0.

    image = intensity.copy()
    image = mask * 255 + (1 - mask) * 0
    h_show('outer section', image)
    outerSectionSum = np.sum(mask)
    data, keys = h_add(data, keys, 'outerSectionSum', outerSectionSum)
    data, keys = h_add(data, keys, 'outerSectionPercentage', outerSectionSum / allPixelSum)
    innerSectionSum = allPixelSum - outerSectionSum
    data, keys = h_add(data, keys, 'innerSectionSum', innerSectionSum)
    innerSectionAvg = innerSectionSum / innerSectionNum
    data, keys = h_add(data, keys, 'innserSectionAvgSize', innerSectionAvg)
    data, keys = h_add(data, keys, 'innerSectionAvgSize Percentage', innerSectionAvg / allPixelSum)
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
    return data, keys


def h_showstats(data, keys):
    for i in range(np.size(data)):
        print(keys[i], data[i])
    print()


def detect(path):
    print('image path',path)
    blur = False
    img_rgb = cv2.imread(path)
    data = np.array([])
    keys = np.array([])
    if img_rgb is None:
        print('no image with that path found')
        return data,keys
    croped = cropDots(img_rgb)
    data, keys, edges, intensity = edgeProcess(data, keys, croped, blur)
    data, keys = backgroundRegions(data, keys, intensity)
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
def calibration():
    #find every image in the folder and use them to calibrate with
    #retrieve names of the images
    paths = os.listdir(calibrationImagesDirectorypath)
    #add the image names to the relative path
    paths = [calibrationImagesDirectorypath+'/' + path for path in paths]
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
def guessFolder(folder):
    #estimate every image in the folder
    folder = activeUserDirectory +'/'+folder
    paths = os.listdir(folder)
    paths = [folder+'/' + path for path in paths]
    for path in paths:
        guess(path)
def guessByDataPoint(alldata,keys,hairAmount,imgdata, datapoint):
    calibrationDataX = alldata[datapoint::np.size(keys)]
    hairAmount = np.array(list(map(int, hairAmount)))
    x = calibrationDataX
    y = hairAmount
    # make sure x goes up, sort both by x
    inds = x.argsort()
    x = x[inds]
    y = y[inds]

    a, b = linregress(x, y)
    spl = interpolate.InterpolatedUnivariateSpline(x, y,k=1)

    if debugstate:
        fitx = np.linspace(x.min(), x.max(), 100)
        fig, ax = plt.subplots(figsize=(30, 10))
        ax.plot(fitx, spl(fitx))
        ax.plot(fitx, lin(a, b, fitx))
        ax.scatter(x, y)
        ax.set_title('Guess by '+keys[datapoint]+' funktion')
        ax.set_xlabel(keys[datapoint])
        ax.set_ylabel('amount of hair')
        plt.show()
    linguess = lin(a, b, imgdata[datapoint])
    splguess = spl(imgdata[datapoint])
    print('(linear funktion) Guess by',keys[datapoint],linguess)
    print('(spline funktion) Guess by', keys[datapoint], splguess)
    # fig2,ax2 = plt.subplot()
    # ax.plot()
    # use percentage for initial estimation of hairamount.
    return linguess,splguess

def lin(a,b,x):
    return a*x+b
def linregress(x,y):
    a = np.cov(x, y)[0, 1] / np.cov(x, y)[0, 0]  # slope
    b = y.mean() - a * x.mean()  # intersept
    return a,b
def guessCombo(alldata,keys,hairAmount,imgdata):
    hairperc = alldata[4::np.size(keys)]
    outersec = alldata[9::np.size(keys)]
    calibrationDataX = hairAmount/hairperc*outersec
    hairAmount = np.array(list(map(int, hairAmount)))
    x = calibrationDataX
    y = hairAmount
    # make sure x goes up, sort both by x
    inds = x.argsort()
    x = x[inds]
    y = y[inds]

    a, b = linregress(x, y)
    spl = interpolate.InterpolatedUnivariateSpline(x, y, k=1)

    fitx = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.plot(fitx, spl(fitx))
    ax.plot(fitx, lin(a, b, fitx))
    ax.scatter(x, y)
    plt.show()
    imgY = 3
    linguess = lin(a, b, imgdata[datapoint])
    splguess = spl(imgdata[datapoint])
    print('Guess by', keys[datapoint], linguess)
    print('Guess by', keys[datapoint], splguess)
    # fig2,ax2 = plt.subplot()
    # ax.plot()
    # use percentage for initial estimation of hairamount.
    return linguess, splguess
def guess(path):
    # read alldata and stats from file.
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
    print('guessing the amount of hair in the picture')
    data,_ = detect(path)
    #print('alldata', alldata)
    #print(data)
    #print(keys)
    #print(set(zip(keys,data)))

    #find out if hair is looser or tigher than normal
    outerSectionPercentageMean = np.mean(alldata[9::np.size(keys)])
    #print(alldata[9::np.size(keys)])
    #print(outerSectionPercentageMean)
    isloosethreshhold = outerSectionPercentageMean/4
    #print('outersectionPercentage',data[9], outerSectionPercentageMean+isloosethreshhold)
    if data[9] < outerSectionPercentageMean-isloosethreshhold:
        print('this hair bunch is loose')
        print('prefer the lower estimations on this one. maybe even lower the percentage')
        data[7] = data[7]/1.5
        data[4] = data[4]/1.5
    estimations = np.array([])
    estimations = np.append(estimations, guessByDataPoint(alldata,keys,hairAmount,data,7))
    estimations = np.append(estimations, guessByDataPoint(alldata,keys,hairAmount,data,4))
    mean = np.mean(estimations)
    for e in estimations:
        if e < 0:
            np.delete(estimations,e)
    mean = np.mean(estimations)
    res = round(mean,0)
    print('mean', mean,'res',res, 'hair percent:', data[4], 'outer sectionSize:', data[9],'innersectionNum', data[7])
    save(path,res)

def save(path,mean):

    #save image data to a file with all the previous estimations by date
    numbers = re.findall(r'\d+', path)
    numbers = numbers[0]
    year = numbers[0:4:]
    month = numbers[4:6:]
    day = numbers[6:8:]
    ymd = year + '-' + month + '-' + day
    estimationResult = np.array([ymd,mean])
    oldData = np.array([])
    if os.path.exists(estimationResultDatapath+'.npy'):
        oldData = np.load(estimationResultDatapath+'.npy')
        oldDates = oldData [::2]
        if ymd in oldDates:
            print('found duplicate date')
    newData = np.append(oldData, estimationResult)
    print('saving data point', ymd, mean, 'ImagePath:', path)
    print()
    np.save(estimationResultDatapath ,newData)
#endregion
# region showanddebugg
def removeLastSaveImg():
    try:
        data = np.load(estimationResultDatapath+'.npy')
        hairAmounts = np.load(hairAmountDatapath+'.npy')
    except:
        print('no data found')
    #data has one pair per img. hairAmounts has one entry per img
    print(data)
    data = data[:-2]
    print(data)
    hairAmounts = hairAmounts[:-1]
    np.save(estimationResultDatapath, data)
    np.save(hairAmountDatapath, hairAmounts)
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
    locator = mdates.DayLocator(interval = 1)
    ax.xaxis.set_major_locator(locator)
    plt.scatter(dates, hairAmounts)
    plt.plot(dates,hairAmounts)
    plt.gcf().autofmt_xdate()
    ax.grid()
    plt.show()
def clearSave():
    np.save(estimationResultDatapath,np.array([]))
    print('erased saved estimation results')
# endregion
datapath = 'EstimationDataUser1'

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
def buildPaths(user):
    global calibrationImagesDirectorypath
    global dataDirectorypath
    global calibrationResultDatapath
    global keyDatapath
    global estimationResultDatapath
    global hairAmountDatapath
    global activeUserDirectory
    activeUserDirectory = usersDirectory +'/'+user
    calibrationImagesDirectorypath = usersDirectory + '/' + user + '/calibrationImages'
    dataDirectorypath = usersDirectory+'/'+user + '/data'
    calibrationResultDatapath = dataDirectorypath + '/calibrationData.out'
    keyDatapath  = dataDirectorypath+'/calibkeyData.out'
    estimationResultDatapath = dataDirectorypath+'/result.out'
    hairAmountDatapath = dataDirectorypath+'/calibhairAmount.out'
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
            try:
                os.remove(keyDatapath+'.npy')
                os.remove(estimationResultDatapath+'.npy')
                os.remove(hairAmountDatapath+'.npy')
                os.remove(calibrationResultDatapath + '.npy')
            except FileNotFoundError:
                pass
            os.rmdir(calibrationImagesDirectorypath)
            os.rmdir(dataDirectorypath)
            os.rmdir(usersDirectory + '/' + user)
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
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.5f}'.format})
    FUNCTION_MAP = {'calibrate': calibration,
                    'guess': guess, # args
                    'plot': plotEstimationResult,
                    'clearSave': clearSave,
                    'addCalibrationImage': addCalibrationImage, #args
                    'guessFolder': guessFolder,
                    #handeling users
                    'allUsers': printAllUsers,
                    'activeUser': printActiveUser,
                    'delete': deleteUser, #args
                    'create': createUser, # args
                    'switch': switchUser, # args
                    'repairUsers': repairUsers,
                    'removeLastGuess': removeLastSaveImg
                    }
    parser = argparse.ArgumentParser(description='Estimate shed hair')
    parser.add_argument("command", choices=FUNCTION_MAP.keys(), help="command to be executed by the programm")
    parser.add_argument("-d", "--debug", help="show Images and extracted Image Data", action="store_true")
    parser.add_argument("arg1", nargs='?',help="depending on command: imagepath or user", type=str)
    parser.add_argument("arg2", nargs='?', help="for command addCalibrationImage: hair Amount")
    args = parser.parse_args()
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
            func(args.arg1)
    elif func==guessFolder:
        if args.arg1 is None :
            print('Error: guess needs a path to a folder as second positional argument.')
            return
        else:
            func(args.arg1)
    elif func == addCalibrationImage:
        if args.arg1 is None :
            print('Error: addCalibrationImage needs a path to an image as second positional argument.')
            return
        else:
            func(args.arg1,args.arg2)
    elif (func == deleteUser) | (func == createUser) | (func == switchUser):
        if args.arg1 is None :
            print('Error: command needs a user name as second postional argument')
            return
        else:
            func(args.arg1)
    else:
        func()
if __name__ == "__main__":
    loadUser()
    commandlinehandeling()

    #addCalibrationImage('Dot_Mummel_21 (1).jpg',21)
    #guess('estimationInput/IMG_20200306_104544_22.jpg')
    #guess('estimationInput/IMG_20200306_104949_22.jpg')
    #guess('Dot_Mummel_4.jpg')
    #guess('Dot_Mummel_60 (1).jpg')
    #guess('Dot_Mummel_21 (2).jpg')

    #plotEstimationResult()
