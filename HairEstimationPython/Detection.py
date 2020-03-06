#! /usr/bin/env python3

import cv2
import numpy as np
import scipy
from scipy import interpolate
from matplotlib import pyplot as plt


# region detect
def add(data, keys, key, value):
    data = np.append(data, value)
    keys = np.append(keys, key)
    return data, keys


def show(title, image):
    show = False
    if show:
        resize = True
        height = image.shape[0]
        width = image.shape[1]
        # image = cv2.resize(image, (width, height))
        if resize:
            image = cv2.resize(image, (int(width * 0.2), int(height * 0.2)))
        cv2.imshow(title, image)


# region cropDots
def processMatchResult(img_rgb, res, threshold, templatew, templateh):
    loc = np.where(res >= threshold)
    cnt = 0
    actualpoints = np.array([])
    differentPointthreshhold = (min(templateh, templateh) / 2) ** 2  # adaptive threshhold
    # print(differentPointthreshhold)
    rectangleImage = img_rgb.copy()
    for pt in zip(*loc[::-1]):
        if fuzzycontains(actualpoints, pt, differentPointthreshhold):
            actualpoints = np.append(actualpoints, pt)
            cnt = cnt + 1
        cv2.rectangle(rectangleImage, pt, (pt[0] + templatew, pt[1] + templateh), (0, 0, 255), 2)
    # show('foundDots', rectangleImage) #show found dots with red rectangle around them
    # print('actual points', actualpoints)
    return cnt, actualpoints, rectangleImage


def fuzzycontains(actualpoints, pt, differentPointthreshhold):
    xpoints = actualpoints[::2]
    ypoints = actualpoints[1::2]
    for actpoints in zip(xpoints, ypoints):
        if (pt[0] - actpoints[0]) ** 2 + (pt[1] - actpoints[1]) ** 2 < differentPointthreshhold:
            return False
    return True


def cropDots(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('TemplateDot.jpg', 0)
    templatew, templateh = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    # show('img',img_rgb)
    # show('match res', res)

    threshold = 0.8
    cnt, actualpoints, rectImg = processMatchResult(img_rgb, res, threshold, templatew, templateh)
    if cnt == 0:
        # trying out inverted image. for blond hair on black background
        revImg = 255 - img_gray
        res = cv2.matchTemplate(revImg, template, cv2.TM_CCOEFF_NORMED)
        # show('revRes', res)
        cnt, actualpoints, rectImg = processMatchResult(img_rgb, res, threshold, templatew, templateh)
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
            # print(crop_img)
            # show('crop image', crop_img)
            retImg = crop_img
            show('rectImg', rectImg)
            retry = False
        if cnt > 4:
            # found to many... make threshhold higher
            threshold = threshold + 0.01
            cnt, actualpoints, rectImg = processMatchResult(img_rgb, res, threshold, templatew, templateh)
        if cnt < 4:
            # found to few dots.. make threshhold lower to let more pass
            threshold = threshold - 0.01
            cnt, actualpoints, rectImg = processMatchResult(img_rgb, res, threshold, templatew, templateh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return retImg


# endregion
def hairPixelPercentage(data, keys, img):
    sum = np.count_nonzero(img > 0)
    all = np.size(img)
    intensityShare = data[0] / sum
    data, keys = add(data, keys, 'intensityShare', intensityShare)
    data, keys = add(data, keys, 'hairpixels', sum)
    data, keys = add(data, keys, 'imagepixels', all)
    percentage = (sum / all) * 100
    data, keys = add(data, keys, 'percentage', percentage)

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
    removalThreshold = 600
    # show('intenisty', intensity)
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
            # remove the region
            returnImage = mask * 0 + (1 - mask) * returnImage
    show('small regions removed', returnImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
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
    intensity = removeSmallRegions(intensity, orig)
    showMissed(intensity, gray, orig)

    intensitySum = np.sum(intensity)
    data, keys = add(data, keys, 'intensitySum:', intensitySum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return data, keys, intensity


def showMissed(intensity, gray, orig):
    # show how much hair is missed
    hairPixelMask = np.ones(gray.shape[:2], dtype="uint8")
    hairPixelMask[:, :] = (intensity != 0)  # 0 or 1 depending on wehter it is ==0
    ret = gray.copy()
    ret = hairPixelMask * 255 + (1 - hairPixelMask) * gray
    show('missed hair', ret)
    show('orig', orig)


def edgeProcess(data, keys, orig, blur):
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    if blur:
        gray = cv2.medianBlur(gray, 5)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(gray, 40, 200, kernel)
    show('edges', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    data, keys, intensity = hairPixelIntensity(data, keys, orig, gray, edges)
    data, keys = hairPixelPercentage(data, keys, intensity)
    return data, keys, edges, intensity


def backgroundRegions(data, keys, intensity):  # only uses intensity image. background black, hair white
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
    data, keys = add(data, keys, 'all pixels', allPixelSum)
    sectionNum = np.unique(markers).size
    data, keys = add(data, keys, 'number of section:', sectionNum)
    innerSectionNum = sectionNum - 2
    data, keys = add(data, keys, 'number of section inclosed:', innerSectionNum)  # -1 is space between and 1 is hair
    # finding size of outermost section
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == 2] = 1  # set pixels marked with 2 to 1. rest is 0.

    image = intensity.copy()
    image = mask * 255 + (1 - mask) * 0
    show('outer section', image)
    outerSectionSum = np.sum(mask)
    data, keys = add(data, keys, 'outerSectionSum', outerSectionSum)
    data, keys = add(data, keys, 'outerSectionPercentage', outerSectionSum / allPixelSum)
    innerSectionSum = allPixelSum - outerSectionSum
    data, keys = add(data, keys, 'innerSectionSum', innerSectionSum)
    innerSectionAvg = innerSectionSum / innerSectionNum
    data, keys = add(data, keys, 'innserSectionAvgSize', innerSectionAvg)
    data, keys = add(data, keys, 'innerSectionAvgSize Percentage', innerSectionAvg / allPixelSum)
    sizes = np.array([])
    for marker in np.unique(markers):
        if marker > 2:
            mask = np.zeros(gray.shape, dtype="uint8")
            mask[markers == marker] = 1
            sizes = np.append(sizes, np.sum(mask))
    innerSectionSizeVariance = np.var(sizes)
    data, keys = add(data, keys, 'innerSectionSizeVariance', innerSectionSizeVariance)
    data, keys = add(data, keys, 'std', np.std(sizes))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return data, keys


def showstats(data, keys):
    for i in range(np.size(data)):
        print(keys[i], data[i])
    print()


def detect(path):
    print(path)
    blur = False
    img_rgb = cv2.imread(path)
    if img_rgb is None:
        print('no image with that path found')
        return
    data = np.array([])
    keys = np.array([])
    croped = cropDots(img_rgb)
    data, keys, edges, intensity = edgeProcess(data, keys, croped, blur)
    data, keys = backgroundRegions(data, keys, intensity)
    # showstats( data,keys)
    return data, keys


def testEdgeDetection(path):
    orig = cv2.imread(path)

    blur = cv2.blur(orig, (5, 5))
    show('blur', blur)
    blur = cv2.GaussianBlur(orig, (5, 5), 0)
    show('gaussianBlur', blur)
    # blur = cv2.medianBlur(orig, 3)
    show('meidanblur', blur)
    # blur = orig
    show('orig', orig)
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # show('thresh', thresh)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.Canny(orig, 30, 200, kernel)
    show('orig edges', edges)
    edges2 = cv2.Canny(blur, 30, 200, kernel)
    show('blur canny', edges2)
    show('diff', edges - edges2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# endregion
def calibrateProcessImages(calibrationIn):
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.5f}'.format})
    alldata = np.array([])
    calibrationPaths = calibrationIn[::2]
    hairAmount = calibrationIn[1::2]
    for path in calibrationPaths:
        data, keys = detect(path)
        alldata = np.append(alldata, data)
    # f = open("calibrationImageStats.", "w+")
    print('save', alldata)
    np.save('test.out', alldata)
    np.save('keys.out', keys)
    np.save('hairamount.out', hairAmount)
    # f.write(alldata)
    # f.close()
    # save alldata and stats in a file
def addCalibrationImage(path,amount):
    np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.5f}'.format})
    data, keys = detect(path)
    alldata = np.load('test.out' + '.npy')
    print(alldata)
    print(data)
    alldata = np.append(alldata, data)
    hairAmount = np.load('hairamount.out' + '.npy')
    print(hairAmount)
    hairAmount = np.append(hairAmount, amount)
    print(hairAmount)
    np.save('test.out', alldata)
    np.save('hairamount.out', hairAmount)
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

    fitx = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(30, 10))
    ax.plot(fitx, spl(fitx))
    ax.plot(fitx, lin(a, b, fitx))
    ax.scatter(x, y)
    plt.show()
    print('Guess by',keys[datapoint],lin(a, b, imgdata[datapoint]))
    print('Guess by', keys[datapoint], spl(imgdata[datapoint]))
    # fig2,ax2 = plt.subplot()
    # ax.plot()
    # use percentage for initial estimation of hairamount.

def lin(a,b,x):
    return a*x+b
def linregress(x,y):
    a = np.cov(x, y)[0, 1] / np.cov(x, y)[0, 0]  # slope
    b = y.mean() - a * x.mean()  # intersept
    return a,b
def guess(path):
    # read alldata and stats from file.
    # this way we can start calibrateStats without having to
    alldata = np.load('test.out' + '.npy')
    keys = np.load('keys.out' + '.npy')
    hairAmount = np.load('hairamount.out' + '.npy')
    data,_ = detect(path)
    guessByDataPoint(alldata,keys,hairAmount,data,7)
    guessByDataPoint(alldata,keys,hairAmount,data,4)

if __name__ == "__main__":
    # testEdgeDetection('NewBlack_Felina_5.jpg')
    # testEdgeDetection('Dot_Mummel_4.jpg')
    # testEdgeDetection('Black_Mummel_Test_ (2).jpg')
    # testEdgeDetection('Black_Mummel_Test_ (3).jpg')
    # testEdgeDetection('Dot_Felina_4_ 1.jpg')
    # testEdgeDetection('Dot_Mummel_1_3.jpg')
    # testEdgeDetection('Dot_Felina_ large.jpg')
    # testEdgeDetection('Dot_Felina_4_ 2.jpg')

    calibrationIn = np.array(['Dot_Mummel_1.jpg',1,'Dot_Mummel_3_ (1).jpg',3,'Dot_Mummel_10.jpg',10,'Dot_Mummel_15_ (2).jpg',15,'Dot_Mummel_22.jpg',22,'Dot_Mummel_25_ (1).jpg',25, 'Dot_mummel_30.jpg',30,'Dot_Mummel_40.jpg',40, 'Dot_Mummel_50 (3).jpg', 50,'Dot_mummel_60 (2).jpg',60])
    #calibrateProcessImages(calibrationIn)
    #guess('Dot_Mummel_10.jpg')
    addCalibrationImage('Dot_Mummel_21 (1).jpg',21)
    guess('Dot_Mummel_4.jpg')
    guess('Dot_Mummel_21 (1).jpg')

    # detect('Dot_Mummel_10.jpg')
    # detect('Dot_Mummel_21 (1).jpg')
    # detect('Dot_Mummel_21 (2).jpg')
    # detect('Dot_Mummel_30.jpg')
    # detect('Dot_Mummel_40.jpg')
    # detect('Dot_Mummel_50 (2).jpg')
    # detect('Dot_Mummel_60 (2).jpg')
    # detect('Dot_Mummel_60 (1).jpg')

    # detect('NewBlack_Felina_5.jpg', False)
    # detect('Dot_Felina_4_ 2.jpg',True)
    # detect('Dot_Felina_4_ 3.jpg',True)
    # detect('testRG.png')
    # detect('Dot_Mummel_1.jpg')
    # detect('Dot_Mummel_1_3.jpg',False)
    # detect('Dot_Mummel_1_4.jpg',False)
    # detect('Dot_Mummel_4.jpg',False)
    # detect('Dot_Mummel_medium.jpg')
