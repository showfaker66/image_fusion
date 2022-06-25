import pywt
import cv2
import numpy as np

# 按照不同的融合方式，计算通道融合
def fuseCoeff(cooef1, cooef2, method):
    if (method == 'mean'):
        cooef = (abs(cooef1) + abs(cooef2)) / 2
        # cooef=0.5*cooef1+1.5*cooef2
    elif (method == 'min'):
        cooef = np.minimum(cooef1, cooef2)
    elif (method == 'max'):
        cooef = np.maximum(abs(cooef1), abs(cooef2))
    return cooef

def main():
    # Params
    FUSION_METHOD = 'mean'  # Can be 'min' || 'max || anything you choose according theory
    FUSION_METHOD1 = 'max'
    # Read the two image
    # orimg_nv = cv2.imread('basic_day_visual.jpg')
    # orimg_ni = cv2.imread('basic_day_infrared.jpg')

    orimg_nv = cv2.imread('night_visual_40m.jpg')
    orimg_ni = cv2.imread('night_infrared_40m.jpg')

    HSVimg_nv=cv2.cvtColor(orimg_nv,cv2.COLOR_BGR2HSV)
    grayimg_ni=cv2.cvtColor(orimg_ni,cv2.COLOR_BGR2GRAY)
    Vimg_nv=HSVimg_nv[:,:,2]

    wavelet = 'db2'
    cooef_nv = pywt.wavedec2(Vimg_nv[:, :], wavelet, level=1)
    cooef_ni = pywt.wavedec2(grayimg_ni[:, :], wavelet, level=1)

    fusedCooef = []
    for i in range(len(cooef_nv)):
        if (i == 0):
            fusedCooef.append(fuseCoeff(cooef_nv[0], cooef_ni[0], FUSION_METHOD))
        else:
            # For the rest of the levels we have tupels with 3 coeeficents
            c1 = fuseCoeff(cooef_nv[i][0], cooef_ni[i][0], FUSION_METHOD1)
            c2 = fuseCoeff(cooef_nv[i][1], cooef_ni[i][1], FUSION_METHOD1)
            c3 = fuseCoeff(cooef_nv[i][2], cooef_ni[i][2], FUSION_METHOD1)
            fusedCooef.append((c1, c2, c3))
    fusedImage = pywt.waverec2(fusedCooef, wavelet)
    fusedImage1 = np.multiply(np.divide(fusedImage - np.min(fusedImage), (np.max(fusedImage) - np.min(fusedImage))),
                              255)
    fusedImage1 = fusedImage1.astype(np.uint8)
    Vimg_new=fusedImage1
    fusedHSV=cv2.merge([HSVimg_nv[:,:,0],HSVimg_nv[:,:,1],Vimg_new])
    fusedBGR=cv2.cvtColor(fusedHSV,cv2.COLOR_HSV2BGR)

    cv2.imshow('0', cv2.resize(orimg_nv, (600, 400)))
    cv2.imshow('1', cv2.resize(orimg_ni, (600, 400)))
    cv2.imshow('2', cv2.resize(fusedBGR, (600, 400)))
    # cv2.imwrite("win.bmp", fusedImage1)
    cv2.waitKey(0)

if __name__=='__main__':
    main()