'''
红外与可见光融合
1.采用对比度拉伸，完成红外图像增强；
2.采用Surf特征点匹配，完成图像配准；
3.采用HSV通道小波变换，完成图像融合
使用条件是Opencv-python与opencv-contrib-python版本小于3.4.2.16
2020-12-2
'''
import cv2
import numpy as np
import pywt #引入小波模块

# 裁剪线性RGB对比度拉伸：（去掉2%百分位以下的数，去掉98%百分位以上的数，上下百分位数一般相同，并设置输出上下限）
def truncated_linear_stretch(image, truncated_value=2, maxout=255, min_out=0):
    def gray_process(gray, maxout=maxout, minout=min_out):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray_new = ((maxout - minout) / (truncated_up - truncated_down)) * gray
        gray_new[gray_new < minout] = minout
        gray_new[gray_new > maxout] = maxout
        return np.uint8(gray_new)
    (b, g, r) = cv2.split(image)
    b = gray_process(b)
    g = gray_process(g)
    r = gray_process(r)
    result = cv2.merge((b, g, r))# 合并每一个通道
    return result
# RGB图片配准函数，采用白天的可见光与红外灰度图，计算两者Surf共同特征点，之间的仿射矩阵。
def Images_matching(img_base, img_target):
    img_base=cv2.cvtColor(img_base,cv2.COLOR_BGR2GRAY)
    img_target=cv2.cvtColor(img_target,cv2.COLOR_BGR2GRAY)
    hessian = 400
    surf = cv2.xfeatures2d.SURF_create(hessian)
    kp1, des1 = surf.detectAndCompute(img_base, None)
    kp2, des2 = surf.detectAndCompute(img_target, None)
    FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
    searchParams = dict(checks=50)  # 指定递归次数
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器
    matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点
    good = []
    # 提取优秀的特征点
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
            good.append(m)
    src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引
    H = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)  # 生成变换矩阵
    return H[0]
# HSV通道小波变换RGB融合，低频均值高频最大值
def Images_fusion(img_base,img_target):
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
    # HSV的小波融合过程
    LOW_METHOD = 'mean'
    HIGH_METHOD = 'max'
    HSVimg_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2HSV)#可见光转换为HSV
    grayimg_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)#红外灰度化
    Vimg_base = HSVimg_base[:, :, 2]
    wavelet = 'db2'
    cooef_base = pywt.wavedec2(Vimg_base[:, :], wavelet, level=1)#base灰度图的小波展开
    cooef_target = pywt.wavedec2(grayimg_target[:, :], wavelet, level=1)#target灰度图的小波展开
    fusedCooef = []
    for i in range(len(cooef_base)):
        if (i == 0):
            fusedCooef.append(fuseCoeff(cooef_base[0], cooef_target[0], LOW_METHOD))#低频部分取均值
        else:
            # 高频部分取最大值
            c1 = fuseCoeff(cooef_base[i][0], cooef_target[i][0], HIGH_METHOD)
            c2 = fuseCoeff(cooef_base[i][1], cooef_target[i][1], HIGH_METHOD)
            c3 = fuseCoeff(cooef_base[i][2], cooef_target[i][2], HIGH_METHOD)
            fusedCooef.append((c1, c2, c3))#高频合并
    tempfusedImage = pywt.waverec2(fusedCooef, wavelet)#小波逆变换
    fusedImage = np.multiply(np.divide(tempfusedImage - np.min(tempfusedImage), (np.max(tempfusedImage) -\
                            np.min(tempfusedImage))),255) #逆变换后归一至（0，255）
    fusedImage = fusedImage.astype(np.uint8)
    Vimg_new = fusedImage
    fusedHSV = cv2.merge([HSVimg_base[:, :, 0], HSVimg_base[:, :, 1], Vimg_new])#用小波变换替换V通道
    fusedBGR = cv2.cvtColor(fusedHSV, cv2.COLOR_HSV2BGR)#融合后的HSV转为BGR
    return fusedBGR
#
def main():
    matchimg_di = cv2.imread('basic_day_infrared.jpg')
    matchimg_dv = cv2.imread('basic_day_visual.jpg')
    # orimg_nv=matchimg_dv
    # orimg_ni=matchimg_di
    # orimg_nv = cv2.imread('night_visual_40m.jpg')
    # orimg_ni = cv2.imread('night_infrared_40m.jpg')
    # orimg_nv = cv2.imread('images/left_2020_11_30-20_06_54.jpg')
    # orimg_ni = cv2.imread('images/right_2020_11_30-20_06_54.jpg')
    orimg_nv = cv2.imread('images/left_2020_11_30-20_08_09.jpg')
    orimg_ni = cv2.imread('images/right_2020_11_30-20_08_09.jpg')

    #用白天图像进行配准
    enhance_matchimg_di = truncated_linear_stretch(matchimg_di)#配准模板红外图像RGB增强
    h, w = orimg_nv.shape[:2]
    H = Images_matching(matchimg_dv, enhance_matchimg_di)

    enhance_orimg_ni=truncated_linear_stretch(orimg_ni)#需融合红外图像RGB增强
    matched_ni = cv2.warpPerspective(enhance_orimg_ni, H, (w, h))#红外图像按照仿射矩阵配准

    fusion = Images_fusion(orimg_nv, matched_ni)

    cv2.imshow('0', cv2.resize(orimg_nv, (600, 400)))
    cv2.imshow('1', cv2.resize(orimg_ni, (600, 400)))
    cv2.imshow('2', cv2.resize(fusion, (600, 400)))
    cv2.waitKey(0)


if __name__ == '__main__':
    main()

