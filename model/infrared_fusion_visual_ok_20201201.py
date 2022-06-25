'''
夜间红外与可见光配准+融合
配准矩阵用白天的照片
'''
import cv2
import numpy as np

# 图片配准函数，采用白天的可见光与红外灰度图，计算两者Surf共同特征点，之间的仿射矩阵。
# 使用条件是Opencv-python与opencv-contrib-python版本小于3.4.2.16
def Images_matching(img_base, img_target):
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

# 红外与可见光YUV像素融合，其中可见光为基准，红外需要配准
def Images_fusion(img_visual,img_infrared):
    img_visual_gray=cv2.cvtColor(img_visual,cv2.COLOR_BGR2GRAY)
    img_visual_YUV=cv2.cvtColor(img_visual,cv2.COLOR_BGR2YUV)
    img_infrared_gray=cv2.cvtColor(img_infrared,cv2.COLOR_BGR2GRAY)
    c = np.zeros_like(img_visual_gray)
    c = min(img_visual_gray.any(), img_infrared_gray.any())
    detlaV = img_visual_gray - c
    deltaI = img_infrared_gray - c
    newimg = cv2.merge([detlaV - deltaI, detlaV, deltaI])
    newimg_YUV = cv2.cvtColor(newimg, cv2.COLOR_BGR2YUV)
    fusion_YUV = cv2.merge([newimg_YUV[:, :, 0], img_visual_YUV[:, :, 1], img_visual_YUV[:, :, 2]])
    fusion_BGR = cv2.cvtColor(fusion_YUV, cv2.COLOR_YUV2BGR)
    return fusion_BGR

#
def main():
    matchimg_di=cv2.imread('basic_day_infrared.jpg')
    matchimg_dv=cv2.imread('basic_day_visual.jpg')
    matchimg_di_gray=cv2.cvtColor(matchimg_di,cv2.COLOR_BGR2GRAY)
    matchimg_dv_gray=cv2.cvtColor(matchimg_dv,cv2.COLOR_BGR2GRAY)

    orimg_nv=cv2.imread('night_visual_40m.jpg')
    orimg_ni=cv2.imread('night_infrared_40m.jpg')
    orimg_nv_gray=cv2.cvtColor(orimg_nv,cv2.COLOR_BGR2GRAY)
    # orimg_ni_gray=cv2.cvtColor(orimg_ni,cv2.COLOR_BGR2GRAY)
    #
    h, w = orimg_nv_gray.shape
    H=Images_matching(matchimg_dv_gray,matchimg_di_gray)

    # matched_base=cv2.warpPerspective(matchimg_di, H, (w, h))

    matched_ni = cv2.warpPerspective(orimg_ni, H, (w, h))
    fusion=Images_fusion(orimg_nv,matched_ni)

    cv2.imshow('0', cv2.resize(orimg_nv, (600, 400)))
    cv2.imshow('1', cv2.resize(fusion, (600, 400)))
    cv2.waitKey(0)

#
if __name__=='__main__':
    main()

