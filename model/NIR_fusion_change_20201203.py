'''
红外与可见光融合
1.采用对比度拉伸，完成红外图像增强；
2.采用Surf特征点匹配，完成图像配准；
3.采用HSV通道小波变换，完成图像融合
使用条件是Opencv-python与opencv-contrib-python版本小于3.4.2.16
2020-12-2
尝试：
先融合再增强，并将通道改为YUV
2020-12-3
'''
import cv2
import numpy as np
import pywt #引入小波模块
import time
from PIL import Image

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
    # 初始化surf算子
    surf = cv2.xfeatures2d.SURF_create(hessian)
    # surf = cv2.xfeatures2d_SURF.create(hessian)
    # surf = cv2.SIFT_create(hessian)
    # 使用surf算子计算特征点和特征点周围的特征向量
    kp1, des1 = surf.detectAndCompute(img_base, None)  # 1136    1136, 64
    kp2, des2 = surf.detectAndCompute(img_target, None)
    # 进行KNN特征匹配
    FLANN_INDEX_KDTREE = 0  # 建立FLANN匹配器的参数
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)  # 配置索引，密度树的数量为5
    searchParams = dict(checks=50)  # 指定递归次数
    flann = cv2.FlannBasedMatcher(indexParams, searchParams)  # 建立匹配器
    matches = flann.knnMatch(des1, des2, k=2)  # 得出匹配的关键点  list: 1136
    good = []
    # 提取优秀的特征点
    for m, n in matches:
        if m.distance < 0.7 * n.distance:  # 如果第一个邻近距离比第二个邻近距离的0.7倍小，则保留
            good.append(m)   # 134
    src_pts = np.array([kp1[m.queryIdx].pt for m in good])  # 查询图像的特征描述子索引  # 134, 2
    dst_pts = np.array([kp2[m.trainIdx].pt for m in good])  # 训练(模板)图像的特征描述子索引
    H = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)  # 生成变换矩阵  H[0]: 3, 3  H[1]: 134, 1
    return H[0]

# YUV通道小波变换RGB融合，低频均值高频最大值
def Images_fusion(img_base,img_target):          # 可见光，配准后的红外光
    # 按照不同的融合方式，计算通道融合
    def fuseCoeff(cooef1, cooef2, method):
        if (method == 'mean'):
            cooef = (abs(cooef1) + abs(cooef2)) / 2  # abs 绝对值
            # cooef=0.5*cooef1+1.5*cooef2
        elif (method == 'min'):
            cooef = np.minimum(cooef1, cooef2)
        elif (method == 'max'):
            cooef = np.maximum(abs(cooef1), abs(cooef2))
        return cooef
    # HSV的小波融合过程
    LOW_METHOD = 'mean'
    HIGH_METHOD = 'max'
    YUVimg_base = cv2.cvtColor(img_base, cv2.COLOR_BGR2YUV)  # 可见光转换为HSV
    # cv2.imwrite("D:/VS/vsprj/cuda/cudawavetest/2.jpg", YUVimg_base)
    grayimg_target = cv2.cvtColor(img_target, cv2.COLOR_BGR2GRAY)  #红外灰度化
    # cv2.imwrite("D:/VS/vsprj/cuda/cudawavetest/1.jpg",grayimg_target)
    Yimg_base = YUVimg_base[:, :, 0]  # 1024，1024
    wavelet = 'haar'
    cooef_base = pywt.wavedec2(Yimg_base[:, :], wavelet, level=1)  # base灰度图的小波展开  512, 512    3
    start = time.time()
    cooef_target = pywt.wavedec2(grayimg_target[:, :], wavelet, level=1)  # target灰度图的小波展开   512, 512    3
    end = time.time()-start
    print("小波展开:{}".format(end))
    fusedCooef = []
    for i in range(len(cooef_base)):
        if (i == 0):
            fusedCooef.append(fuseCoeff(cooef_base[0], cooef_target[0], LOW_METHOD))  # 低频部分取均值
        else:
            # 高频部分取最大值
            c1 = fuseCoeff(cooef_base[i][0], cooef_target[i][0], HIGH_METHOD)
            c2 = fuseCoeff(cooef_base[i][1], cooef_target[i][1], HIGH_METHOD)
            c3 = fuseCoeff(cooef_base[i][2], cooef_target[i][2], HIGH_METHOD)
            fusedCooef.append((c1, c2, c3))  # 高频合并
    tempfusedImage = pywt.waverec2(fusedCooef, wavelet)  # 小波逆变换
    fusedImage = np.multiply(np.divide(tempfusedImage - np.min(tempfusedImage), (np.max(tempfusedImage) -\
                            np.min(tempfusedImage))),255)  # 逆变换后归一至（0，255）
    start = time.time()
    fusedImage = fusedImage.astype(np.uint8)
    Yimg_new = fusedImage
    fusedYUV = cv2.merge([Yimg_new,YUVimg_base[:, :, 1], YUVimg_base[:, :, 2]])  # 用小波变换替换V通道
    end = time.time() - start
    print("图像重建:{}".format(end))
    fusedBGR = cv2.cvtColor(fusedYUV, cv2.COLOR_YUV2BGR)  # 融合后的HSV转为BGR
    return fusedBGR
#
def main():
    matchimg_di = cv2.imread('images/oripics/basic_day_infrared.jpg')  # 1080, 1920, 3
    matchimg_dv = cv2.imread('images/oripics/basic_day_visual.jpg')
    # matchimg_di = cv2.imread('left_2020_11_30-15_15_31.jpg')  # 1080, 1920, 3
    # matchimg_dv = cv2.imread('right_2020_11_30-15_15_31.jpg')
    # 1080, 1920, 3
    orimg_nv=matchimg_dv
    orimg_ni=matchimg_di
    # orimg_nv = cv2.imread('images/oripics/night_visual_40m.jpg')
    # orimg_ni = cv2.imread('images/oripics/night_infrared_40m.jpg')
    # orimg_nv = cv2.imread('images/oripics/left_2020_11_30-20_06_54.jpg')
    # orimg_ni = cv2.imread('images/oripics/right_2020_11_30-20_06_54.jpg')
    # orimg_nv = cv2.imread('images/oripics/left_2020_11_30-20_08_09.jpg')
    # orimg_ni = cv2.imread('images/oripics/right_2020_11_30-20_08_09.jpg')
    orimg_nv = cv2.imread('images/oripics/left_video_2020_11_30-20_09_32_73.jpg')  # 1024, 1024, 3
    orimg_ni = cv2.imread('images/oripics/right_video_2020_11_30-20_09_32_73.jpg')  # 1024, 1024, 3

    #用白天图像进行配准
    # enhance_matchimg_di = truncated_linear_stretch(matchimg_di)# 配准模板红外图像RGB增强
    h, w = orimg_nv.shape[:2]   # 1024 1024
    H = Images_matching(matchimg_dv, matchimg_di)   # (3, 3)

    # enhance_orimg_ni=truncated_linear_stretch(orimg_ni) # 需融合红外图像RGB增强
    matched_ni = cv2.warpPerspective(orimg_ni, H, (w, h))
    cv2.imwrite("./1.jpg",matched_ni)# 红外图像按照仿射矩阵配准 1024, 1024, 3
    start = time.time()
    fusion = Images_fusion(orimg_nv, matched_ni)
    cv2.imwrite("./2.jpg",fusion)
    end = time.time()-start
    print(end)
    # enhance=truncated_linear_stretch(fusion)#融合图像RGB增强

    # cv2.imshow('0', cv2.resize(orimg_nv, (600, 400)))
    # cv2.imshow('1', cv2.resize(orimg_ni, (600, 400)))
    cv2.imshow('2', cv2.resize(fusion, (1200, 800)))
    cv2.waitKey(0)

    # name_fusionfile='YUV_04.jpg'
    # path_fusionfile='images/fusion/'+name_fusionfile
    # cv2.imwrite(path_fusionfile, fusion)

    video_path_infrared = "../videos/ir/video_2020_11_30-20_05_30.avi"
    video_path_visible = "../videos/vi/video_2020_11_30-20_05_30.avi"
    video_save_path = "../videos/out.avi"
    video_fps = 25

    # matchimg_di = cv2.imread('images/oripics/basic_day_infrared.jpg')  # 1080, 1920, 3
    # matchimg_dv = cv2.imread('images/oripics/basic_day_visual.jpg')
    matchimg_di = cv2.imread('left_2020_11_30-15_15_31.jpg')  # 1080, 1920, 3
    matchimg_dv = cv2.imread('right_2020_11_30-15_15_31.jpg')
    H = Images_matching(matchimg_dv, matchimg_di)

    capture_in = cv2.VideoCapture(video_path_infrared)
    capture_vi = cv2.VideoCapture(video_path_visible)
    if video_save_path != "":
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        size = (int(capture_vi.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture_vi.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

    fps = 0.0
    while (True):
        t1 = time.time()
        # 读取某一帧
        ref1, frame1 = capture_in.read()
        ref2, frame2 = capture_vi.read()
        # 格式转变，BGRtoRGB
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        # 转变成Image
        # frame1 = Image.fromarray(np.uint8(frame1))
        # frame2 = Image.fromarray(np.uint8(frame2))
        h, w = frame1.shape[:2]
        matched_ni = cv2.warpPerspective(frame2, H, (w, h))
        cv2.imwrite("./1.jpg", matched_ni)
        # 进行检测
        fusion = Images_fusion(frame1, matched_ni)
        fusion = np.array(fusion)
        end = time.time() - t1
        # RGBtoBGR满足opencv显示格式
        frame = cv2.cvtColor(fusion, cv2.COLOR_RGB2BGR)

        fps = (fps + (1. / end )) / 2
        print("fps= %.2f" % (fps))
        frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("video", frame)
        c = cv2.waitKey(1) & 0xff
        if video_save_path != "":
            out.write(frame)

        if c == 27:
            capture.release()
            break
    capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
