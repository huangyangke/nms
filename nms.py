import numpy as np

def py_nms(dets, sc, thresh):
    """Pure Python NMS baseline."""
    # x1、y1、x2、y2、以及score赋值
    # （x1、y1）（x2、y2）为box的左上和右下角标
    x1 = dets[:, 0]
    y1 = dets[:, 1]#左上角的坐标值
 
    x2 = dets[:, 2]
    y2 = dets[:, 3]#右下角的阈值
 
    scores = sc
    # 每一个候选框的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # order是按照score降序排序的，从大到小
    order = scores.argsort()[::-1]
 
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # 计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # 计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        # 计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        # 找到重叠度不高于阈值的矩形框索引
        inds = np.where(ovr <= thresh)[0]
        # print("inds:",inds)
        # 将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
        order = order[inds + 1]
    return keep

def calculate_iou(a, b):
    """
    a [num1, 4]
    b [num2, 4]
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2]) # 左上
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:]) # 右下

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)
    
def py_cpu_softnms(dets, scores, iou_threshold=0.3, agnostic=False, sigma=0.5, soft_threshold=0.001, method=2):
    """
    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [x1, y1, x2, y2, score]
    :param scores:     每个 boxes 对应的分数
    :param iou_threshold:     iou 交叠门限
    :param agnostic: 进行nms是否也去除不同类别之间的框 默认False
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """
    dets = dets.copy()
    scores = scores.copy()
    
    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    # 不同类别加上偏移量 这样计算iou的时候不同类别的框永远为0
    classes_shift = dets[:, 4:5] * (0 if agnostic else max_wh)  
    dets += classes_shift

    # indexes concatenate boxes with the last column
    N = dets.shape[0] # 输入box数量
    indexes = np.array([np.arange(N)])
    dets = np.concatenate((dets, indexes.T), axis=1) # (N, 4 + 1 + 1)
    indexes_dim = dets.shape[1] - 1
    
    # the order of boxes coordinate is [x1, y1, x2, y2]
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        tscore = scores[i].copy()
        tarea = areas[i].copy()
        pos = i + 1 # 当前box后面的box开始索引

        if i != N-1:
            maxscore = np.max(scores[pos:], axis=0)
            maxpos = np.argmax(scores[pos:], axis=0)
        else:
            maxscore = scores[-1]
            maxpos = 0
        if tscore < maxscore:
            # 执行位置交换 将最大的换到前面来 注意这里的i + 1就是pos相当于偏移量
            dets[i, :] = dets[maxpos + i + 1, :]
            dets[maxpos + i + 1, :] = tBD
            tBD = dets[i, :]

            scores[i] = scores[maxpos + i + 1]
            scores[maxpos + i + 1] = tscore
            tscore = scores[i]

            areas[i] = areas[maxpos + i + 1]
            areas[maxpos + i + 1] = tarea
            tarea = areas[i]

        # 注意此时的dets[i, 1]是i~N-1中score最大的box了
        # IoU calculate
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[pos:] - inter)

        # 依据ovr来进行分数惩罚
        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear 线形抑制
            weight = np.ones(ovr.shape)
            weight[ovr > iou_threshold] = weight[ovr > iou_threshold] - ovr[ovr > iou_threshold]
        elif method == 2:  # gaussian 高斯抑制
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS 和原始nms一致
            weight = np.ones(ovr.shape)
            weight[ovr > iou_threshold] = 0

        scores[pos:] = weight * scores[pos:]

    # select the boxes and keep the corresponding indexes
    inds = dets[:, indexes_dim][scores > soft_threshold]
    keep = inds.astype(int)

    return keep


if __name__ == '__main__':
    boxes = np.array([[200, 200, 400, 400], [220, 220, 420, 420], 
                      [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
    boxscores = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    index = py_cpu_softnms(boxes, boxscores, method=3)
    print(index)
    
    import torchvision
    import torch
    index = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(boxscores), 0.3)
    print(index)
