import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


def add_ones(x):
    # concatenates the original array x with the column of ones along the second axis (columns).
    # This converts the N×2 array to an N×3 array where each point is represented
    # in homogeneous coordinates as [x,y,1].
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


IRt = np.eye(4)


def extractPose(F):
    # W = np.mat([[0,-1,0],[1,0,0],[0,0,1]])
    W = np.asmatrix([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    U, d, Vt = np.linalg.svd(F)
    assert np.linalg.det(U) > 0 # если < 0 кадр/пара кадр–кадр имеет крайне плохое качество, отражение или вылет через численную нестабильность
    if np.linalg.det(Vt) < 0:
        Vt *= -1
    R = np.dot(np.dot(U, W), Vt)
    if np.sum(R.diagonal()) < 0:
        R = np.dot(np.dot(U, W.T), Vt)
    t = U[:, 2]
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    # print(d)
    return ret


def extract(img):
    orb = cv2.ORB_create()

    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detection
    pts = cv2.goodFeaturesToTrack(gray_img, 8000, qualityLevel=0.01, minDistance=10)

    if pts is None:
        return np.array([]), None

    # Extraction
    kps = [cv2.KeyPoint(f[0][0], f[0][1], 20) for f in pts]
    kps, des = orb.compute(gray_img, kps)

    return np.array([(kp.pt[0], kp.pt[1]) for kp in kps]), des


def normalize(Kinv, pts):
    # The inverse camera intrinsic matrix 𝐾 − 1 transforms 2D homogeneous points
    # from pixel coordinates to normalized image coordinates. This transformation centers
    # the points based on the principal point (𝑐𝑥 , 𝑐𝑦) and scales them
    # according to the focal lengths 𝑓𝑥 and 𝑓𝑦, effectively mapping the points
    # to a normalized coordinate system where the principal point becomes the origin and
    # the distances are scaled by the focal lengths.
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]
    # `[:, 0:2]` selects the first two columns of the resulting array, which are the normalized x and y coordinates.
    # `.T` transposes the result back to N x 3.


def denormalize(K, pt):
    ret = np.dot(K, [pt[0], pt[1], 1.0])
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))


class Matcher(object):
    def __init__(self):
        self.last = None

def match_frames(f1, f2):
    """
    На вход: два объекта Frame с полями:
      - f1.des, f2.des — дескрипторы (N×32)
      - f1.pts, f2.pts — нормализованные точки (N×2)
    На выход: idx1, idx2 — индексы inliers в f1.pts и f2.pts, и 4×4 Rt-матрица.
    Если совпадений < 8 или не найдено inliers, возвращает (None, None, None).
    """

    # 1) BFMatcher + Lowe’s ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)
    idx1, idx2 = [], []
    for m, n in matches:
        # Lowe’s ratio: оставляем только уникальные пары :contentReference[oaicite:2]{index=2}
        if m.distance < 0.75 * n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]
            # дополнительный фильтр по смещению
            if np.linalg.norm(p1 - p2) < 0.1:
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)

    idx1 = np.array(idx1, dtype=int)
    idx2 = np.array(idx2, dtype=int)

    # 2) Если слишком мало точек — пропускаем кадр
    if len(idx1) < 8:
        return None, None, None

    # 3) Составляем массивы для findFundamentalMat
    pts1 = f1.pts[idx1]
    pts2 = f2.pts[idx2]

    # 4) Оцениваем фундаментальную матрицу RANSACʼом
    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=0.005,
        confidence=0.99,
        maxIters=200
    )  # :contentReference[oaicite:3]{index=3}

    if F is None or mask is None:
        return None, None, None

    # 5) Выбираем только истинные inliers
    mask = mask.ravel().astype(bool)
    idx1_in = idx1[mask]
    idx2_in = idx2[mask]

    if len(idx1_in) < 8:
        # даже после RANSAC осталось слишком мало точек
        return None, None, None

    # 6) Извлекаем 4×4 Rt-матрицу из F
    Rt = extractPose(F)

    return idx1_in, idx2_in, Rt

def match_frames_old(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(f1.des, f2.des, k=2)

    # Lowe's ratio test
    ret = []
    idx1, idx2 = [], []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            p1 = f1.pts[m.queryIdx]
            p2 = f2.pts[m.trainIdx]

            # Distance test
            # dditional distance test, ensuring that the
            # Euclidean distance between p1 and p2 is less than 0.1
            if np.linalg.norm((p1 - p2)) < 0.1:
                # Keep idxs
                idx1.append(m.queryIdx)
                idx2.append(m.trainIdx)
                ret.append((p1, p2))
                pass

    # # Если соответствует меньше 8, алгоритм не может однозначно вычислить фундаментальную матрицу
    # assert len(ret) >= 8

    if len(ret) < 8:
        # недостаточно точек — пропускаем этот кадр
        return None, None, None

    ret = np.array(ret)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)

    # Fit matrix
    model, inliers = ransac(
        (ret[:, 0], ret[:, 1]),
        FundamentalMatrixTransform,
        min_samples=8,
        residual_threshold=0.005,
        max_trials=200,
    )

    # Ignore outliers
    ret = ret[inliers]
    Rt = extractPose(model.params)

    return idx1[inliers], idx2[inliers], Rt


class Frame(object):
    def __init__(self, mapp, img, K):
        self.K = K
        self.Kinv = np.linalg.inv(self.K)
        self.pose = IRt

        self.id = len(mapp.frames)
        mapp.frames.append(self)

        pts, self.des = extract(img)

        if self.des.any() != None:
            self.pts = normalize(self.Kinv, pts)
