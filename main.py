import glob

import cv2
import numpy as np

from calibration import get_camera_matrix
from display import Display
from extractor import Frame, add_ones, denormalize, match_frames
from pointmap import Map, Point

# Camera intrinsics
W, H = 1280, 720
K, dist_coeffs = get_camera_matrix()


Kinv = np.linalg.inv(K)

# display = Display(1920, 1080)
mapp = Map()
mapp.create_viewer()


def triangulate(pose1, pose2, pts1, pts2):
    ret = np.zeros((pts1.shape[0], 4))
    pose1 = np.linalg.inv(pose1)
    pose2 = np.linalg.inv(pose2)
    for i, p in enumerate(zip(add_ones(pts1), add_ones(pts2))):
        A = np.zeros((4, 4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]

    return ret


def process_frame(img):

    img = cv2.resize(img, (W, H))
    frame = Frame(mapp, img, K)
    if frame.id == 0:
        return

    # previous frame f2 to the current frame f1.
    f1 = mapp.frames[-1]
    f2 = mapp.frames[-2]

    idx1, idx2, Rt = match_frames(f1, f2)
    # print(f"=------------Rt {Rt}")
    # f2.pose represents the transformation from the world coordinate system to the coordinate system of the previous frame f2.
    # Rt represents the transformation from the coordinate system of f2 to the coordinate system of f1.
    # By multiplying Rt with f2.pose, you get a new transformation that directly maps the world coordinate system to the coordinate system of f1.
    f1.pose = np.dot(Rt, f2.pose)

    # The output is a matrix where each row is a 3D point in homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊]
    pts4d = triangulate(f1.pose, f2.pose, f1.pts[idx1], f2.pts[idx2])

    # This line normalizes the 3D points by dividing each row by its fourth coordinate W
    # The homogeneous coordinates [𝑋, 𝑌, 𝑍, 𝑊] are converted to Euclidean coordinates
    pts4d /= pts4d[:, 3:]

    # Reject points without enough "Parallax" and points behind the camera
    # checks if the absolute value of the fourth coordinate W is greater than 0.005.
    # checks if the z-coordinate of the points is positive.
    # returns, A boolean array indicating which points satisfy both criteria.
    good_pts4d = (np.abs(pts4d[:, 3]) > 0.005) & (pts4d[:, 2] > 0)

    for i, p in enumerate(pts4d):
        #  If the point is not good (i.e., good_pts4d[i] is False), the loop skips the current iteration and moves to the next point.
        if not good_pts4d[i]:
            continue
        pt = Point(mapp, p)
        pt.add_observation(f1, i)
        pt.add_observation(f2, i)

    for pt1, pt2 in zip(f1.pts[idx1], f2.pts[idx2]):
        u1, v1 = denormalize(K, pt1)
        u2, v2 = denormalize(K, pt2)

        # cv2.circle(img, (u1,v1), 3, (0,255,0), 2)
        cv2.circle(img, (u1, v1), 2, (77, 243, 255))

        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))
        cv2.circle(img, (u2, v2), 2, (204, 77, 255))

    # 2-D display
    # img = cv2.resize(img, ( 320, 180))
    # display.paint(img)

    # 3-D display
    mapp.display()
    mapp.display_image(img)


if __name__ == "__main__":
    # Используем вебкамеру вместо видеофайла
    cap = cv2.VideoCapture(0)  # 0 - индекс первой доступной камеры

    # Установка разрешения камеры (опционально)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while cap.isOpened():
        ret, frame = cap.read()
        # print("\n#################  [NEW FRAME]  #################\n")
        if ret == True:
            process_frame(frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
