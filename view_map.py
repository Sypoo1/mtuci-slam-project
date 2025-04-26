import OpenGL.GL as gl
import pangolin

from pointmap import Map


def run_viewer(poses, points):
    # Настройка окна Pangolin (аналогично вашему Map.viewer_init)
    pangolin.CreateWindowAndBind("MapViewer", 1024, 768)
    gl.glEnable(gl.GL_DEPTH_TEST)
    scam = pangolin.OpenGlRenderState(
        pangolin.ProjectionMatrix(1024, 768, 420, 420, 512, 384, 0.2, 10000),
        pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0),
    )
    dcam = pangolin.CreateDisplay()
    dcam.SetBounds(0.0, 1.0, 0.0, 1.0, -1024 / 768)
    handler = pangolin.Handler3D(scam)
    dcam.SetHandler(handler)

    while not pangolin.ShouldQuit():
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        dcam.Activate(scam)
        gl.glLineWidth(1)
        gl.glColor3f(0.0, 1.0, 0.0)
        pangolin.DrawCameras(poses)
        gl.glPointSize(2)
        gl.glColor3f(1.0, 0.0, 0.0)
        pangolin.DrawPoints(points)
        pangolin.FinishFrame()


if __name__ == "__main__":
    poses, points = Map.load("map.npz")
    run_viewer(poses, points)
