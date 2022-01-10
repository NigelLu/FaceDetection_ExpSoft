import io
import base64
import numpy as np
from PIL import Image
from detector import Detector
# from container import Container
from flask import Flask, render_template, request, redirect, url_for


def create_app():

    app = Flask(__name__)
    detector = Detector()

    # app.config["SECRET_KEY"] = 'password'
    # app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///tmp/test.db'
    # db = SQLAlchemy(app)

    @app.route("/", methods=["GET"])
    def home():
        return render_template("home.html")

    @app.route("/liveness", methods=["POST"])
    def liveness():

        # try decode base64 image and obtain a numpy array object
        try:
            img_str = request.json["test_img"]
            img = Image.open(io.BytesIO(base64.b64decode(img_str)))
            img = np.array(img)
        except:
            return "图片解码失败"

        # try liveness detection, default confidence level as 0.8
        try:
            if detector.liveness(test_img=img, confidence_threshold=0.8):
                return "活体检测通过"
            else:
                return "活体检测未通过"
        except:
            return "api错误，活体检测失败"
        
    @app.route("/match", methods=["POST"])
    def match():
        # try decode base64 image and obtain a numpy array object
        try:
            test_img_str = request.json["test_img"]
            test_img = Image.open(io.BytesIO(base64.b64decode(test_img_str)))
            test_img = np.array(test_img)

            true_img_str = request.json["true_img"]
            true_img = Image.open(io.BytesIO(base64.b64decode(true_img_str)))
            true_img = np.array(true_img)
        except:
            return "图片解码失败"

        # try match the true image from database with the test image
        try:
            if detector.match(true_img, test_img):
                return "人脸特征吻合，匹配通过"
            else:
                return "人脸特征不吻合，匹配未通过"
        except:
            return "api错误，人脸匹配失败"

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
