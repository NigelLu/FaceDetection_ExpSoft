import base64
from container import Container
from flask import Flask, render_template, request, redirect, url_for


def create_app():
    container = Container()

    app = Flask(__name__)
    app.container = container

    return app

app = create_app()

# app.config["SECRET_KEY"] = 'password'
# app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///tmp/test.db'
# db = SQLAlchemy(app)

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    ...


if __name__ == '__main__':
    app.run(debug=True)
