#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from email import message
import time
import os
import streamlit as st
import pandas as pd

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    flash,
    redirect,
    url_for,
    send_file,
)
from flask_restful import Api, Resource

from application.api.utils import run_predict
from application import config


app = Flask(
    __name__,
    static_url_path="/Hello/static",
    template_folder="/Users/nguyenthinhquyen/source/ml_ecosystem/application/api/static",
)
api = Api(app)


app.secret_key = "secret key"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.config["JSON_AS_ASCII"] = False

ALLOWED_EXTENSIONS = set(["txt"])


class Stat(Resource):
    def get(self):
        return dict(error=0, message="server start")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/hello/predict", methods=["GET", "POST"])
def predict():
    start_time = time.time()
    if request.method.lower() == "post":
        # try:
        print("hello")
        if "file" not in request.files:
            return jsonify(dict(error=1, message="Data invaild"))
        file = request.files["file"]
        if file.filename == "":
            return jsonify(dict(error=1, message="Data invaild"))
        if file and allowed_file(file.filename):
            # read file here
            file_contents = file.readlines()
            avg_loss, image_name = run_predict(file_contents)
            flash(f"Average loss: {avg_loss}")
            return render_template("upload.html", filename=image_name)
    # except:
    #     return jsonify(dict(error=1, message="Something Error"))
    return render_template("upload.html")


# @app.route("/hello/streamlit")
# def streamlit():
#     st.set_page_config(page_title="My Streamlit App")
#     st.write("Hello, world!")
#     return dict(error=0, message="hello test")


def main():
    api.add_resource(Stat, "/")
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=3050, help="port(default: 3050)")
    args = parser.parse_args()
    port = int(args.port)
    app.debug = True
    app.run("0.0.0.0", port=port, threaded=True)


if __name__ == "__main__":
    main()
