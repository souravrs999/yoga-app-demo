# env/usr/bin python

""" Necessary Imports """
from .utils.utils import capture_data
from flask import Blueprint, render_template, Response, request, redirect, url_for
from flask_login import login_required, current_user

""" Initialize blueprint """
main = Blueprint("main", __name__)

""" Index route """


@main.route("/")
def index():

    """ Render the index webpage """
    return render_template("index.html")


""" User profile webpages """


@main.route("/profile")
def profile():

    """ Returns profile webpage for the current user """
    return render_template("profile.html", name=current_user.name)


@main.route("/stream", methods=["GET", "POST"])
def stream():

    """ return the streaming template """
    return render_template("video-feed.html")


@main.route("/pose/<pose_name>")
def pose(pose_name):

    if pose_name == 'cobra':
        return render_template('cobra.html')
    elif pose_name == 'mountain':
        return render_template('mountain.html')

@main.route('/pose/live/<pose_name>')
def live(pose_name):

    """ returns the encoded frame """
    frame_data = capture_data(pose_name=pose_name,src=0)

    return Response(frame_data, mimetype="multipart/x-mixed-replace; boundary=frame")
