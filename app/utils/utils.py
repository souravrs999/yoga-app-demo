#! env/usr/bin python

""" Necessary Imports """
import sys
import os
import cv2
import numpy as np
import datetime
import mediapipe as mp

pose_dir = "data"


def draw_joints(img, landmark_map, W, H):

    for joint in landmark_map:
        cv2.circle(
            img,
            (
                int(landmark_map[joint]["x"]),
                int(landmark_map[joint]["y"]),
            ),
            3,
            (255, 255, 255),
            2,
        )


def draw_bones(img, landmark_map, W, H):

    left_shoulder_elbow = cv2.line(
        img,
        (
            int(landmark_map["left_shoulder"]["x"]),
            int(landmark_map["left_shoulder"]["y"]),
        ),
        (
            int(landmark_map["left_elbow"]["x"]),
            int(landmark_map["left_elbow"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    left_elbow_wrist = cv2.line(
        img,
        (
            int(landmark_map["left_elbow"]["x"]),
            int(landmark_map["left_elbow"]["y"]),
        ),
        (
            int(landmark_map["left_wrist"]["x"]),
            int(landmark_map["left_wrist"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    right_shoulder_elbow = cv2.line(
        img,
        (
            int(landmark_map["right_shoulder"]["x"]),
            int(landmark_map["right_shoulder"]["y"]),
        ),
        (
            int(landmark_map["right_elbow"]["x"]),
            int(landmark_map["right_elbow"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    right_elbow_wrist = cv2.line(
        img,
        (
            int(landmark_map["right_elbow"]["x"]),
            int(landmark_map["right_elbow"]["y"]),
        ),
        (
            int(landmark_map["right_wrist"]["x"]),
            int(landmark_map["right_wrist"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    shoulder = cv2.line(
        img,
        (
            int(landmark_map["right_shoulder"]["x"]),
            int(landmark_map["right_shoulder"]["y"]),
        ),
        (
            int(landmark_map["left_shoulder"]["x"]),
            int(landmark_map["left_shoulder"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    waist = cv2.line(
        img,
        (
            int(landmark_map["right_hip"]["x"]),
            int(landmark_map["right_hip"]["y"]),
        ),
        (
            int(landmark_map["left_hip"]["x"]),
            int(landmark_map["left_hip"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    right_shoulder_waist = cv2.line(
        img,
        (
            int(landmark_map["right_shoulder"]["x"]),
            int(landmark_map["right_shoulder"]["y"]),
        ),
        (
            int(landmark_map["right_hip"]["x"]),
            int(landmark_map["right_hip"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    left_shoulder_waist = cv2.line(
        img,
        (
            int(landmark_map["left_shoulder"]["x"]),
            int(landmark_map["left_shoulder"]["y"]),
        ),
        (
            int(landmark_map["left_hip"]["x"]),
            int(landmark_map["left_hip"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    left_hip_knee = cv2.line(
        img,
        (
            int(landmark_map["left_hip"]["x"]),
            int(landmark_map["left_hip"]["y"]),
        ),
        (
            int(landmark_map["left_knee"]["x"]),
            int(landmark_map["left_knee"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    left_knee_ankle = cv2.line(
        img,
        (
            int(landmark_map["left_knee"]["x"]),
            int(landmark_map["left_knee"]["y"]),
        ),
        (
            int(landmark_map["left_ankle"]["x"]),
            int(landmark_map["left_ankle"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    right_hip_knee = cv2.line(
        img,
        (
            int(landmark_map["right_hip"]["x"]),
            int(landmark_map["right_hip"]["y"]),
        ),
        (
            int(landmark_map["right_knee"]["x"]),
            int(landmark_map["right_knee"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    right_knee_ankle = cv2.line(
        img,
        (
            int(landmark_map["right_knee"]["x"]),
            int(landmark_map["right_knee"]["y"]),
        ),
        (
            int(landmark_map["right_ankle"]["x"]),
            int(landmark_map["right_ankle"]["y"]),
        ),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def compute_angles(img, pose_name, landmark_map, results, control_results, W, H):
    def angle(coords):

        import math

        x1, x2, y1, y2 = coords
        return int(math.atan((y2 - y1) / (x2 - x1)) * 180 / math.pi)

    def switch(pose_name):
        def put_text(img, msg):

            cv2.putText(
                img,
                msg,
                (int(W * 0.1), int(H * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            def put_line(color, coords):

                x1, x2, y1, y2 = coords

                if str(color) == "green":
                    color = (0, 255, 0)
                elif str(color) == "red":
                    color = (0, 0, 255)
                else:
                    color = (255, 255, 255)

                cv2.line(
                    img,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    2,
                    cv2.LINE_AA,
                )

        pose = str(pose_name).lower()

        left_leg_ankle = angle(
            (
                landmark_map["left_hip"]["x"],
                landmark_map["left_ankle"]["x"],
                landmark_map["left_hip"]["y"],
                landmark_map["left_ankle"]["y"],
            )
        )
        right_leg_ankle = angle(
            (
                landmark_map["right_hip"]["x"],
                landmark_map["right_ankle"]["x"],
                landmark_map["right_hip"]["y"],
                landmark_map["right_ankle"]["y"],
            )
        )
        left_arm_ankle = angle(
            (
                landmark_map["left_shoulder"]["x"],
                landmark_map["left_wrist"]["x"],
                landmark_map["left_shoulder"]["y"],
                landmark_map["left_wrist"]["y"],
            )
        )
        right_arm_ankle = angle(
            (
                landmark_map["right_shoulder"]["x"],
                landmark_map["right_wrist"]["x"],
                landmark_map["right_shoulder"]["y"],
                landmark_map["right_wrist"]["y"],
            )
        )
        left_shoulder_ankle = angle(
            (
                landmark_map["left_shoulder"]["x"],
                landmark_map["left_ankle"]["x"],
                landmark_map["left_shoulder"]["y"],
                landmark_map["left_ankle"]["y"],
            )
        )

        def mountain():
            def dist(a, b):

                from scipy.spatial import distance as dist

                return dist.euclidean(a, b)

            left_hip_wrist = dist(
                (landmark_map["left_hip"]["x"], landmark_map["left_hip"]["y"]),
                (landmark_map["left_wrist"]["x"], landmark_map["left_wrist"]["y"]),
            )
            right_hip_wrist = dist(
                (landmark_map["right_hip"]["x"], landmark_map["right_hip"]["y"]),
                (landmark_map["right_wrist"]["x"], landmark_map["right_wrist"]["y"]),
            )
            left_right_knee = dist(
                (landmark_map["left_knee"]["x"], landmark_map["left_knee"]["y"]),
                (landmark_map["right_knee"]["x"], landmark_map["right_knee"]["y"]),
            )

            if left_right_knee >= 50 and left_right_knee <= 70:

                left_hip_knee = cv2.line(
                    img,
                    (
                        int(landmark_map["left_hip"]["x"]),
                        int(landmark_map["left_hip"]["y"]),
                    ),
                    (
                        int(landmark_map["left_knee"]["x"]),
                        int(landmark_map["left_knee"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                left_knee_ankle = cv2.line(
                    img,
                    (
                        int(landmark_map["left_knee"]["x"]),
                        int(landmark_map["left_knee"]["y"]),
                    ),
                    (
                        int(landmark_map["left_ankle"]["x"]),
                        int(landmark_map["left_ankle"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                right_hip_knee = cv2.line(
                    img,
                    (
                        int(landmark_map["right_hip"]["x"]),
                        int(landmark_map["right_hip"]["y"]),
                    ),
                    (
                        int(landmark_map["right_knee"]["x"]),
                        int(landmark_map["right_knee"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                right_knee_ankle = cv2.line(
                    img,
                    (
                        int(landmark_map["right_knee"]["x"]),
                        int(landmark_map["right_knee"]["y"]),
                    ),
                    (
                        int(landmark_map["right_ankle"]["x"]),
                        int(landmark_map["right_ankle"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if (
                left_hip_wrist >= 50
                and left_hip_wrist <= 90
                and right_hip_wrist >= 50
                and right_hip_wrist <= 90
            ):

                left_shoulder_elbow = cv2.line(
                    img,
                    (
                        int(landmark_map["left_shoulder"]["x"]),
                        int(landmark_map["left_shoulder"]["y"]),
                    ),
                    (
                        int(landmark_map["left_elbow"]["x"]),
                        int(landmark_map["left_elbow"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                left_elbow_wrist = cv2.line(
                    img,
                    (
                        int(landmark_map["left_elbow"]["x"]),
                        int(landmark_map["left_elbow"]["y"]),
                    ),
                    (
                        int(landmark_map["left_wrist"]["x"]),
                        int(landmark_map["left_wrist"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                right_shoulder_elbow = cv2.line(
                    img,
                    (
                        int(landmark_map["right_shoulder"]["x"]),
                        int(landmark_map["right_shoulder"]["y"]),
                    ),
                    (
                        int(landmark_map["right_elbow"]["x"]),
                        int(landmark_map["right_elbow"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                right_elbow_wrist = cv2.line(
                    img,
                    (
                        int(landmark_map["right_elbow"]["x"]),
                        int(landmark_map["right_elbow"]["y"]),
                    ),
                    (
                        int(landmark_map["right_wrist"]["x"]),
                        int(landmark_map["right_wrist"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        def cobra():

            if left_shoulder_ankle >= 20 and left_shoulder_ankle <= 30:
                # print(
                #     f"left_angle: {left_shoulder_ankle}",
                #     file=sys.stderr,
                # )
                left_shoulder_hip = cv2.line(
                    img,
                    (
                        int(landmark_map["left_shoulder"]["x"]),
                        int(landmark_map["left_shoulder"]["y"]),
                    ),
                    (
                        int(landmark_map["left_hip"]["x"]),
                        int(landmark_map["left_hip"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                left_hip_knee = cv2.line(
                    img,
                    (
                        int(landmark_map["left_hip"]["x"]),
                        int(landmark_map["left_hip"]["y"]),
                    ),
                    (
                        int(landmark_map["left_knee"]["x"]),
                        int(landmark_map["left_knee"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                right_shoulder_hip = cv2.line(
                    img,
                    (
                        int(landmark_map["right_shoulder"]["x"]),
                        int(landmark_map["right_shoulder"]["y"]),
                    ),
                    (
                        int(landmark_map["right_hip"]["x"]),
                        int(landmark_map["right_hip"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                left_right_knee = cv2.line(
                    img,
                    (
                        int(landmark_map["right_hip"]["x"]),
                        int(landmark_map["right_hip"]["y"]),
                    ),
                    (
                        int(landmark_map["right_knee"]["x"]),
                        int(landmark_map["right_knee"]["y"]),
                    ),
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        def default():
            pass

        dict = {"mountain": mountain, "cobra": cobra}
        dict.get(pose_name, default)()

    switch(pose_name)


def map_landmarks(img, results, W, H, mp_pose, pose_name, control_results):

    landmark_map = {
        "left_shoulder": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            * H,
        },
        "right_shoulder": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            * H,
        },
        "left_elbow": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * H,
        },
        "right_elbow": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
            * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
            * H,
        },
        "left_wrist": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * H,
        },
        "right_wrist": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
            * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            * H,
        },
        "left_hip": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * H,
        },
        "right_hip": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * H,
        },
        "left_knee": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * H,
        },
        "right_knee": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * H,
        },
        "left_ankle": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * H,
        },
        "right_ankle": {
            "x": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x
            * W,
            "y": results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
            * H,
        },
    }

    draw_joints(img, landmark_map, W, H)
    draw_bones(img, landmark_map, W, H)
    compute_angles(img, pose_name, landmark_map, results, control_results, W, H)


def construct_control(pose_name):

    pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    for image in os.listdir(pose_dir):
        if image.split("-")[1] == str(pose_name):
            img = cv2.imread(os.path.join(pose_dir, image))
            control_result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            pose.close()

            return control_result


""" A function to capture datat from webcam """


def capture_data(pose_name, src=0):

    """ Grab the frame from the camera """
    cap = cv2.VideoCapture(src)
    _, frame = cap.read()
    control_result = construct_control(pose_name)
    out = cv2.VideoWriter(
        "output.avi",
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        30,
        (frame.shape[1], frame.shape[0]),
    )
    mp_pose = mp.solutions.pose
    pose = mp.solutions.pose.Pose(
        upper_body_only=False, smooth_landmarks=True, min_detection_confidence=0.5
    )

    while cap.isOpened():

        """ Grab frame """
        ret, frame = cap.read()

        """ If it returned true """
        if not ret:
            break
        H, W, _ = frame.shape
        frame.flags.writeable = False
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        map_landmarks(frame, results, W, H, mp_pose, pose_name, control_result)
        # frame.flags.writeable = True
        out.write(frame)
        # cv2.imshow("image", frame)

        """ if the img count recieved our necessary limit break the loop """
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame = cv2.flip(frame, 1)
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    cap.release()
    cv2.destroyAllWindows()
