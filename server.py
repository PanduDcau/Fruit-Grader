import pymongo
from bson import ObjectId
import tensorflow as tf
import numpy as np
import logging
from flask import Flask, Response, request, render_template, jsonify
from PIL import Image, ImageOps
from io import BytesIO
import cv2
import os
import random

from pymongo import MongoClient
import json

DATABASE_INTEGRATION = True

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

if DATABASE_INTEGRATION:
    # Connect to the MongoDB Atlas cluster
    client = MongoClient("mongodb+srv:"
                         "//dasabro:Dasabro_123@cluster1.e7bgf6n.mongodb.net/?retryWrites=true&w=majority")

    # Select the database and collection
    db = client["fruit-grader"]
    predictions_collection = db["predictions"]

fruit_logic_map = None

with open("config/fruit_logic_map.json") as fruit_logic_map_file:
    fruit_logic_map = json.load(fruit_logic_map_file)

def insert_prediction(prediction):
    # Insert the predicted results
    prediction_id = predictions_collection.insert_one(prediction).inserted_id
    print("Prediction added with ID:", prediction_id)


def remove_image_background(input_image):
    input_image = np.asarray(input_image)

    # Convert to grayscale
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, thresh = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, hierarchy = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Choose the contour with the largest area
    contour_sizes = [(cv2.contourArea(contour), contour)
                     for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    # Create a mask and fill the contour with white
    mask = np.zeros(input_image.shape[:2], np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)

    # Apply the mask to the original image to remove the background
    result = cv2.bitwise_and(input_image, input_image, mask=mask)

    # Create a white background
    background = np.full(input_image.shape, 255, dtype=np.uint8)

    # # Apply the mask to the white background
    # background_mask = cv2.bitwise_not(mask)
    # bg_removed = cv2.bitwise_and(background, background, mask=background_mask)
    #
    # # Add the background to the result image
    # output_image = cv2.add(result, bg_removed)

    output_image = Image.fromarray(result) #output_image

    # Generate a random number between 0 and 999
    num = random.randint(0, 999)

    output_dir = "Background_Image"
    # Save the result
    output_filename = 'Background_Image\output_image' + '_' + str(num) + '.jpg'
    print(output_filename)
    # cv2.imwrite(output_filename)

    return output_image


@app.route("/")
# Render the home page
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
# Predict endpoint
def predict():
    # Get the image file from the request
    if "image" not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    image = request.files["image"]

    remove_background = bool(
        request.form["remove_background"].lower() == 'true')

    # Read the image file, resize and preprocess it
    try:
        img_bytes = image.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")
        if remove_background == True:
            image = remove_image_background((image))

        size = (224, 224)
        image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
    except Exception as e:
        logger.exception("Failed to preprocess the image")
        return jsonify({"error": "Failed to preprocess the image"}), 400

    # Perform the initial prediction
    try:
        initial_model_path = "models/initial_model/keras_model.h5"
        initial_model = tf.keras.models.load_model(
            initial_model_path, compile=False)
        initial_prediction = initial_model.predict(data)
        label_index = np.argmax(initial_prediction)
        initial_label_path = "models/initial_model/labels.txt"
        class_labels = [fruit["label"] for fruit in fruit_logic_map.values()]
        label = class_labels[label_index]
    except Exception as e:
        logger.exception("Failed to make an initial prediction")

    # Perform predictions on the 4 specific fruit class models
    try:
        fruit_predictions = []
        total_score = 0
        for i in range(4):
            fruit_model_path = "models/fruit_class_{}/model_{}/keras_model.h5".format(
                str(label_index), str(i))
            fruit_model = tf.keras.models.load_model(
                fruit_model_path, compile=False)
            fruit_prediction = fruit_model.predict(data)
            fruit_label_index = np.argmax(fruit_prediction)
            fruit_class_labels = [prediction_class["label"]
                                  for prediction_class in
                                  fruit_logic_map[str(label_index)]["models"][str(i)]["classes"].values()]
            fruit_label = fruit_class_labels[fruit_label_index]
            fruit_predictions.append({"fruit_label": fruit_label, "fruit_probability": str(
                fruit_prediction[0][fruit_label_index])})

            total_score += (fruit_logic_map[str(label_index)]["models"][str(i)]["weight"] *
                            fruit_logic_map[str(label_index)]["models"][str(i)]["classes"][str(fruit_label_index)][
                                "score"])
    except Exception as e:
        logger.exception("Failed to make predictions on fruit specific models")
        return jsonify({"error": "Failed to make predictions on fruit specific models"}), 500

    total_score = round(total_score, 2)
    fruit_grade = None
    for grade, thresholds in fruit_logic_map[str(label_index)]["grades"].items():
        if (total_score > thresholds["min"] and total_score <= thresholds["max"]):
            fruit_grade = grade

    if DATABASE_INTEGRATION:
        # call insert_prediction function with predicted results
        insert_prediction({"label": label, "probability": str(
            initial_prediction[0][label_index]), "fruit_predictions": fruit_predictions, "fruit_grade": fruit_grade,
                           "total_score": total_score})

    # Return the result as a JSON object
    return jsonify(
        {"label": label, "probability": str(initial_prediction[0][label_index]), "fruit_predictions": fruit_predictions,
         "fruit_grade": fruit_grade, "total_score": total_score})


@app.route("/predictions")
# Render predictions table saved in database
def predictions():
    # fetch all predictions from the database
    predictions = list(predictions_collection.find())
    return render_template("predictions.html", predictions=predictions)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
