
from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "soil_ai_secret_key"

# =====================================================
# MODEL LOADING
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "model", "soil_model.h5")

model = tf.keras.models.load_model(model_path)

# =====================================================
# SOIL CLASSES
# =====================================================

soil_classes = ['alluvial', 'black', 'clay', 'red']

# =====================================================
# CROP RECOMMENDATION
# =====================================================

crop_map = {

    "alluvial": [
        "Rice",
        "Wheat",
        "Sugarcane",
        "Maize",
        "Pulses",
        "Groundnut",
        "Watermelon",
        "Banana",
        "Mango",
        "Potato",
        "Tomato",
        "Onion",
        "Mustard",
        "Jute"
    ],

    "clay": [
        "Rice",
        "Wheat",
        "Barley",
        "Lettuce",
        "Cabbage",
        "Broccoli",
        "Beans",
        "Peas",
        "Sunflower",
        "Cotton",
        "Sugarcane",
        "Tomato",
        "Chili",
        "Spinach"
    ],

    "red": [
        "Groundnut",
        "Millets",
        "Potato",
        "Pulses",
        "Tobacco",
        "Cotton",
        "Tomato",
        "Onion",
        "Carrot",
        "Chili"
    ],

    "black": [
        "Cotton",
        "Soybean",
        "Sorghum",
        "Wheat",
        "Sunflower",
        "Linseed",
        "Millets",
        "Tobacco",
        "Sugarcane",
        "Groundnut",
        "Pulses",
        "Maize",
        "Chili",
        "Tomato"
    ]
}

# =====================================================
# SOIL DETAILS DATABASE
# =====================================================

soil_info = {

    "alluvial": {
        "minerals": "Potash, Lime, Phosphoric Acid",
        "ph": "6.5 - 8.4",
        "color": "Light Grey",
        "fertility": "Very Fertile",
        "water": "High Water Retention",
        "fertilizer": "NPK, Compost",

        "description":
        "Alluvial soil is highly fertile and rich in nutrients. It is mainly found near river areas and is excellent for crops like rice, wheat, sugarcane, banana, and potato."
    },

    "black": {
        "minerals": "Iron, Magnesium, Aluminum",
        "ph": "7.0 - 8.5",
        "color": "Black",
        "fertility": "High",
        "water": "Excellent Moisture Holding",
        "fertilizer": "Nitrogen, Phosphorus",

        "description":
        "Black soil is rich in iron and magnesium with excellent moisture retention capacity. It is highly suitable for cotton, soybean, sunflower, wheat, and sugarcane farming."
    },

    "clay": {
        "minerals": "Silica, Alumina",
        "ph": "5.5 - 7.0",
        "color": "Dark Brown",
        "fertility": "Moderate to High",
        "water": "Very High Retention",
        "fertilizer": "Organic Compost",

        "description":
        "Clay soil has high water retention and contains fine mineral particles. It is good for crops like rice, cabbage, broccoli, beans, and spinach."
    },

    "red": {
        "minerals": "Iron Oxide, Potash",
        "ph": "6.0 - 7.0",
        "color": "Red",
        "fertility": "Medium",
        "water": "Low Water Retention",
        "fertilizer": "Potassium, Organic Matter",

        "description":
        "Red soil is rich in iron oxide and supports crops such as groundnut, millets, potato, tobacco, cotton, and pulses."
    }

}

# =====================================================
# FILE VALIDATION
# =====================================================

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):

    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# =====================================================
# SIMPLE USER DATABASE
# =====================================================

users = {}

# =====================================================
# HOME
# =====================================================

@app.route('/')
def home():

    if 'user' in session:
        return redirect(url_for('welcome'))

    return redirect(url_for('login'))

# =====================================================
# REGISTER
# =====================================================

@app.route('/register', methods=['GET', 'POST'])
def register():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        if email in users:
            return "User already exists!"

        users[email] = password

        return redirect(url_for('login'))

    return render_template("register.html")

# =====================================================
# LOGIN
# =====================================================

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'POST':

        email = request.form['email']
        password = request.form['password']

        if email in users and users[email] == password:

            session['user'] = email

            return redirect(url_for('welcome'))

        return "Invalid Email or Password!"

    return render_template("login.html")

# =====================================================
# WELCOME PAGE
# =====================================================

@app.route('/welcome')
def welcome():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template("welcome.html")

# =====================================================
# DASHBOARD
# =====================================================

@app.route('/dashboard')
def dashboard():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template("dashboard.html")

# =====================================================
# PREDICTION PAGE
# =====================================================

@app.route('/predict-page')
def predict_page():

    if 'user' not in session:
        return redirect(url_for('login'))

    return render_template("index.html")

# =====================================================
# PREDICTION
# =====================================================

@app.route('/predict', methods=['POST'])
def predict():

    if 'user' not in session:
        return redirect(url_for('login'))

    # CHECK IMAGE
    if 'image' not in request.files:
        return "No Image Uploaded!"

    file = request.files['image']

    if file.filename == '':
        return "No File Selected!"

    if not allowed_file(file.filename):
        return "Only PNG, JPG, JPEG files allowed!"

    # =====================================================
    # CREATE UPLOAD FOLDER
    # =====================================================

    upload_folder = os.path.join("static", "uploads")

    os.makedirs(upload_folder, exist_ok=True)

    # =====================================================
    # SAVE IMAGE
    # =====================================================

    filename = secure_filename(file.filename)

    path = os.path.join(upload_folder, filename)

    file.save(path)

    # =====================================================
    # IMAGE PREPROCESSING
    # =====================================================

    img = image.load_img(path, target_size=(128, 128))

    img = image.img_to_array(img)

    img = np.expand_dims(img, axis=0)

    img = img / 255.0

    # =====================================================
    # MODEL PREDICTION
    # =====================================================

    prediction = model.predict(img)

    index = np.argmax(prediction)

    confidence = round(float(np.max(prediction)) * 100, 2)

    soil = soil_classes[index]

    crops = crop_map.get(soil, [])

    details = soil_info.get(soil, {})

    description = details.get(
        "description",
        "No description available"
    )

    # =====================================================
    # RESULT PAGE
    # =====================================================

    return render_template(

        "result.html",

        soil=soil.title(),

        crops=crops,

        image=path,

        confidence=confidence,

        details=details,

        description=description

    )

# =====================================================
# LOGOUT
# =====================================================

@app.route('/logout')
def logout():

    session.clear()

    return redirect(url_for('login'))

# =====================================================
# ERROR PAGE
# =====================================================

@app.errorhandler(404)
def not_found(error):

    return render_template("404.html"), 404

# =====================================================
# RUN APP
# =====================================================

if __name__ == "__main__":

    app.run(host="0.0.0.0", debug=True)