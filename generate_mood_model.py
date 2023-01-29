from keras.models import load_model

# Load the pre-trained mood model
model = load_model("mood_model.h5")

# Save the model as an .h5 file
model.save("mood_model.h5")
