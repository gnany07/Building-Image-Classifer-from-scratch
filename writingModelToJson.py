from keras.models import load_model

classifier=load_model('mymodel.h5')
model_json=classifier.to_json()
with open('model.json','w') as json_file:
	json_file.write(model_json)