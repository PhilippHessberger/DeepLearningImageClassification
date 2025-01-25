import os
import cv2
import tensorflow as tf
import absl.logging

from distiller import Distiller
from model_factory import new_model_from_pretrained, create_custom_model, create_modified_vgg19
from image_handler import augment_and_save_dataset, load_tensorflow_dataset_from_folder
from training_stats import TrainingStats, TrainingStatsStudent

# so 'WARNING:absl:Found untraced functions' does not show up again
absl.logging.set_verbosity(absl.logging.ERROR)

# flags for easier changing of code behavior
create_new_dataset = False

# if you want to load a trained model
load_model = 'teacher_vgg19_base_dense100_batchnorm_dense10_10epochs_freezebase.h5'
summary = True

# if you want to create a new custom model
create_model = False

# load pretrained model and change top layers
new_model_from_pretrained_model = None

# load teacher model if you want knowledge distillation
teacher_model_name = None # 'teacher_vgg19_base_dense100_batchnorm_dense10_10epochs_freezebase.h5'
knowledge_distillation = False

# if you want to train a model
training = False

# if you want to evaluate a loaded OR trained model
validation = False

# if you want to do final evaluation on a loaded OR trained model
testing = False
single_label_testing = False
single_image_testing = False

# if you want to save a model and do a test or validation evaluation on a model
model_file_name = None

# self-explanatory
number_of_epochs = 1

directory_names = [
	'Arbutus unedo',
	'Betula pendula',
	'Buxus sempervirens',
	'Corylus avellana',
	'Crataegus monogyna',
	'Hedera helix',
	'Platanus x',
	'Robinia pseudoacacia',
	'Ulmus minor',
	'Viburnum tinus'
]

# batch_size = 32
training_folder = 'training_data'  # 70% -> 700 per label, but everything x3 for unfiltered data
validation_folder = 'validation_data'  # 20% -> 200 per label
testing_folder = 'testing_data'  # 10% -> 100 per label
single_label_testing_folder = 'single_label_testing_folder'

# create own dataset
if create_new_dataset:
	filepath = 'D:/Repositories/Datasets/PlantCLEF/ms3_dataset/'
	augment_and_save_dataset(filepath, directory_names, testing_folder, limit=100, only_resize_and_rotate=True)

if create_model:
	model = create_modified_vgg19()

if teacher_model_name:
	teacher_model = tf.keras.models.load_model('trained_models/' + str(teacher_model_name))
	if not load_model:
		student_model = create_custom_model()
		# student_control_model = keras.models.clone_model(student_model)
	else:
		load_model_name = 'trained_models/' + str(load_model)
		student_model = tf.keras.models.load_model(load_model_name)

with tf.device('/device:GPU:0'):
	if load_model is not None and teacher_model_name is None:
		load_model_name = 'trained_models/' + str(load_model)
		model = tf.keras.models.load_model(load_model_name)  # , compile=False)
	# model.compile(optimizer='adam',
	# 			  loss='categorical_crossentropy',
	# 			  metrics=['accuracy'])

	if new_model_from_pretrained_model is not None:
		model = new_model_from_pretrained(new_model_from_pretrained_model)

	if training:
		# callback for checkpoint saving weights after every epoch
		# checkpoint_filepath = 'trained_models/' + str(model_file_name) + '_checkpoints/epoch_{epoch}'
		# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
		# 	filepath=checkpoint_filepath,
		# 	save_weights_only=True,
		# 	monitor='val_accuracy',
		# 	mode='max',
		# 	save_best_only=False)

		# create own callback for learning rate
		class LearningRateReducerCb(tf.keras.callbacks.Callback):
			def on_epoch_end(self, epoch, logs=None):
				old_learningrate = self.model.optimizer.lr.read_value()
				new_learningrate = old_learningrate * 0.1
				print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_learningrate,
																				 new_learningrate))
				self.model.optimizer.lr.assign(new_learningrate)


		# create callback for tensorboard
		tensorboard_training_log = tf.keras.callbacks.TensorBoard(log_dir='logs\\{}'.format(model_file_name))

		# load training dataset
		training_dataset = load_tensorflow_dataset_from_folder(
			folder_name=training_folder)  # filepath='D:/Repositories/Datasets/PlantCLEF/training')

		validation_dataset = load_tensorflow_dataset_from_folder(folder_name=validation_folder)

		# train model
		model.fit(
			training_dataset,
			epochs=number_of_epochs,
			callbacks=[
				TrainingStats("training_stats.txt", model_file_name),
				# LearningRateReducerCb(), this activates a custom learning rate
				# model_checkpoint_callback
			],
			validation_data=validation_dataset,
			validation_freq=1
		)

	if knowledge_distillation and teacher_model_name:
		distiller = Distiller(student=student_model, teacher=teacher_model)

		distiller.compile(
			optimizer=tf.keras.optimizers.Adam(),
			metrics=[tf.keras.metrics.CategoricalAccuracy()],
			student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
			distillation_loss_fn=tf.keras.losses.KLDivergence(),
			alpha=0.1,
			temperature=10,
		)

		# student_control_model.compile(
		# 	optimizer='adam',
		# 	loss='categorical_crossentropy',
		# 	metrics=['accuracy'])

		training_dataset = load_tensorflow_dataset_from_folder(folder_name=training_folder)
		validation_dataset = load_tensorflow_dataset_from_folder(folder_name=validation_folder)

		distiller.fit(
			training_dataset,
			epochs=number_of_epochs,
			callbacks=[
				TrainingStatsStudent("training_stats.txt", model_file_name)
			],
			validation_data=validation_dataset,
			validation_freq=1
		)

	# student_control_model.fit(
	# 	training_dataset,
	# 	epochs=number_of_epochs,
	# 	callbacks=[
	# 		TrainingStats("training_stats.txt", model_file_name + '_control')
	# 	],
	# 	validation_data=validation_dataset,
	# 	validation_freq=1
	# )

	if model_file_name is not None:
		if knowledge_distillation and teacher_model_name:
			distiller.student.save('trained_models/' + str(model_file_name), save_format='tf')
		# student_control_model.save('trained_models/' + str(model_file_name) + 'control', save_format='tf')
		else:
			model.save('trained_models/' + str(model_file_name) + '.h5', save_format='h5')

	if validation:
		if model_file_name is not None:
			model_name = model_file_name
		elif load_model is not None:
			model_name = load_model

		# create callback for tensorboard
		validation_dataset = load_tensorflow_dataset_from_folder(folder_name=testing_folder)

		loss, acc = model.evaluate(validation_dataset)  # , callbacks=[tensorboard_validation_log])

		with open('training_stats.txt', "a") as f:
			f.write(
				f"\neval model_file_name: {model_name}\neval_loss: {round(loss, 3)}\neval_acc: {round(acc, 3)}\n")

	if testing:
		# create callback for tensorboard
		testing_dataset = load_tensorflow_dataset_from_folder(folder_name=testing_folder)
		loss, acc = model.evaluate(testing_dataset)  # , callbacks=[tensorboard_testing_log])
		print('model accuracy:', acc)
		print('model loss:', loss)

	if single_label_testing:
		# create callback for tensorboard
		# tensorboard_testing_log = tf.keras.callbacks.TensorBoard(log_dir='logs\\{}'.format(str(model_file_name) + 'single_testing'))
		single_label_testing_dataset = load_tensorflow_dataset_from_folder(folder_name=single_label_testing_folder)
		loss, acc = model.evaluate(single_label_testing_dataset)  # , callbacks=[tensorboard_testing_log])
		print('model accuracy:', acc)
		print('model loss:', loss)

		with open('training_stats.txt', "a") as f:
			f.write(
				f"\neval model_file_name: {load_model}\neval_loss: {round(loss, 3)}\neval_acc: {round(acc, 3)}\n")

	if single_image_testing:
		folder = 'D:/Repositories/DeepLearningImageClassification/data/testing_data/Arbutus unedo/'
		# images = []

		# for filename in os.listdir(folder):
		# 	img = cv2.imread(os.path.join(folder, filename))
		# 	if img is not None:
		# 		img = cv2.resize(img, (256, 256))
		# 		images.append(img)

		image = cv2.imread(os.path.join(folder, '2071_v4.jpg'))
		img = cv2.resize(image, (256, 256))

		# predict images
		# prediction = model.predict(np.array(images))
		# for i in range(len(prediction)):
		# print(directory_names[np.argmax(prediction[i])])
		# print(prediction[i])

		prediction = model.predict(img)

	if summary:
		model.summary()
