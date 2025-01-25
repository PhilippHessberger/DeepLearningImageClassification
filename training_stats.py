import tensorflow as tf


# create own callback for stats logging
class TrainingStats(tf.keras.callbacks.Callback):
	def __init__(self, stats_filepath, model_file_name):
		super().__init__()
		self.model_file_name = model_file_name
		self.stats_filepath = stats_filepath

	def on_train_begin(self, logs=None):
		with open(self.stats_filepath, "a") as f:
			f.write(f"\nmodel_file_name: {self.model_file_name}\n")

	def on_epoch_end(self, epoch, logs=None):
		if logs is None:
			logs = {}

		with open(self.stats_filepath, "a") as f:
			f.write(f"current_epoch: {epoch}\t\tloss: {round(logs.get('loss'), 3)}\t\taccuracy: {round(logs.get('accuracy'), 3)}\t\tval_loss: {round(logs.get('val_loss'), 3)}\t\tval_accuracy: {round(logs.get('val_accuracy'), 3)}\n")


# create own callback for stats logging
class TrainingStatsStudent(tf.keras.callbacks.Callback):
	def __init__(self, stats_filepath, model_file_name):
		super().__init__()
		self.model_file_name = model_file_name
		self.stats_filepath = stats_filepath

	def on_train_begin(self, logs=None):
		with open(self.stats_filepath, "a") as f:
			f.write(f"\nmodel_file_name: {self.model_file_name}\n")

	def on_epoch_end(self, epoch, logs=None):
		if logs is None:
			logs = {}

		with open(self.stats_filepath, "a") as f:
			f.write(
				f"current_epoch: {epoch}\t\tloss: {round(logs.get('student_loss'), 3)}\t\taccuracy: {round(logs.get('categorical_accuracy'), 3)}\t\tval_loss: {round(logs.get('val_student_loss'), 3)}\t\tval_accuracy: {round(logs.get('val_categorical_accuracy'), 3)}\n")
