import os
import cv2
import math
import sys, getopt
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import *
from glob import glob
from functools import reduce
from collections import defaultdict
from focal_loss import BinaryFocalLoss
from TrackNet import ResNet_Track
from tensorflow import keras
from tensorflow.keras import backend as K
import yaml


# read parameters
yaml_path = sys.argv[1]
file = open(yaml_path, mode='r', encoding="utf-8")
file_data = file.read()
file.close()
parameters = yaml.load(file_data)


tol = parameters['tol']
save_weights = parameters['save_weights']
HEIGHT = parameters['HEIGHT']
WIDTH = parameters['WIDTH']
BATCH_SIZE = parameters['batch_size']
FRAME_STACK = parameters['frame_stack']
pre_trained = parameters['pre_trained']

optimizer = keras.optimizers.Adadelta(lr=parameters['lr'])
if not pre_trained:
	model=ResNet_Track(input_shape=(FRAME_STACK, HEIGHT, WIDTH))
	model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=optimizer, metrics=[keras.metrics.BinaryAccuracy()]) 
else:
	model=ResNet_Track(input_shape=(FRAME_STACK, HEIGHT, WIDTH))
	model.load_weights(parameters['load_weights'])
	model.compile(loss=BinaryFocalLoss(gamma=2), optimizer=optimizer, metrics=[keras.metrics.BinaryAccuracy()])

print('Beginning training......')
match_path = parameters['match_folder']

match_list = [os.sep.join([os.getcwd(), match_path, match]) for match in os.listdir(match_path)]

wait = 0
best_loss = float('inf')
losses = []
for i in range(parameters['epochs']):
	x_train, x_test, y_train, y_test = split_train_test(match_list, ratio=parameters['split_ratio'], shuffle=True)
	train_steps = check_steps(x_train+x_test, BATCH_SIZE, FRAME_STACK)
	print("==========Epoch {}, Train steps: {}, Learning rate: {:.4f}==========".format(i, train_steps, float(K.get_value(model.optimizer.lr))))
	history = model.fit(data_generator(BATCH_SIZE, x_train+x_test, y_train+y_test, FRAME_STACK), 
						steps_per_epoch=train_steps,
						epochs=1,
						verbose=1)
	loss = sum(history.history['loss'])
	losses.append(loss)
	
	# validation
	TP = TN = FP1 = FP2 = FN = 0
	test_iter = iter(data_generator(BATCH_SIZE, x_test, y_test, FRAME_STACK))
	test_steps = check_steps(x_test, BATCH_SIZE, FRAME_STACK)
	print("==========Epoch {} start validation==========".format(i))
	for j in range(test_steps):
		x_batch, y_batch = next(test_iter)
		y_pred = model.predict(x_batch, batch_size=BATCH_SIZE)
		y_pred = y_pred > 0.5
		y_pred = y_pred.astype('float32')

		tp, tn, fp1, fp2, fn = confusion(y_pred, y_batch[:, 0,...], tol)
		TP += tp
		TN += tn
		FP1 += fp1
		FP2 += fp2
		FN += fn
	
	accuracy, precision, recall = compute_acc((TP, TN, FP1, FP2, FN))
	avg_acc = (accuracy + precision + recall)/3
	print("Epoch {} accuracy: {:.3f}".format(i, accuracy))
	print("Epoch {} precision: {:.3f}".format(i, precision))
	print("Epoch {} recall: {:.3f}".format(i, recall))
	print("Epoch {} average = (accuracy + precision + recall)/3: {:.3f}".format(i, avg_acc))
	
	# learnging rate callback and saving model
	if loss < best_loss - parameters['min_delta']:
		wait = 0
		best_loss = loss
		model.save_weights(save_weights)
	else:
		wait += 1
		if wait >= parameters['patience']:
			old_lr = float(K.get_value(model.optimizer.lr))
			if old_lr > parameters['min_lr']:
				new_lr = old_lr * parameters['r_factor']
				new_lr = max(new_lr, parameters['min_lr'])
				K.set_value(model.optimizer.lr, new_lr)
				wait = 0
				print("Reduce model learning rate to {}".format(new_lr))
			
plt.plot(losses)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig("Loss_epoch.jpg")	

print('Saving weights......')
# model.save_weights(save_weights+'_final')
model.save_weights('weight_final')
print('Done......')
