import tensorflow as tf
import numpy as np


def load_graph(frozen_graph_filename):
	with tf.gfile.GFile(frozen_graph_filename, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
	with tf.Graph().as_default() as graph:
		tf.import_graph_def(
			graph_def,
			input_map=None,
			return_elements=None,
			name="",
			op_dict=None,
			producer_op_list=None
		)
	return graph


class ImportGraph():
	def __init__(
			self, graph, gpu_device="0", input_tensor_names=('in_rgb', 'in_flow'),
			output_tensor_name='output'):
		self.graph = graph
		self.gpu_device = gpu_device
		self.x_rgb = self.graph.get_tensor_by_name('{}:0'.format(input_tensor_names[0]))
		# self.x_flow = self.graph.get_tensor_by_name('{}:0'.format(input_tensor_names[1]))
		self.is_training = self.graph.get_tensor_by_name('{}:0'.format('is_training'))
		self.predictor = self.graph.get_tensor_by_name('{}:0'.format(output_tensor_name))

		session_conf = tf.ConfigProto(use_per_session_threads=True)
		self.sess = tf.Session(graph=self.graph, config=session_conf)
		self.warm_up()

	def warm_up(self):
		self.sess.run(self.predictor, feed_dict={self.x_rgb: np.ones((1,64,224,224,3)),self.is_training:False})

	def run(self, data, is_training=False, bs=1):
		### NOTE: hot fix ###
		# bs = 4
		in_rgb = np.concatenate(([data] * bs))
		result = self.sess.run(self.predictor, feed_dict={self.x_rgb: in_rgb, self.is_training: is_training})
		# self.sess.close()
		return result

	def close(self):
		self.sess.close()
