import os, shutil
import tensorflow as tf

saver = tf.train.import_meta_graph('./resources/models/flight_spell/saved_model.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
sess = tf.Session()
saver.restore(sess, "./resources/models/flight_spell/saved_model.ckpt")


# Freezing graph to use with Tensorflow Serving
inputs = graph.get_tensor_by_name("inputs/inputs:0")
inputs_length = graph.get_tensor_by_name("inputs_length:0")
targets_length = graph.get_tensor_by_name("targets_length:0")
keep_prob = graph.get_tensor_by_name("keep_prob:0")

predictions = graph.get_tensor_by_name("predictions/predictions:0")

model_input = {
    'inputs': tf.saved_model.utils.build_tensor_info(inputs),
    'inputs_length': tf.saved_model.utils.build_tensor_info(inputs_length),
    'targets_length': tf.saved_model.utils.build_tensor_info(targets_length),
    'keep_prob': tf.saved_model.utils.build_tensor_info(keep_prob)
}
model_output = tf.saved_model.utils.build_tensor_info(predictions)

signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=model_input,
    outputs={'outputs': model_output},
    method_name= tf.saved_model.signature_constants.PREDICT_METHOD_NAME)


dir_name = "./resources/models/flight_spell/serve/"
if os.path.isdir(dir_name):
    shutil.rmtree(dir_name)
os.makedirs(dir_name)

builder = tf.saved_model.builder.SavedModelBuilder('./resources/models/flight_spell/serve/1/')
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature_definition
    })
builder.save()
 
sess.close()
