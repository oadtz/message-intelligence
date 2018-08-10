import tensorflow as tf

saver = tf.train.import_meta_graph('./resources/models/flight_spell.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
sess = tf.Session()
saver.restore(sess, "./resources/models/flight_spell.ckpt")

output_nodes = list()
for op in graph.get_operations():
    for output in ['predictions']:
        if output in op.name:
            output_nodes.append(op.name)
            print(op.name)

output_graph_def = tf.graph_util.convert_variables_to_constants(
                        sess, # The session
                        graph.as_graph_def(), # input_graph_def is useful for retrieving the nodes 
                        [op.name for op in graph.get_operations()]
            )

output_graph="./resources/models/flight_spell.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
 
sess.close()
