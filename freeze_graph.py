import os, shutil
import tensorflow as tf

saver = tf.train.import_meta_graph('./resources/models/flight_spell.ckpt.meta', clear_devices=True)
graph = tf.get_default_graph()
sess = tf.Session()
saver.restore(sess, "./resources/models/flight_spell.ckpt")


'''
print(len([op.name for op in graph.get_operations()]))

output_graph_def = tf.graph_util.convert_variables_to_constants(
                        sess, # The session
                        graph.as_graph_def(), # input_graph_def is useful for retrieving the nodes 
                        ['RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/Min',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/Max',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/while/Exit',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/while/Exit_1',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/while/Exit_3',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/while/Exit_4',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/fw/fw/transpose_1',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/Min',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/Max',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/while/Exit',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/while/Exit_1',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/while/Exit_3',
                            'RNN_Encoder_Cell_2D/encoder_0/bidirectional_rnn/bw/bw/while/Exit_4',
                            'RNN_Encoder_Cell_2D/encoder_0/ReverseSequence',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/Min',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/Max',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/while/Exit',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/while/Exit_1',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_1',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/ExpandDims_3',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/Min',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/Max',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/while/Exit',
                            'RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/while/Exit_1',
                            'BahdanauAttention/memory_layer/Tensordot/concat',
                            'AttentionWrapperZeroState/checked_cell_state',
                            'AttentionWrapperZeroState/checked_cell_state_1',
                            'AttentionWrapperZeroState/ExpandDims_1',
                            'decode/Training_Decoder/TrainingHelper/strided_slice_1',
                            'decode/Training_Decoder/decoder/TrainingHelperInitialize/cond/switch_t',
                            'decode/Training_Decoder/decoder/while/BasicDecoderStep/TrainingHelperNextInputs/cond/switch_t',
                            'decode/Training_Decoder/decoder/while/BasicDecoderStep/TrainingHelperNextInputs/cond/switch_f',
                            'decode/Training_Decoder/decoder/while/Exit',
                            'decode/Training_Decoder/decoder/while/Exit_5',
                            'decode/Training_Decoder/decoder/while/Exit_6',
                            'decode/Training_Decoder/decoder/while/Exit_8',
                            'decode/Training_Decoder/decoder/while/Exit_9',
                            'decode/Training_Decoder/decoder/while/Exit_10',
                            'decode/Training_Decoder/decoder/transpose_1',
                            'decode_1/Inference_Decoder/decoder/while/BasicDecoderStep/cond/switch_t',
                            'decode_1/Inference_Decoder/decoder/while/BasicDecoderStep/cond/switch_f',
                            'decode_1/Inference_Decoder/decoder/while/Exit',
                            'decode_1/Inference_Decoder/decoder/while/Exit_3',
                            'decode_1/Inference_Decoder/decoder/while/Exit_4',
                            'decode_1/Inference_Decoder/decoder/while/Exit_5',
                            'decode_1/Inference_Decoder/decoder/while/Exit_6',
                            'decode_1/Inference_Decoder/decoder/while/Exit_7',
                            'decode_1/Inference_Decoder/decoder/while/Exit_8',
                            'decode_1/Inference_Decoder/decoder/while/Exit_9',
                            'decode_1/Inference_Decoder/decoder/while/Exit_10',
                            'decode_1/Inference_Decoder/decoder/transpose',
                            'cost/sequence_loss/SparseSoftmaxCrossEntropyWithLogits/Shape',
                            'optimze/gradients/b_count_3',
                            'optimze/gradients/b_count_7',
                            'optimze/gradients/b_count_11',
                            'optimze/gradients/cost/sequence_loss/truediv_grad/tuple/control_dependency_1',
                            'optimze/gradients/cost/sequence_loss/mul_grad/tuple/control_dependency_1',
                            'optimze/gradients/zeros_like',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/Exit_2_grad/b_exit',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/Exit_7_grad/b_exit',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/Enter_1_grad/Exit',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/Enter_5_grad/Exit',
                            'optimze/gradients/decode/Training_Decoder/decoder/TrainingHelperInitialize/cond/Merge_grad/tuple/control_dependency_1',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/BasicDecoderStep/TrainingHelperNextInputs/cond/Merge_grad/tuple/control_dependency_1',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/Select_1_grad/tuple/control_dependency',
                            'optimze/gradients/decode/Training_Decoder/TrainingHelper/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/mul_grad/tuple/control_dependency',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/ones_like_grad/Sum',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/lstm_cell/add_grad/tuple/control_dependency_1',
                            'optimze/gradients/BahdanauAttention/Reshape_grad/Reshape',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/lstm_cell/concat_grad/Shape_1',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/dropout/mul_grad/tuple/control_dependency_1',
                            'optimze/gradients/decode/Training_Decoder/decoder/while/BasicDecoderStep/decoder/attention_wrapper/dropout/div_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/concat_grad/Shape',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/while/Enter_2_grad/Exit',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/Sum',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/Sum',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/while/Enter_2_grad/Exit',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_grad/Sum',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/DropoutWrapperZeroState/LSTMCellZeroState/zeros_1_grad/Sum',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/zeros_grad/Sum',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/zeros_grad/Sum',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/while/lstm_cell/add_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/while/lstm_cell/add_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/while/lstm_cell/concat_grad/Shape',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/while/dropout/mul_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/while/lstm_cell/concat_grad/Shape',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/while/dropout/div_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/while/dropout/mul_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/while/dropout/div_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/fw/fw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1',
                            'optimze/gradients/RNN_Encoder_Cell_2D/encoder_1/bidirectional_rnn/bw/bw/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/tuple/control_dependency_1',
                            'Variable/Adam/read',
                            'Variable/Adam_1/read',
                            'encoder_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam/read',
                            'encoder_1/bidirectional_rnn/fw/lstm_cell/kernel/Adam_1/read',
                            'encoder_1/bidirectional_rnn/fw/lstm_cell/bias/Adam/read',
                            'encoder_1/bidirectional_rnn/fw/lstm_cell/bias/Adam_1/read',
                            'encoder_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam/read',
                            'encoder_1/bidirectional_rnn/bw/lstm_cell/kernel/Adam_1/read',
                            'encoder_1/bidirectional_rnn/bw/lstm_cell/bias/Adam/read',
                            'encoder_1/bidirectional_rnn/bw/lstm_cell/bias/Adam_1/read',
                            'Variable_1/Adam/read',
                            'Variable_1/Adam_1/read',
                            'memory_layer/kernel/Adam/read',
                            'memory_layer/kernel/Adam_1/read',
                            'decode/decoder/attention_wrapper/lstm_cell/kernel/Adam/read',
                            'decode/decoder/attention_wrapper/lstm_cell/kernel/Adam_1/read',
                            'decode/decoder/attention_wrapper/lstm_cell/bias/Adam/read',
                            'decode/decoder/attention_wrapper/lstm_cell/bias/Adam_1/read',
                            'decode/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel/Adam/read',
                            'decode/decoder/attention_wrapper/bahdanau_attention/query_layer/kernel/Adam_1/read',
                            'decode/decoder/attention_wrapper/bahdanau_attention/attention_v/Adam/read',
                            'decode/decoder/attention_wrapper/bahdanau_attention/attention_v/Adam_1/read',
                            'decode/decoder/attention_wrapper/attention_layer/kernel/Adam/read',
                            'decode/decoder/attention_wrapper/attention_layer/kernel/Adam_1/read',
                            'decode/decoder/dense/kernel/Adam/read',
                            'decode/decoder/dense/kernel/Adam_1/read',
                            'decode/decoder/dense/bias/Adam/read',
                            'decode/decoder/dense/bias/Adam_1/read',
                            'Merge/MergeSummary',
                            'save/control_dependency',
                            'save_1/control_dependency',
                            'save_2/control_dependency',
                            'save_3/control_dependency',
                            'save_4/control_dependency',
                            'save_5/control_dependency',
                            'save_6/control_dependency',
                            'save_7/control_dependency',
                            'save_8/control_dependency',
                            'save_9/control_dependency',
                            'save_10/control_dependency']
            )

output_graph="./resources/models/flight_spell.pb"
with tf.gfile.GFile(output_graph, "wb") as f:
    f.write(output_graph_def.SerializeToString())
'''

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


dir_name = "./resources/models/flight_spell/"
if os.path.isdir(dir_name):
    shutil.rmtree(dir_name)
os.makedirs(dir_name)

builder = tf.saved_model.builder.SavedModelBuilder('./resources/models/flight_spell/1/')
builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
            signature_definition
    })
builder.save()
 
sess.close()
