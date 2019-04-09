import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from bert import modeling
import tensorflow as tf
import tensorflow_hub as hub
from data_handler import DataHandler


args = {
    "train_size": -1,
    "val_size": -1,
    "full_data_dir": "./../Data/",
    "data_dir": "./../Data/",
    "task_name": "toxic_multilabel",
    "no_cuda": False,
    "bert_model": "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
    "output_dir": "./../Data/Output",
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 32,
    "eval_batch_size": 32,
    "learning_rate": 3e-5,
    "num_train_epochs": 4.0,
    "warmup_proportion": 0.1,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                       num_train_steps, num_warmup_steps, use_tpu,
                       use_one_hot_embeddings):
      """Returns `model_fn` closure for TPUEstimator."""

      def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
          """The `model_fn` for TPUEstimator."""

          tf.logging.info("*** Features ***")
          for name in sorted(features.keys()):
              tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

          input_ids = features["input_ids"]
          input_mask = features["input_mask"]
          segment_ids = features["segment_ids"]
          label_ids = features["label_ids"]
          is_real_example = None
          if "is_real_example" in features:
              is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
          else:
              is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

          is_training = (mode == tf.estimator.ModeKeys.TRAIN)

          (total_loss, per_example_loss, logits, probabilities) = create_model(
              bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
              num_labels, use_one_hot_embeddings)

          tvars = tf.trainable_variables()
          initialized_variable_names = {}
          scaffold_fn = None
          if init_checkpoint:
              (assignment_map, initialized_variable_names
               ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
              if use_tpu:

                  def tpu_scaffold():
                      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                      return tf.train.Scaffold()

                  scaffold_fn = tpu_scaffold
              else:
                  tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

          tf.logging.info("**** Trainable Variables ****")
          for var in tvars:
              init_string = ""
              if var.name in initialized_variable_names:
                  init_string = ", *INIT_FROM_CKPT*"
              tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                              init_string)

          output_spec = None
          if mode == tf.estimator.ModeKeys.TRAIN:

              train_op = optimization.create_optimizer(
                  total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

              output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  loss=total_loss,
                  train_op=train_op,
                  scaffold_fn=scaffold_fn)
          elif mode == tf.estimator.ModeKeys.EVAL:

              def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                  predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                  accuracy = tf.metrics.accuracy(
                      labels=label_ids, predictions=predictions, weights=is_real_example)
                  loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                  return {
                      "eval_accuracy": accuracy,
                      "eval_loss": loss,
                  }

              eval_metrics = (metric_fn,
                              [per_example_loss, label_ids, logits, is_real_example])
              output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  loss=total_loss,
                  eval_metrics=eval_metrics,
                  scaffold_fn=scaffold_fn)
          else:
              output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                  mode=mode,
                  predictions={"probabilities": probabilities},
                  scaffold_fn=scaffold_fn)
          return output_spec

      return model_fn

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(args["bert_model"])
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


processor = DataHandler()
label_list = processor.get_labels()
num_labels = len(label_list)
tokenizer = create_tokenizer_from_hub_module()
train_examples = None
num_train_steps = None
if args['do_train']:
    train_examples = processor.get_train_examples(args['full_data_dir'], size=args['train_size'])
    #     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_steps = int(len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps']
                          * args['num_train_epochs'])
# data_c = DataHandler()
# train = data_c.load_train_data()
# train = data_c.add_label_column(train)
# test = data_c.load_test_data()
# test = data_c.add_label_column(test)
#
# DATA_COLUMN = "text_comment"
# LABEL_COLUMN = "label"
#
# # print(train.columns)
#
# # This is a path to an uncased (all lowercase) version of BERT
# BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
#
# # Use the InputExample class from BERT's run_classifier code to create examples from the data
# train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
#                                                                    text_a = x[DATA_COLUMN],
#                                                                    text_b = None,
#                                                                    label = x[LABEL_COLUMN]), axis = 1)
#
# test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None,
#                                                                    text_a = x[DATA_COLUMN],
#                                                                    text_b = None,
#                                                                    label = x[LABEL_COLUMN]), axis = 1)
#
#
# def create_tokenizer_from_hub_module():
#     """Get the vocab file and casing info from the Hub module."""
#     with tf.Graph().as_default():
#         bert_module = hub.Module(BERT_MODEL_HUB)
#         tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
#         with tf.Session() as sess:
#             vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
#                                                   tokenization_info["do_lower_case"]])
#
#     return bert.tokenization.FullTokenizer(
#         vocab_file=vocab_file, do_lower_case=do_lower_case)
#
# # Set sequences to be at most 512 tokens long.
# MAX_SEQ_LENGTH = 512
# # Convert our train and test features to InputFeatures that BERT understands.
# train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
# test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
#
#
# tokenizer = create_tokenizer_from_hub_module()
#
