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
    "num_train_epochs": 4,
    "warmup_proportion": 0.1,
    "local_rank": -1,
    "seed": 42,
    "gradient_accumulation_steps": 1,
    "optimize_on_cpu": False,
    "fp16": False,
    "loss_scale": 128
}

BERT_MODEL_HUB = args["bert_model"]


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
    train_examples = processor.get_train_examples(args['full_data_dir'], size=1000)
    #     train_examples = processor.get_train_examples(args['data_dir'], size=args['train_size'])
    num_train_steps = int(len(train_examples) / args['train_batch_size'] / args['gradient_accumulation_steps']
                          * args['num_train_epochs'])


train_features = processor.convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)

all_input_ids = tf.Variable([f.input_ids for f in train_features], dtype=tf.int32)
all_input_mask = tf.Variable([f.input_mask for f in train_features], dtype=tf.int32)
all_segment_ids = tf.Variable([f.segment_ids for f in train_features], dtype=tf.int32)
all_label_ids = tf.Variable([f.label_ids for f in train_features], dtype=tf.float32)

train_data = tf.data.Dataset.from_tensor_slices((all_input_ids, all_input_mask, all_segment_ids, all_label_ids))
train_data = train_data.batch(batch_size=args["train_batch_size"])

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):

  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)

  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]

  hidden_size = output_layer.shape[-1].value

  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):

    # Dropout helps prevent overfitting
    output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    # log_probs = tf.nn.log_softmax(logits, axis=-1)

    # Convert labels into one-hot encoding
    # one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.int32)

    # predicted_labels = tf.squeeze(tf.argmax(log_probs, axis=-1, output_type=tf.int32))
    # If we're predicting, we want predicted labels and the probabiltiies.
    if is_predicting:
      return logits

    # If we're train/eval, compute loss between predicted and actual label
    per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")
    loss = tf.reduce_mean(per_example_loss)
    return loss


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

args['train_batch_size'] = int(args['train_batch_size'] / args['gradient_accumulation_steps'])

num_warmup_steps = int(num_train_steps * args["warmup_proportion"])


def fit(op="train"):

    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    costs = []  # To keep track of the cost
    epoch_cost = 0

    iterator = train_data.make_initializable_iterator()
    next_batch = iterator.get_next()

    mini_input_ids, mini_input_mask, mini_segment_ids, mini_label_ids = next_batch

    loss = create_model(is_predicting=False, input_ids=mini_input_ids, input_mask=mini_input_mask,
                        segment_ids=mini_segment_ids, labels=mini_label_ids, num_labels=num_labels)

    optimizer = bert.optimization.create_optimizer(loss, init_lr=args["learning_rate"],
                                                   num_train_steps=num_train_steps,
                                                   num_warmup_steps=num_warmup_steps, use_tpu=False)



    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Initialize saver
    saver = tf.train.Saver()

    # Do the training loop
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        if op == 'train':
            # Run the initialization
            sess.run(init)
            sess.run(iterator.initializer)
            for i in range(args["num_train_epochs"]):

                while True:
                    try:

                        # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                        _, minibatch_cost = sess.run([optimizer, loss])
                        epoch_cost += minibatch_cost / args["num_train_epochs"]
                    except tf.errors.OutOfRangeError:
                        break
                print(epoch_cost)
                costs.append(epoch_cost)


fit()