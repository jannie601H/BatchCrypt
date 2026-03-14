import sys, os
import time
import gzip
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
from collections import defaultdict
from collections import deque
# import matplotlib.pyplot as plt
# import pysnooper
import argparse
from functools import reduce

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ftl import augmentation
from ftl.encryption import paillier, encryption
from joblib import Parallel, delayed

os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

# tf.enable_eager_execution()

# from tensorflow import contrib

# tfe = contrib.eager
tfe = tf.keras.metrics


print("TensorFlow version: {}".format(tf.__version__))
# expected tensorflow 1.14
print("Eager execution: {}".format(tf.executing_eagerly()))

# load fashion mnist dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)

def load_fashion_mnist_from_local(path="/root/.keras/datasets"):
    files = {
            "train_images": "train-images-idx3-ubyte.gz",
            "train_labels": "train-labels-idx1-ubyte.gz",
            "test_images":  "t10k-images-idx3-ubyte.gz",
            "test_labels":  "t10k-labels-idx1-ubyte.gz",
            }

    def load_images(filename):
        with gzip.open(os.path.join(path, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)

    def load_labels(filename):
        with gzip.open(os.path.join(path, filename), 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    x_train = load_images(files["train_images"])
    y_train = load_labels(files["train_labels"])
    x_test = load_images(files["test_images"])
    y_test = load_labels(files["test_labels"])
    return (x_train, y_train), (x_test, y_test)

# data loading
#(train_images, train_labels), (test_images, test_labels) = load_fashion_mnist_from_local("/root/.keras/datasets")
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)

train_images = train_images / 255.0
test_images = test_images / 255.0

def build_datasets(num_clients):

    # split_idx = int(len(x_train) / num_clients)
    avg_length = int(len(train_images) / num_clients)
    split_idx = [_ * avg_length for _ in range(1, num_clients)]

    # [train_images_0, train_images_1] = np.split(train_images, [split_idx])
    # [train_labels_0, train_labels_1] = np.split(train_labels, [split_idx])
    x_train_clients = np.split(train_images, split_idx)
    y_train_clients = np.split(train_labels, split_idx)


    # # party A
    # train_dataset_0 = tf.data.Dataset.from_tensor_slices((train_images_0, train_labels_0))
    # # party B
    # train_dataset_1 = tf.data.Dataset.from_tensor_slices((train_images_1, train_labels_1))
    train_dataset_clients = [tf.data.Dataset.from_tensor_slices(item) for item in zip(x_train_clients, y_train_clients)]

    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 100

    # train_dataset_0 = train_dataset_0.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    # train_dataset_1 = train_dataset_1.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    for i in range(len(train_dataset_clients)):
        train_dataset_clients[i] = train_dataset_clients[i].shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

    return train_dataset_clients


# build the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()
# predictions = model(features)
# print(predictions[:5])
cce = tf.keras.losses.SparseCategoricalCrossentropy()


def loss(model, x, y):
    y_ = model(x)
    return cce(y_true=y, y_pred=y_)


# l = loss(model, features, labels)
# print("Loss test: {}".format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# optimizer = tf.train.AdamOptimizer(learning_rate=0.005)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
global_step = tf.Variable(0)


def clip_gradients(grads, min_v, max_v):
    results = [tf.clip_by_value(t, min_v, max_v) for t in grads]
    return results


def do_sum(x1, x2):
    results = []
    for i in range(len(x1)):
        results.append(x1[i] + x2[i])
    return results


def aggregate_gradients(gradient_list):
    results = reduce(do_sum, gradient_list)
    return results


def aggregate_losses(loss_list):
    return np.sum(loss_list)


def quantize(party, bit_width=16):
    result = []
    for component in party:
        x, _ = encryption.quantize_matrix(component, bit_width=bit_width)
        result.append(x)
    return result


def quantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        x, _ = encryption.quantize_matrix_stochastic(component, bit_width=bit_width, r_max=r_max)
        result.append(x)
    return result


def unquantize(party, bit_width=16, r_max=0.5):
    result = []
    for component in party:
        if isinstance(component, tf.Tensor):
            component = component.numpy()
        result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    return result


def unquantize_per_layer(party, r_maxs, bit_width=16):
    result = []
    for component, r_max in zip(party, r_maxs):
        if isinstance(component, tf.Tensor):
            component = component.numpy()
        result.append(encryption.unquantize_matrix(component, bit_width=bit_width, r_max=r_max).astype(np.float32))
    return result

if __name__ == '__main__':
    seed = 123
    # tf.random.set_random_seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    parser = argparse.ArgumentParser()
    # parser.add_argument('--experiment', type=str, required=True,
    #                     choices=["plain", "batch", "only_quan", "aciq_quan"])
    parser.add_argument('--experiment', type=str, default="batch_zero",
                        choices=["plain", "batch", "only_quan", "aciq_quan", "clip_quan", "batch_zero"])
    parser.add_argument('--num_clients', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--q_width', type=int, default=8)
    parser.add_argument('--clip', type=float, default=0.3)
    args = parser.parse_args()

    options = vars(args)
    output_name = "fmnist_" + "_".join([ "{}_{}".format(key, options[key]) for key in options ])

    log_dir = os.path.join("logs", output_name)
    os.makedirs(log_dir, exist_ok=True)

    # keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    test_loss_results = []
    test_accuracy_results = []
    enc_grads = []
    enc_grads_shape = []

    num_epochs  = args.num_epochs
    clip        = args.clip
    num_clients = args.num_clients
    q_width     = args.q_width

    # this key pair should be shared by party A and B
    publickey, privatekey = paillier.PaillierKeypair.generate_keypair(n_length=2048)

    loss_array = []
    accuracy_array = []
    epoch_time_array = []
    sparsity_array_per_layer = []
    # clip_thresholds_array = []
    # rmax_array = []
    # comm_log_array = []
    bc_skip_log = []

    total_train_start = time.time()

    ENCRYPTED_ZERO = publickey.encrypt(0) ## zero encrypted constant value

    for epoch in range(num_epochs):
        batch_idx = 0
        # epoch_loss_avg = tfe.metrics.Mean()
        epoch_loss_avg = tf.keras.metrics.Mean()
        # epoch_accuracy = tfe.metrics.Accuracy()
        epoch_accuracy = tf.keras.metrics.Accuracy()

        train_dataset_clients = build_datasets(num_clients)

        for data_clients in zip(*train_dataset_clients):
            batch_idx += 1
            print("{} clients are in federated training".format(len(data_clients)))
            loss_batch_clients = []
            grads_batch_clients = []

            start_t = time.time()

            # calculate loss and grads locally
            for x, y in data_clients:
                loss_temp, grads_temp = grad(model, x, y)
                loss_batch_clients.append(loss_temp.numpy())
                grads_batch_clients.append([x.numpy() for x in grads_temp])

            # federated_lr_plain.py
            if args.experiment == "plain":
                # NOTE: The clip value here is "1" in federated_lr_plain.py
                # grads_0 = clip_gradients(grads_0, -1 * clip, clip)
                # grads_1 = clip_gradients(grads_1, -1 * clip, clip)

                # in plain version, no clipping before applying
                # grads_batch_clients = [clip_gradients(item, -1 * clip, clip) 
                #                         for item in grads_batch_clients]


                client_weight = 1.0 / num_clients
                start = time.time()
                grads      = aggregate_gradients(grads_batch_clients)
                end_enc = time.time()
                print("aggregation finished in %f" % (end_enc - start))
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

            # federated_lr_batch_zero.py
            elif args.experiment == "batch_zero":
                grads_batch_clients = [
                    clip_gradients(item, -clip / num_clients, clip / num_clients)
                    for item in grads_batch_clients
                ]

                for c_idx in range(num_clients):
                    for layer_idx in range(len(model.trainable_variables)):
                        g_np = grads_batch_clients[c_idx][layer_idx]
                        # print(f"[DEBUG - clipping] client={c_idx}, layer={layer_idx}, max(|g|) after clip = {np.max(np.abs(g_np)):.6f}")

                num_components = len(model.trainable_variables)
                r_maxs = [clip * 1.1 for _ in range(num_components)]  # ★ overflow 여유

                enc_grads_batch_clients = []
                enc_grads_shape_batch_clients = []
                enc_grads_skip_mask_batch_clients = []

                for c_idx in range(num_clients):
                    enc_client = []
                    shape_client = []
                    mask_client = []
                    for layer_idx in range(num_components):
                        arr = grads_batch_clients[c_idx][layer_idx].numpy()
                        enc_blocks, og_shape, skip_mask_blocks = encryption.encrypt_matrix_batch_zero(
                            publickey,
                            arr,
                            epoch,
                            batch_size=16,
                            bit_width=q_width,
                            r_max=r_maxs[layer_idx]
                        )
                        enc_client.append(enc_blocks)
                        shape_client.append(og_shape)
                        mask_client.append(skip_mask_blocks)
                    enc_grads_batch_clients.append(enc_client)
                    enc_grads_shape_batch_clients.append(shape_client)
                    enc_grads_skip_mask_batch_clients.append(mask_client)

                enc_losses = [publickey.encrypt(l) for l in loss_batch_clients]
                agg_enc_loss = reduce(lambda a, b: a + b, enc_losses)

                agg_enc_grads = []
                for layer_idx in range(num_components):
                    n_blocks = max(
                        len(enc_grads_skip_mask_batch_clients[c][layer_idx])
                        for c in range(num_clients)
                    )

                    comp_blocks = []
                    # 클라이언트별로 enc_blocks는 skip_mask=True에 해당하는 것만 있음 → pop(0) 사용
                    enc_block_ptrs = [deque(enc_grads_batch_clients[c][layer_idx]) for c in range(num_clients)]

                    for b in range(n_blocks):
                        blocks_to_sum = []
                        for c in range(num_clients):
                            skip_mask = enc_grads_skip_mask_batch_clients[c][layer_idx]
                            if b < len(skip_mask) and skip_mask[b]:
                                blocks_to_sum.append(enc_block_ptrs[c].popleft())
                            else:
                                blocks_to_sum.append(ENCRYPTED_ZERO)
                        summed = reduce(lambda a, b: a + b, blocks_to_sum)
                        comp_blocks.append(summed)
                    agg_enc_grads.append(comp_blocks)

                loss_value = privatekey.decrypt(agg_enc_loss)

                # skip_masks_per_comp = []
                # for layer_idx in range(num_components):
                #     merged = np.logical_or.reduce([
                #         enc_grads_skip_mask_batch_clients[c][layer_idx]
                #         for c in range(num_clients)
                #     ])
                #     merged = merged[:len(agg_enc_grads[layer_idx])]  # 정확히 맞춤
                #     skip_masks_per_comp.append(merged)

                grads = []
                for layer_idx in range(num_components):
                    comp_blocks = agg_enc_grads[layer_idx]
                    og_shape = enc_grads_shape_batch_clients[0][layer_idx]

                    plain = encryption.decrypt_matrix_batch_zero(
                        privatekey,
                        comp_blocks,
                        og_shape,
                        batch_size=16,
                        bit_width=q_width,
                        r_max=r_maxs[layer_idx]
                    )
                    grads.append(plain)

            # federated_lr_batch.py
            elif args.experiment == "batch":
                # # party A
                # grads_0 = clip_gradients(grads_0, -1 * clip / num_clients, clip / num_clients)
                # # party B
                # grads_1 = clip_gradients(grads_1, -1 * clip / num_clients, clip / num_clients)
                grads_batch_clients = [clip_gradients(item, -1 * clip / num_clients, clip / num_clients) 
                                        for item in  grads_batch_clients]

                for c_idx in range(num_clients):
                    for layer_idx in range(len(model.trainable_variables)):
                        g_np = grads_batch_clients[c_idx][layer_idx]
                        # print(f"[DEBUG - clipping] client={c_idx}, layer={layer_idx}, max(|g|) after clip = {np.max(np.abs(g_np)):.6f}")

                # # party A
                # enc_grads_0 = []
                # enc_grads_shape_0 = []
                # for component in grads_0:
                #     enc_g, enc_g_s = encryption.encrypt_matrix_batch(publickey, component.numpy(),
                #                                                      bit_width=q_width, r_max=clip)
                #     enc_grads_0.append(enc_g)
                #     enc_grads_shape_0.append(enc_g_s)
                # loss_value_0 = encryption.encrypt(publickey, loss_value_0)
                # # party B
                # enc_grads_1 = []
                # enc_grads_shape_1 = []
                # for component in grads_1:
                #     enc_g, enc_g_s = encryption.encrypt_matrix_batch(publickey, component.numpy(),
                #                                                      bit_width=q_width, r_max=clip)
                #     enc_grads_1.append(enc_g)
                #     enc_grads_shape_1.append(enc_g_s)
                # loss_value_1 = encryption.encrypt(publickey, loss_value_1)
                enc_grads_batch_clients = []
                enc_grads_shape_batch_clients = []
                for grad_client in grads_batch_clients:
                    enc_grads_client = []
                    enc_grads_shape_client = []
                    for component in grad_client:
                        enc_g, enc_g_s = encryption.encrypt_matrix_batch(publickey, component.numpy(),
                                                                         bit_width=q_width, r_max=clip)
                        enc_grads_client.append(enc_g)
                        enc_grads_shape_client.append(enc_g_s)

                    enc_grads_batch_clients.append(enc_grads_client)
                    enc_grads_shape_batch_clients.append(enc_grads_shape_client)

                loss_batch_clients = [encryption.encrypt(publickey, item) for item in loss_batch_clients] 

                # arbiter aggregate gradients
                # enc_grads = aggregate_gradients([enc_grads_0, enc_grads_1])
                # loss_value = aggregate_losses([loss_value_0, loss_value_1])
                enc_grads = aggregate_gradients(enc_grads_batch_clients)
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                # on party A and B individually
                loss_value = encryption.decrypt(privatekey, loss_value)
                grads = []
                for i in range(len(enc_grads)):
                    # plain_g = encryption.decrypt_matrix_batch(privatekey, enc_grads_0[i], enc_grads_shape_0[i])
                    plain_g = encryption.decrypt_matrix_batch(privatekey, enc_grads[i], enc_grads_shape_batch_clients[0][i])
                    grads.append(plain_g)

            # federated_lr_only_quan.py
            elif args.experiment == "only_quan":
                for idx in range(len(grads_batch_clients)):
                    grads_batch_clients[idx] = [x.numpy() for x in grads_batch_clients[idx]]

                # clipping_thresholds = encryption.calculate_clip_threshold(grads_0)
                theta = 2.5
                grads_batch_clients_mean = []
                grads_batch_clients_mean_square = []
                for client_idx in range(len(grads_batch_clients)):
                    temp_mean = [np.mean(grads_batch_clients[client_idx][layer_idx])
                                 for layer_idx in range(len(grads_batch_clients[client_idx]))]
                    temp_mean_square = [np.mean(grads_batch_clients[client_idx][layer_idx] ** 2)
                                        for layer_idx in range(len(grads_batch_clients[client_idx]))]
                    grads_batch_clients_mean.append(temp_mean)
                    grads_batch_clients_mean_square.append(temp_mean_square)
                grads_batch_clients_mean = np.array(grads_batch_clients_mean)
                grads_batch_clients_mean_square = np.array(grads_batch_clients_mean_square)

                layers_size = np.array([_.size for _ in grads_batch_clients[0]])
                clipping_thresholds = theta * (
                            np.sum(grads_batch_clients_mean_square * layers_size, 0) / (layers_size * num_clients)
                            - (np.sum(grads_batch_clients_mean * layers_size, 0) / (layers_size * num_clients)) ** 2) ** 0.5

                print("clipping_thresholds", clipping_thresholds)

                # r_maxs = [x * 2 for x in clipping_thresholds]
                r_maxs = [x * num_clients for x in clipping_thresholds]

             	# grads_0 = encryption.clip_with_threshold(grads_0, clipping_thresholds)
            	# grads_1 = encryption.clip_with_threshold(grads_1, clipping_thresholds)
            	# grads_0 = quantize_per_layer(grads_0, r_maxs, bit_width=q_width)
            	# grads_1 = quantize_per_layer(grads_1, r_maxs, bit_width=q_width)
                grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                       for item in grads_batch_clients]

                grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=q_width) 
                                       for item in grads_batch_clients]

                # grads = aggregate_gradients([grads_0, grads_1], weight=0.5)
                # loss_value = aggregate_losses([0.5 * loss_value_0, 0.5 * loss_value_1])

               # grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)
                client_weight = 1.0 / num_clients
                grads      = aggregate_gradients(grads_batch_clients)
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)

            elif args.experiment == "aciq_quan":
                sizes = [tf.size(item).numpy() * num_clients for item in grads_batch_clients[0]]
                max_values = []
                min_values = []
                for layer_idx in range(len(grads_batch_clients[0])):
                    max_values.append([np.max([item[layer_idx] for item in grads_batch_clients])])
                    min_values.append([np.min([item[layer_idx] for item in grads_batch_clients])])
                grads_max_min = np.concatenate([np.array(max_values),np.array(min_values)], axis=1)
                clipping_thresholds = encryption.calculate_clip_threshold_aciq_g(grads_max_min, sizes, bit_width=q_width)

                r_maxs = [x * num_clients for x in clipping_thresholds]
                grads_batch_clients = [encryption.clip_with_threshold(item, clipping_thresholds)
                                       for item in grads_batch_clients]
                grads_batch_clients = [quantize_per_layer(item, r_maxs, bit_width=q_width)
                                       for item in grads_batch_clients]

                grads = aggregate_gradients(grads_batch_clients)
                client_weight = 1.0 / num_clients
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                grads = unquantize_per_layer(grads, r_maxs, bit_width=q_width)

            elif args.experiment == "clip_quan":
                grads_batch_clients = [
                        clip_gradients(item, -1 * clip / num_clients, clip / num_clients)
                        for item in grads_batch_clients
                        ]

                grads_batch_clients = [
                        quantize(item, bit_width=q_width)
                        for item in grads_batch_clients
                        ]

                # aggregation
                client_weight = 1.0 / num_clients
                grads = aggregate_gradients(grads_batch_clients)
                loss_value = aggregate_losses([item * client_weight for item in loss_batch_clients])

                grads = unquantize(grads, bit_width=q_width, r_max=clip)

            ######
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            for i, g in enumerate(grads):
                print(f"Layer {i} grad max: {np.max(g):.6f}, min: {np.min(g):.6f}")

            # Track progress
            epoch_loss_avg(loss_value)  # add current batch loss
            # compare predicted label to actual label
            # epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)
            epoch_accuracy(tf.argmax(model(test_images), axis=1, output_type=tf.int32),
                           test_labels)

            loss_array.append(loss_value)

            elapsed_time = time.time() - start_t
            epoch_time_array.append(elapsed_time)

            # gradient sparsity per layer
            zero_ratio_per_layer = [np.mean(g == 0) for g in grads]
            sparsity_array_per_layer.append(zero_ratio_per_layer)
            print("Gradient sparsity per layer: ", zero_ratio_per_layer)

            # skip ratio log
            total_skipped = 0
            total_blocks = 0
            for c in range(num_clients):
                for layer_idx in range(num_components):
                    skip_mask = enc_grads_skip_mask_batch_clients[c][layer_idx]
                    total_skipped += np.sum(~skip_mask)  # False = skipped
                    total_blocks += len(skip_mask)
            skip_ratio = total_skipped / total_blocks if total_blocks > 0 else 0
            bc_skip_log.append(skip_ratio)
            print(f"skip ratio: {skip_ratio * 100:.2f}% ({total_skipped}/{total_blocks})")

            # save clipping threasholds, r_maxs
            # if args.experiment in ["batch"]:
            #     clip_thresholds_array.append(clipping_thresholds)
            #     rmax_array.append(r_maxs)
            #     print("Clipping thresholds per layer: ", clipping_thresholds)
            #     print("r_maxs per layer: ", r_maxs)

            accuracy_value = epoch_accuracy.result().numpy()
            accuracy_array.append(accuracy_value)

            elapsed_time = time.time() - start_t
            print( "loss: {} \taccuracy: {} \telapsed time: {}".format(loss_value, accuracy_value, elapsed_time) )
            
        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        test_loss_v = loss(model, test_images, test_labels)
        test_accuracy_v = accuracy_score(test_labels, tf.argmax(model(test_images), axis=1, output_type=tf.int32))
        test_loss_results.append(test_loss_v)
        test_accuracy_results.append(test_accuracy_v)

        if epoch % 1 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))
        model.save_weights("model_{}_e{:03d}.weights.h5".format(output_name, epoch))
        print("Saved model to disk")

    np.savetxt(os.path.join(log_dir, 'train_loss.txt'), train_loss_results)
    np.savetxt(os.path.join(log_dir, 'train_accuracy.txt'), train_accuracy_results)
    np.savetxt(os.path.join(log_dir, 'test_loss.txt'), test_loss_results)
    np.savetxt(os.path.join(log_dir, 'test_accuracy.txt'), test_accuracy_results)
    np.savetxt(os.path.join(log_dir, 'epoch_time.txt'), epoch_time_array)
    np.savetxt(os.path.join(log_dir, 'sparsity.txt'), sparsity_array_per_layer)
    np.savetxt(os.path.join(log_dir, 'skip_ratio.txt'), bc_skip_log)
    
    # serialize model to JSON
#    model_json = model.to_json()
#    with open("model_{}.json".format(output_name), "w") as json_file:
#        json_file.write(model_json)
    # serialize weights to HDF5
#    model.save_weights(os.path.join(log_dir, "model.h5"))
#    print("Saved model to disk")

# save total time
total_train_time = time.time() - total_train_start
print("Total training time: {:.2f} seconds".format(total_train_time))
with open(os.path.join(log_dir, "total_time.txt"), "w") as f:
    f.write(str(total_train_time))

# fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
# fig.suptitle('Training Metrics')

# axes[0].set_ylabel("Loss", fontsize=14)
# axes[0].plot(train_loss_results)

# axes[1].set_ylabel("Accuracy", fontsize=14)
# axes[1].set_xlabel("Epoch", fontsize=14)
# axes[1].plot(train_accuracy_results)
# plt.show()
