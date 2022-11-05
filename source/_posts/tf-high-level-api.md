---
title: TensorFlow Higher-Level APIs的使用
tags:
  - machine-learning
  - deep-learning
  - tensorflow
date: 2019-01-05 16:22:16
---

##  [Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator), [Experiment](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment), and [Dataset](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset)的关系

![Overview of the Experiment, Estimator and DataSet framework and how they interact. (These components will be explained in the following sections)](1*zoNZvvuJb06yAghetc6BfQ.png)

<!-- more -->

## Estimator定义

The [**Estimator**](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) class represents a model, as well as how this model should be trained and evaluated. We can create an estimator as follows:

```python
return tf.estimator.Estimator(
    model_fn=model_fn,  # First-class function
    params=params,  # HParams
    config=run_config  # RunConfig
)
```

### 参数解释

To create the Estimator we need to pass in a model function, a collection of parameters and some configuration.

*   The **parameters** should be a collection of the model’s hyperparameters. This can be a dictionary, but we will represent it in this example as an [HParams](https://www.tensorflow.org/api_docs/python/tf/contrib/training/HParams) object, which acts as a [namedtuple](https://docs.python.org/2/library/collections.html#collections.namedtuple).
*   The **configuration** specifies how the **training and evaluation** are run, and where to store the results. This configuration will be represented by a [RunConfig](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/RunConfig) object, which communicates everything the Estimator needs to know about the environment in which the model will be run.
*   The **model function** is a Python function, which builds the model given the input. (More on this later)

#### Model function

The model function is a Python function which is passed as a [first-class function](https://en.wikipedia.org/wiki/First-class_function) to the Estimator. We’ll see later that TensorFlow uses first-class functions in other places. The benefit of representing the model as a function is that the model can be recreated over and over by instantiating the function. The model can be recreated during the training with different input, for example, to run validation tests during training.

The model function takes the **input features** as parameters and the corresponding **labels** as tensors. It also takes a [**mode**](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/ModeKeys) that signals if the model is training, evaluating or performing inference. The last parameter to the model function should be a collection of **hyperparameters**, which are the same as those passed to the Estimator. This model function should return an [**EstimatorSpec**](https://www.tensorflow.org/api_docs/python/tf/estimator/EstimatorSpec) object which will define the complete model.

The EstimatorSpec takes in the prediction, loss, training and evaluation [Operations](https://www.tensorflow.org/api_docs/python/tf/Operation) so it defines the full model graph used for training, evaluation, and inference. Because the EstimatorSpec just takes in regular TensorFlow Operations, we can use frameworks like [TF-Slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) to define our model.


## Estimator创建方式

1. 预创建的 Estimator
2. 自定义 Estimator
3. 从 Keras 模型创建 Estimator

### 3从 Keras 模型创建 Estimator的例子：
```python
# Instantiate a Keras inception v3 model.
keras_inception_v3 = tf.keras.applications.inception_v3.InceptionV3(weights=None)
# Compile model with the optimizer, loss, and metrics you'd like to train with.
keras_inception_v3.compile(optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
                          loss='categorical_crossentropy',
                          metric='accuracy')
# Create an Estimator from the compiled Keras model. Note the initial model
# state of the keras model is preserved in the created Estimator.
est_inception_v3 = tf.keras.estimator.model_to_estimator(keras_model=keras_inception_v3)

# Treat the derived Estimator as you would with any other Estimator.
# First, recover the input name(s) of Keras model, so we can use them as the
# feature column name(s) of the Estimator input function:
keras_inception_v3.input_names  # print out: ['input_1']
# Once we have the input name(s), we can create the input function, for example,
# for input(s) in the format of numpy ndarray:
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1": train_data},
    y=train_labels,
    num_epochs=1,
    shuffle=False)
# To train, we call Estimator's train function:
est_inception_v3.train(input_fn=train_input_fn, steps=2000)
```

### 这样做的好处

Keras 模型本身容易建立，导出到Estimator后又可以利用 Estimator 的优势，例如分布式训练。

## Estimator 的优势

### 从意义角度

  * 环境适配性好。可以在本地主机、分布式多服务器环境中运行基于 Estimator 的模型，而无需更改模型。此外，可以在 CPU、GPU 或 TPU 上运行基于 Estimator 的模型，而无需重新编码模型。
  * 简化了在模型开发者之间共享实现的过程。
  * 采用 Estimator 创建模型通常比采用低阶 TensorFlow API 更简单。
  * 其本身在 [`tf.layers`](https://www.tensorflow.org/api_docs/python/tf/layers?hl=zh-cn) 之上构建而成，可以简化自定义过程。
  * Estimator 会自动构建图。
  * Estimator 提供安全的分布式训练循环，可以控制如何以及何时：
    * 构建图
    * 初始化变量
    * 开始排队
    * 处理异常
    * 创建检查点文件并从故障中恢复
    * 保存 TensorBoard 的摘要

### 从使用角度

*   **学习流程**：Estimator 封装了对机器学习不同阶段的控制，用户无需不断的为新机器学习任务重复编写训练、评估、预测的代码。可以专注于对网络结构的控制。
*   **网络结构**：Estimator 的网络结构是在 model_fn 中独立定义的，用户创建的任何网络结构都可以在 Estimator 的控制下进行机器学习。这可以允许用户很方便的使用别人定义好的 model_fn。
*   **数据导入**：Estimator 的数据导入也是由 input_fn 独立定义的。例如，用户可以非常方便的只通过改变 input_fn 的定义，来使用相同的网络结构学习不同的数据。

使用 Estimator 编写应用时，您必须将数据输入管道从模型中分离出来。这种分离简化了不同数据集的实验流程。

## 预创建的 Estimator 程序的结构

依赖预创建的 Estimator 的 TensorFlow 程序通常包含下列四个步骤：

1.  **编写一个或多个数据集导入函数。** 例如，您可以创建一个函数来导入训练集，并创建另一个函数来导入测试集。每个数据集导入函数都必须返回两个对象：

    *   一个字典，其中键是特征名称，值是包含相应特征数据的张量（或 SparseTensor）
    *   一个包含一个或多个标签的张量

```python
def input_fn(dataset):
   ...  # manipulate dataset, extracting the feature dict and the label
   return feature_dict, label
```

2.  **定义特征列。** 每个 [`tf.feature_column`](https://www.tensorflow.org/api_docs/python/tf/feature_column?hl=zh-cn) 都标识了特征名称、特征类型和任何输入预处理操作。例如，以下代码段创建了三个存储整数或浮点数据的特征列。前两个特征列仅标识了特征的名称和类型。第三个特征列还指定了一个 lambda，该程序将调用此 lambda 来调节原始数据：

```python
# Define three numeric feature columns.
population = tf.feature_column.numeric_column('population')
crime_rate = tf.feature_column.numeric_column('crime_rate')
median_education = tf.feature_column.numeric_column('median_education',
                    normalizer_fn=lambda x: x - global_education_mean)
```

3.  **实例化相关的预创建的 Estimator。** 例如，下面是对名为 `LinearClassifier` 的预创建 Estimator 进行实例化的示例代码：

```python
# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.LinearClassifier(
    feature_columns=[population, crime_rate, median_education],
    )
```

4.  **调用训练、评估或推理方法。**例如，所有 Estimator 都提供训练模型的 `train` 方法。
```python
 # my_training_set is the function created in Step 1 estimator.train(input_fn=my_training_set, steps=2000)
```

## Experiment

The [**Experiment**](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Experiment) class defines how to train a model and integrates nicely with the Estimator. We can create an experiment as follows:

```python
experiment = tf.contrib.learn.Experiment(
    estimator=estimator,  # Estimator
    train_input_fn=train_input_fn,  # First-class function
    eval_input_fn=eval_input_fn,  # First-class function
    train_steps=params.train_steps,  # Minibatch steps
    min_eval_frequency=params.min_eval_frequency,  # Eval frequency
    train_monitors=[train_input_hook],  # Hooks for training
    eval_hooks=[eval_input_hook],  # Hooks for evaluation
    eval_steps=None  # Use evaluation feeder until its empty
)
view raw
```

The Experiment takes as input:

  * An **estimator** (for example the one we defined above).
  * **Train and evaluation data** as a [first-class function](https://en.wikipedia.org/wiki/First-class_function). The same concept as the model function explained earlier is used here. By passing in a function instead of operation, the input graph can be recreated if needed. We’ll talk more about this later.
  * [**Training and Evaluating hooks**](https://www.tensorflow.org/api_guides/python/train#Training_Hooks). These hooks can be used to save or monitor specific things, or to set up certain operations in the Graph or Session. For example, we will be passing in operations to help initialize the data loaders (again, more later).
  * Various parameters describing how long to train for and when to evaluate.

Once we have defined the experiment, we can run it to train and evaluate the model with [learn_runner.run](http://tf.contrib.learn.learn_runner.run/) as follows:

```python
learn_runner.run(
    experiment_fn=experiment_fn,  # First-class function
    run_config=run_config,  # RunConfig
    schedule="train_and_evaluate",  # What to run
    hparams=params  # HParams
)
```


Like the model function and the data functions, the learn runner takes in the function that creates the experiment as a parameter.

### Dataset

We’ll be using the [**Dataset**](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset) class and the corresponding [**Iterator**](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Iterator) to represent our training and evaluation data, and to create data feeders that iterate over the data during training. In this example, we will use the [MNIST](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist) data that’s available in Tensorflow, and build a Dataset wrapper around it. For example, we will represent the training input data as:

```python
# Define the training inputs
def get_train_inputs(batch_size, mnist_data):
    """Return the input function to get the training data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Training_data'):
            # Get Mnist data
            images = mnist_data.train.images.reshape([-1, 28, 28, 1])
            labels = mnist_data.train.labels
            # Define placeholders
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return train_inputs, iterator_initializer_hook
```

Calling this _get_train_inputs_ will return a [first-class function](https://en.wikipedia.org/wiki/First-class_function) that creates the data loading operations in a TensorFlow graph, together with a Hook to initialize the iterator.

The MNIST data used in this example is initially represented as a [Numpy array](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html). We create a [placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) tensor that gets the data fed in; we use a placeholder in order to avoid copying the data. Next, we create a sliced dataset with the help of _from_tensor_slices._ We will make sure that this dataset runs for an infinite amount of epochs (the experiment can take care of limiting the number of epochs), and that the data gets shuffled and put into batches of the required size.

To iterate over the data we need to create an iterator from the dataset. Because we are using a placeholder we need to initialize the placeholder in the relevant session with the NumPy data. We can do this by creating an [initializable iterator](https://www.tensorflow.org/api_docs/python/tf/contrib/data/Dataset#make_initializable_iterator). We will create a custom defined _IteratorInitializerHook _object to initialize the iterator when the graph is created:

```python
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)
```

The _IteratorInitializerHook_ inherits from [**SessionRunHook**](https://www.tensorflow.org/api_docs/python/tf/train/SessionRunHook). This hook will call _after_create_session_ as soon as the relevant session is created, and initialize the placeholder with the right data. This hook is returned by our _get_train_inputs_ function and will be passed to the Experiment object upon creation.

The data loading operations returned by the _train_inputs_ function are TensorFlow [operations](https://www.tensorflow.org/api_docs/python/tf/Operation) that will return a new batch every time they are evaluated.



## 引用

1. [Higher-Level APIs in TensorFlow](https://medium.com/onfido-tech/higher-level-apis-in-tensorflow-67bfb602e6c0)
2. [TensorFlow Estimator](https://www.tensorflow.org/guide/estimators?hl=zh-cn)
3. [如何使用TensorFlow中的高级API：Estimator、Experiment和Dataset](https://juejin.im/post/59b4cc816fb9a00a6974c5a3)
4. [TensorFlow高层API：Custom Estimator建立CNN+RNN](https://zhuanlan.zhihu.com/p/33681224)
5. There is a paper called [“TensorFlow Estimators: Managing Simplicity vs. Flexibility in High-Level Machine Learning Frameworks”](https://terrytangyuan.github.io/data/papers/tf-estimators-kdd-paper.pdf) describing the high level-design of the Estimator framework.
6. [TensorFlow has more documentation on using the Dataset API](https://www.tensorflow.org/versions/r1.3/programmers_guide/datasets).
7. There are 2 versions of the Estimator class. We are using the one at [tf.estimator.Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) in this example, but there is also an older unstable version at [tf.contrib.learn.Estimator](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Estimator).
8. There are also 2 versions of the RunConfig class. While we are using the one at [tf.contrib.learn.RunConfig](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/RunConfig) there is also a version at [tf.estimator.RunConfig](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig). I couldn’t get the latter one to work with the Experiment framework so I stuck with the tf.contrib version.
9. While we didn’t use them in this example, the Estimator framework defines predefined estimators for typical models such as [classifiers](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier) and [regressors](https://www.tensorflow.org/api_docs/python/tf/estimator/DNNRegressor). These predefined estimators are easy to use and come with a [detailed tutorial](https://www.tensorflow.org/extend/estimators).
10. TensorFlow also defines an abstraction for the “head” of a model, the part that sits on top of the architecture and defines the loss, evaluation and training operations. This head will take care of things like defining the model function, and all the required Operations. You can find a version at [tf.contrib.learn.Head](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/Head). There is also a [prototype version](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/estimator/canned/head.py) in the newer estimator framework. We decided not to use it in this example due to its development being quite unstable.
11. This blog uses the TensorFlow [slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim) framework to define the architecture of the model. Slim is a lightweight library for defining complex models in tensorflow. They also define [pre-defined architectures and pre-trained models](https://github.com/tensorflow/models/tree/master/slim).


## 示例代码

```python

"""Script to illustrate usage of tf.estimator.Estimator in TF v1.3"""
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.contrib import slim
from tensorflow.contrib.learn import ModeKeys
from tensorflow.contrib.learn import learn_runner


# Show debugging output
tf.logging.set_verbosity(tf.logging.DEBUG)

# Set default flags for the output directories
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    flag_name='model_dir', default_value='./mnist_training',
    docstring='Output directory for model and training stats.')
tf.app.flags.DEFINE_string(
    flag_name='data_dir', default_value='./mnist_data',
    docstring='Directory to download the data to.')


# Define and run experiment ###############################
def run_experiment(argv=None):
    """Run the training experiment."""
    # Define model parameters
    params = tf.contrib.training.HParams(
        learning_rate=0.002,
        n_classes=10,
        train_steps=5000,
        min_eval_frequency=100
    )

    # Set the run_config and the directory to save the model and stats
    run_config = tf.contrib.learn.RunConfig()
    run_config = run_config.replace(model_dir=FLAGS.model_dir)

    learn_runner.run(
        experiment_fn=experiment_fn,  # First-class function
        run_config=run_config,  # RunConfig
        schedule="train_and_evaluate",  # What to run
        hparams=params  # HParams
    )


def experiment_fn(run_config, params):
    """Create an experiment to train and evaluate the model.
    Args:
        run_config (RunConfig): Configuration for Estimator run.
        params (HParam): Hyperparameters
    Returns:
        (Experiment) Experiment for training the mnist model.
    """
    # You can change a subset of the run_config properties as
    run_config = run_config.replace(
        save_checkpoints_steps=params.min_eval_frequency)
    # Define the mnist classifier
    estimator = get_estimator(run_config, params)
    # Setup data loaders
    mnist = mnist_data.read_data_sets(FLAGS.data_dir, one_hot=False)
    train_input_fn, train_input_hook = get_train_inputs(
        batch_size=128, mnist_data=mnist)
    eval_input_fn, eval_input_hook = get_test_inputs(
        batch_size=128, mnist_data=mnist)
    # Define the experiment
    experiment = tf.contrib.learn.Experiment(
        estimator=estimator,  # Estimator
        train_input_fn=train_input_fn,  # First-class function
        eval_input_fn=eval_input_fn,  # First-class function
        train_steps=params.train_steps,  # Minibatch steps
        min_eval_frequency=params.min_eval_frequency,  # Eval frequency
        train_monitors=[train_input_hook],  # Hooks for training
        eval_hooks=[eval_input_hook],  # Hooks for evaluation
        eval_steps=None  # Use evaluation feeder until its empty
    )
    return experiment


# Define model ############################################
def get_estimator(run_config, params):
    """Return the model as a Tensorflow Estimator object.
    Args:
         run_config (RunConfig): Configuration for Estimator run.
         params (HParams): hyperparameters.
    """
    return tf.estimator.Estimator(
        model_fn=model_fn,  # First-class function
        params=params,  # HParams
        config=run_config  # RunConfig
    )


def model_fn(features, labels, mode, params):
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    is_training = mode == ModeKeys.TRAIN
    # Define model's architecture
    logits = architecture(features, is_training=is_training)
    predictions = tf.argmax(logits, axis=-1)
    # Loss, training and eval operations are not needed during inference.
    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.INFER:
        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=tf.cast(labels, tf.int32),
            logits=logits)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


def get_train_op_fn(loss, params):
    """Get the training Op.
    Args:
         loss (Tensor): Scalar Tensor that represents the loss function.
         params (HParams): Hyperparameters (needs to have `learning_rate`)
    Returns:
        Training Op
    """
    return tf.contrib.layers.optimize_loss(
        loss=loss,
        global_step=tf.contrib.framework.get_global_step(),
        optimizer=tf.train.AdamOptimizer,
        learning_rate=params.learning_rate
    )


def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'Accuracy': tf.metrics.accuracy(
            labels=labels,
            predictions=predictions,
            name='accuracy')
    }


def architecture(inputs, is_training, scope='MnistConvNet'):
    """Return the output operation following the network architecture.
    Args:
        inputs (Tensor): Input Tensor
        is_training (bool): True iff in training mode
        scope (str): Name of the scope of the architecture
    Returns:
         Logits output Op for the network.
    """
    with tf.variable_scope(scope):
        with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                weights_initializer=tf.contrib.layers.xavier_initializer()):
            net = slim.conv2d(inputs, 20, [5, 5], padding='VALID',
                              scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')
            net = slim.conv2d(net, 40, [5, 5], padding='VALID',
                              scope='conv3')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool4')
            net = tf.reshape(net, [-1, 4 * 4 * 40])
            net = slim.fully_connected(net, 256, scope='fn5')
            net = slim.dropout(net, is_training=is_training,
                               scope='dropout5')
            net = slim.fully_connected(net, 256, scope='fn6')
            net = slim.dropout(net, is_training=is_training,
                               scope='dropout6')
            net = slim.fully_connected(net, 10, scope='output',
                                       activation_fn=None)
        return net


# Define data loaders #####################################
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


# Define the training inputs
def get_train_inputs(batch_size, mnist_data):
    """Return the input function to get the training data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def train_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Training_data'):
            # Get Mnist data
            images = mnist_data.train.images.reshape([-1, 28, 28, 1])
            labels = mnist_data.train.labels
            # Define placeholders
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.repeat(None)  # Infinite iterations
            dataset = dataset.shuffle(buffer_size=10000)
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            # Return batched (features, labels)
            return next_example, next_label

    # Return function and hook
    return train_inputs, iterator_initializer_hook


def get_test_inputs(batch_size, mnist_data):
    """Return the input function to get the test data.
    Args:
        batch_size (int): Batch size of training iterator that is returned
                          by the input function.
        mnist_data (Object): Object holding the loaded mnist data.
    Returns:
        (Input function, IteratorInitializerHook):
            - Function that returns (features, labels) when called.
            - Hook to initialise input iterator.
    """
    iterator_initializer_hook = IteratorInitializerHook()

    def test_inputs():
        """Returns training set as Operations.
        Returns:
            (features, labels) Operations that iterate over the dataset
            on every evaluation
        """
        with tf.name_scope('Test_data'):
            # Get Mnist data
            images = mnist_data.test.images.reshape([-1, 28, 28, 1])
            labels = mnist_data.test.labels
            # Define placeholders
            images_placeholder = tf.placeholder(
                images.dtype, images.shape)
            labels_placeholder = tf.placeholder(
                labels.dtype, labels.shape)
            # Build dataset iterator
            dataset = tf.contrib.data.Dataset.from_tensor_slices(
                (images_placeholder, labels_placeholder))
            dataset = dataset.batch(batch_size)
            iterator = dataset.make_initializable_iterator()
            next_example, next_label = iterator.get_next()
            # Set runhook to initialize iterator
            iterator_initializer_hook.iterator_initializer_func = \
                lambda sess: sess.run(
                    iterator.initializer,
                    feed_dict={images_placeholder: images,
                               labels_placeholder: labels})
            return next_example, next_label

    # Return function and hook
    return test_inputs, iterator_initializer_hook


# Run script ##############################################
if __name__ == "__main__":
    tf.app.run(
        main=run_experiment
    )
```