---
title: "Caffe Notes"
date: 2018-11-02T11:04:50+08:00
draft: false
tags: [
    "caffe"
]
categories: [
    "Development",
    "Deep_Learning",
]
---

## tools/caffe.cpp
The statement [`::gflags::ParseCommandLineFlags(pargc, pargv, true);`][1]
changes `argc`. [`if (argc == 2) {`][3] limits the number of arguments to two.
Options starting with `-` or `--` do not contribute to the number of arguments.
So [the only valid argument][3] is `train`, `test`, `device_query`
or `time`.

### train
[`RegisterBrewFunction(train);`][4] registers the [train][5] handler which is invoked when `./caffe train` is used.

Accepted options:

- [`--solver`][6]
    * Example: `--solver=models/bvlc_reference_caffenet/solver.prototxt`
    * [`ReadSolverParamsFromTextFileOrDie()`][7]
    * [`ReadProtoFromTextFile()`][8]
    * [`int fd = open(filename, O_RDONLY);`][9]
    * therefore, the path to the solver proto file is relative to the current
executable `caffe`.
- [`--snapshot`][10]
    * Example: `--snapshot=models/xxx/train_iter_10000.solverstate`
    * If provided, do not provide `--weights`, otherwise it panics.
    * [`solver->Restore(FLAGS_snapshot.c_str());`][21]
- [`--weights`][11]
    * Example: `--weights="model/bvlc_reference_caffenet.caffemodel"`
    * Example: `--weights="foo.caffemodel,bar.caffemodel,foobar.caffemodel`
    * If provided, do not provide `--snapshot`, otherwise it panics.
    * [`solver_param.add_weights(FLAGS_weights);`][20]
- [`--stage`][12]
    * Example: `--stage="foo,bar"`
    * [`boost::split(stages, FLAGS_stage, boost::is_any_of(","));`][13]
    * Note that multiple stages are separated with `,`
    * [`solver_param.mutable_train_state()->add_stage(stages[i]);`][16]
- [`--level`][14]
    * Example: `--level=0`
    * [`solver_param.mutable_train_state()->set_level(FLAGS_level);`][15]
- [`--gpu`][17]
    * Example: `--gpu="0"`, to use gpu 0
    * Example: `--gpu="0,3"`, to use gpu 0 and gpu 3
    * Example: `--gpu=all`, to use all available gpus
    * [`boost::split(strings, FLAGS_gpu, boost::is_any_of(","));`][18]
    * When not specified, if solver specifies solver mode and solver mode is GPU
        - if device id is specified, use the specified device id
        - else use gpu 0
    * If specified, values specified in solver proto are ignored
    * [`Caffe::set_solver_count(gpus.size());`][19]
    * If multiple GPUs are specified, caffe must be built with `USE_NCCL` enabled, otherwise
it panics.

Inside the `train()` function, it first [creates][22] a solver:
```cpp
  shared_ptr<caffe::Solver<float> >
      solver(caffe::SolverRegistry<float>::CreateSolver(solver_param));
```
and then calls [`solver->Solve();`][23]

#### A sample solver prototxt
A sample solver prototxt can be found at [examples/mnist/lenet_solver.prototxt][24]
```
# The train/test net protocol buffer definition
net: "examples/mnist/lenet_train_test.prototxt"
# test_iter specifies how many forward passes the test should carry out.
# In the case of MNIST, we have test batch size 100 and 100 test iterations,
# covering the full 10,000 testing images.
test_iter: 100
# Carry out testing every 500 training iterations.
test_interval: 500
# The base learning rate, momentum and the weight decay of the network.
base_lr: 0.01
momentum: 0.9
weight_decay: 0.0005
# The learning rate policy
lr_policy: "inv"
gamma: 0.0001
power: 0.75
# Display every 100 iterations
display: 100
# The maximum number of iterations
max_iter: 10000
# snapshot intermediate results
snapshot: 5000
snapshot_prefix: "examples/mnist/lenet"
# solver mode: CPU or GPU
solver_mode: GPU
```

#### How is a solver created?
- [include/caffe/solver_factory.hpp][25]

The base class is [class Solver][26]; the [sgd solver][27]
is declared as

```cpp
class SGDSolver : public Solver<Dtype> {
```
and is registered as [REGISTER_SOLVER_CLASS(SGD);][28]

We can create a solver of type `SGDSolver` by using its [type][29].

```proto
optional string type = 40 [default = "SGD"];
```

```cpp
// https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/solver_factory.hpp#L74
    const string& type = param.type();
    CreatorRegistry& registry = Registry();
    return registry[type](param);
```

#### The SGD solver
The [contstructor][30]

```cpp
  explicit SGDSolver(const SolverParameter& param)
      : Solver<Dtype>(param) { PreSolve(); }
```
Inside the constructor of the base class `Solver`:
    * [`InitTrainNet()`][32]
        - read the net parameter (it can be from one of the four sources).
        - set net phase to `TRAIN`
        - create a new net from net param
        - load pretrained weights if any using [net->CopyTrainedLayersFrom][33]
            * a layer is unqiuely identified by its [name][34]
            * only [blobs][35] of a layer are copied from the pretrained file

#### Gradient Clipping
[Gradient clipping][37]

```cpp
template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}
```

#### Weight Decay
[Weight decay][38], refer to [here][39]

```cpp
template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
....
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
....
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());
....
```

There are two places to set the weight decay:

- For a network, every layer can set its own decay, which is aggregated
into a net. See the code below

```proto
message LayerParameter {
    repeated ParamSpec param = 6;
...
message ParamSpec {
    optional float decay_mult = 4 [default = 1.0];
```

- For a solver,

```proto
message SolverParameter {
  optional float weight_decay = 12; // The weight decay.
  // regularization types supported: L1 and L2
  // controlled by weight_decay
  optional string regularization_type = 29 [default = "L2"];
...
```
- the total weight for weight decay is the product of the weight decay
of the solver and the corresponding layer.

AlexNet use `0.0005*learning_rate=0.0005*0.01=5e-6` for weight decay.

#### Learning Rate
Every layer can set a learning rate:

```proto
message LayerParameter {
    repeated ParamSpec param = 6;
...
message ParamSpec {
    optional float lr_mult = 3 [default = 1.0];
```

The actual learning rate is the current learning rate multiplied with `lr_mult`.

The current learning rate is [determined][41] by

```cpp
// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
```

`GetLearningRate()` is used in the momentum update.


#### Momentum Update
Refer to [here][42]

```proto
message SolverParameter {
    optional float momentum = 11; // The momentum value.
```

See the [code][40]

```cpp
template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data());
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
```

```
last_time = momentum*last_time + learning_rate*current_computed
current_computed = last_time
```

(Note that weight decay has been applied to `current_computed` in a previous step.)

#### Nesterov Momentum Update
See the [code][43]

```cpp
template <typename Dtype>
void NesterovSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    // save history momentum for stepping back
    caffe_copy(net_params[param_id]->count(),
        this->history_[param_id]->cpu_data(),
        this->update_[param_id]->mutable_cpu_data());

    // update history
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              this->history_[param_id]->mutable_cpu_data());

    // compute update: step back then over step
    caffe_cpu_axpby(net_params[param_id]->count(), Dtype(1) + momentum,
        this->history_[param_id]->cpu_data(), -momentum,
        this->update_[param_id]->mutable_cpu_data());

    // copy
    caffe_copy(net_params[param_id]->count(),
        this->update_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
```

```
saved = last_time
last_time = momentum*last_time + learning_rate*current_computed
current_computed = (1+momentum)*last_time - momentum*saved
```

#### Adam Momentum Update
See the [code][44]

#### How is a net created?

All parameters to be learned are saved in `Net::learnable_params()`

- [`ReadNetParamsFromTextFileOrDie()`][45]
    * [`ReadProtoFromTextFile()`][46]
    * [`UpgradeNetAsNeeded()`][47]

#### How to update a net
There are three versions: v0, v1 and v2.

##### From v0 to v1

```proto
message NetParameter {
  // DEPRECATED. See InputParameter. The input blobs to the network.
  repeated string input = 3;
  // The layers that make up the net.  Each of their configurations, including
  // connectivity and behavior, is specified as a LayerParameter.
  repeated LayerParameter layer = 100;  // ID 100 so layers are printed last.

  // DEPRECATED: use 'layer' instead.
  repeated V1LayerParameter layers = 2;
...

message V1LayerParameter {
    optional V0LayerParameter layer = 1;
```
when `layers` is set and `V0LayerParameter layer` is present in
`layers`, then we need to upgrade from V0 to V1.


##### Data Upgrade
When `V1LayerParameter` is presented and its type is data, image data or window data.

##### From v1 to v2
If `V1LayerParameter` is presented, then it is performed.

##### Input Upgrade
It is performed when `input` is present.

A `LayerParameter` of type is [added][48].

##### BatchNorm Upgrade
If `LayerParameter` is of type `BatchNorm` and it has 3 parameters:

```proto
message LayerParameter {
  optional string type = 2; // the layer type, BatchNorm
  repeated ParamSpec param = 6; // and if it has 3 parameters
```

See the [code][49] below:

```cpp
void UpgradeNetBatchNorm(NetParameter* net_param) {
  for (int i = 0; i < net_param->layer_size(); ++i) {
    // Check if BatchNorm layers declare three parameters, as required by
    // the previous BatchNorm layer definition.
    if (net_param->layer(i).type() == "BatchNorm"
        && net_param->layer(i).param_size() == 3) {
      // set lr_mult and decay_mult to zero. leave all other param intact.
      for (int ip = 0; ip < net_param->layer(i).param_size(); ip++) {
        ParamSpec* fixed_param_spec =
          net_param->mutable_layer(i)->mutable_param(ip);
        fixed_param_spec->set_lr_mult(0.f);
        fixed_param_spec->set_decay_mult(0.f);
      }
    }
  }
}
```

It sets the learning rate and weight decay to 0.

#### How to avoid upgrading
1. Do not use V0LayerParameter
2. Do not use V1LayerParameter
3. Do not use `input`.
4. Use LayerParameter
    - upgrade is unavoided if a layer type is `BatchNorm`

### How are layers connected
- `Net::Net()` calls `ReadNetParamsFromTextFileOrDie()`
to load `NetParameter` from a proto txt file.

```cpp
  explicit Net(const NetParameter& param);
  explicit Net(const string& param_file, Phase phase,
      const int level = 0, const vector<string>* stages = NULL);

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param) {
  Init(param);
}

template <typename Dtype>
Net<Dtype>::Net(const string& param_file, Phase phase,
    const int level, const vector<string>* stages) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  // Set phase, stages and level
  param.mutable_state()->set_phase(phase);
  if (stages != NULL) {
    for (int i = 0; i < stages->size(); i++) {
      param.mutable_state()->add_stage((*stages)[i]);
    }
  }
  param.mutable_state()->set_level(level);
  Init(param);
}
```

```proto
enum Phase {
   TRAIN = 0;
   TEST = 1;
}

message NetState {
  optional Phase phase = 1 [default = TEST];
  optional int32 level = 2 [default = 0];
  repeated string stage = 3;
}
```

If we have a proto txt file, we can first read it into `NetParameter`,
set `NetState`, and then call `Net::Net(const string& filename, ...)`

- [`Net::Init()`][50]
    * `FilterNet()` to exclude layers using

```proto
message LayerParameter {
  // The train / test phase for computation.
  optional Phase phase = 10;

  // Rules controlling whether and when a layer is included in the network,
  // based on the current NetState.  You may specify a non-zero number of rules
  // to include OR exclude, but not both.  If no include or exclude rules are
  // specified, the layer is always included.  If the current NetState meets
  // ANY (i.e., one or more) of the specified rules, the layer is
  // included/excluded.
  repeated NetStateRule include = 8;
  repeated NetStateRule exclude = 9;

...

message NetStateRule {
  // Set phase to require the NetState have a particular phase (TRAIN or TEST)
  // to meet this rule.
  optional Phase phase = 1;

  // Set the minimum and/or maximum levels in which the layer should be used.
  // Leave undefined to meet the rule regardless of level.
  optional int32 min_level = 2;
  optional int32 max_level = 3;

  // Customizable sets of stages to include or exclude.
  // The net must have ALL of the specified stages and NONE of the specified
  // "not_stage"s to meet the rule.
  // (Use multiple NetStateRules to specify conjunctions of stages.)
  repeated string stage = 4;
  repeated string not_stage = 5;
}
```

We can set either `include` or `exclude` or none. If `include` and
`exclude` are both set, then it panics.

- [`InsertSplits()`][51]

```proto
message LayerParameter {
  optional string name = 1; // the layer name
  optional string type = 2; // the layer type
  repeated string bottom = 3; // the name of each bottom blob
  repeated string top = 4; // the name of each top blob

  // The amount of weight to assign each top blob in the objective.
  // Each layer assigns a default value, usually of either 0 or 1,
  // to each top blob.
  repeated float loss_weight = 5;
```

- connect layers
    * if a layer does not have phase specified, then set it to the net phase
    * if `propagate_down` is specified, check that it equals to the number of bottoms of this layer
    * `layers_.push_back(LayerRegistry<Dtype>::CreateLayer(layer_param));`


### How to include/exclude a layer

We can specify some properties of the net and if the layer
does not have any relevant properties, then the layer is included
by default. If we want to exclude the layer, we have to do
some additional work by specifying its `exclude` to match
the given properties of the net. If any one of the properties
is satisfied, the layer is excluded.

Another way is to specify the `include` property of the layer
and if the net does not have the specified property required
by the layer, the layer is not included.

`include` and `exclude` cannot be specified at the same time;
otherwise it panics.

#### Phase

- Set the phase of the net to `TRAIN`
    * do nothing to include the layer!
    * to exclude the layer: set the phase for the `exclude` in `LayerParameter`
to `TRAIN`
    * Note that `LayerParameter.Phase` is not considered here!

#### Level
- Set the level of the net to some number, e.g., 10
    * do nothing to include the layer
    * to exclude the layer: set the level for the `exclude` in `LayerParameter` to
        - `min_level` : e.g., 11, since the level of the net is 10, which is less than 11, so the layer is excluded
        - `max_level`: e.g., 8, since the level of the net is 10, which is larger than 8, so the layer is excluded
        - if either `min_level` or `max_level` does not meet the level of the net,the layer is excluded


#### Stage
- Set the stage of the net to some list of strings, e.g., `{"foo", "bar"}`
    * do nothing to include the layer
    * add either "foo" or "bar" or both to `exclude` of the layer

## Data Members of the Class Net
- `vector<string> layer_names_;`
    * `layer_names_[0]` is the name of the 0th layer
    * `layer_names_[1]` is the name of the 1st layer
    * and so on
- `vector<shared_ptr<Blob<Dtype> > > blobs_`
    * pointers of all blobs
- `vector<Blob<Dtype>*> net_input_blobs_`
    * saves the pointers of input blobs, which are also saved in `blobs_`
- `vector<int> net_input_blob_indices_;`
    * `net_input_blob_indices_[0]`: the blob `blobs_[net_input_blob_indices[0]]`
    * `net_input_blob_indices_[1]`: the blob `blobs_[net_input_blob_indices[1]]`


## Layer

```proto
message LayerParameter {
  repeated BlobProto blobs = 7;
```

```cpp
  vector<shared_ptr<Blob<Dtype> > > blobs_;

  explicit Layer(const LayerParameter& param)
    : layer_param_(param) {
      // Set phase and copy blobs (if there are any).
      phase_ = param.phase();
      if (layer_param_.blobs_size() > 0) {
        blobs_.resize(layer_param_.blobs_size());
        for (int i = 0; i < layer_param_.blobs_size(); ++i) {
          blobs_[i].reset(new Blob<Dtype>());
          blobs_[i]->FromProto(layer_param_.blobs(i));
        }
      }
    }

```

Setup the internal blobs of a layer in the constructor.

```cpp
  void SetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    CheckBlobCounts(bottom, top);
    LayerSetUp(bottom, top);
    Reshape(bottom, top);
    SetLossWeights(top);
  }

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) = 0;
```

## Base Data Layer

```cpp
template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() == 1) {
    output_labels_ = false;
  } else {
    output_labels_ = true;
  }
  data_transformer_.reset(
      new DataTransformer<Dtype>(transform_param_, this->phase_));
  data_transformer_->InitRand();
  // The subclasses should setup the size of bottom and top
  DataLayerSetUp(bottom, top);
}

  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  // Data layers have no bottoms, so reshaping is trivial.
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {}

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
```

Subclasses of `BaseDataLayer` have to implement `DataLayerSetUp()`.

Note that the 0th blob of the data layer cannot be labels!

## BasePrefetchingDataLayer

```cpp
// base_data_layer.hpp

template <typename Dtype>
class BasePrefetchingDataLayer :
    public BaseDataLayer<Dtype>, public InternalThread {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
  }
}
```

### InternalThread

```cpp
  shared_ptr<boost::thread> thread_;

  virtual void InternalThreadEntry() {}
...

  try {
    thread_.reset(new boost::thread(&InternalThread::entry, this, device, mode,
          rand_seed, solver_count, solver_rank, multiprocess));
  } catch (std::exception& e) {
    LOG(FATAL) << "Thread exception: " << e.what();
  }
...
void InternalThread::entry(int device, Caffe::Brew mode, int rand_seed,
    int solver_count, int solver_rank, bool multiprocess) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(device));
#endif
  Caffe::set_mode(mode);
  Caffe::set_random_seed(rand_seed);
  Caffe::set_solver_count(solver_count);
  Caffe::set_solver_rank(solver_rank);
  Caffe::set_multiprocess(multiprocess);

  InternalThreadEntry();
}
```

## DataLayer

```cpp
template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype> {
  virtual inline const char* type() const { return "Data"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }
    ...

template<typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
```

Use data layer if the data is from a database, i.e., leveldb.
For example, refer to [examples/mnist/lenet_train_test.prototxt][64]

```proto
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TRAIN
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_train_lmdb"
    batch_size: 64
    backend: LMDB
  }
}
layer {
  name: "mnist"
  type: "Data"
  top: "data"
  top: "label"
  include {
    phase: TEST
  }
  transform_param {
    scale: 0.00390625
  }
  data_param {
    source: "examples/mnist/mnist_test_lmdb"
    batch_size: 100
    backend: LMDB
  }
}
```

## InputLayer

See the [code][54]

```cpp
template <typename Dtype>
void InputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_top = top.size();
  const InputParameter& param = this->layer_param_.input_param();
  const int num_shape = param.shape_size();
  CHECK(num_shape == 0 || num_shape == 1 || num_shape == num_top)
      << "Must specify 'shape' once, once per top blob, or not at all: "
      << num_top << " tops vs. " << num_shape << " shapes.";
  if (num_shape > 0) {
    for (int i = 0; i < num_top; ++i) {
      const int shape_index = (param.shape_size() == 1) ? 0 : i;
      top[i]->Reshape(param.shape(shape_index));
    }
  }
}

// net.hpp

  inline const vector<Blob<Dtype>*>& input_blobs() const {
    return net_input_blobs_;
  }
```

```proto
message BlobShape {
  repeated int64 dim = 1 [packed = true];
}

message InputParameter {
  // This layer produces N >= 1 top blob(s) to be assigned manually.
  // Define N shapes to set a shape for each top.
  // Define 1 shape to set the same shape for every top.
  // Define no shape to defer to reshaping manually.
  repeated BlobShape shape = 1;
}

name: "LeNet"
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param { shape: { dim: 64 dim: 1 dim: 28 dim: 28 } }
}
```

In [net.cpp][52]

```cpp
    int num_top = layer_param.top_size();
    for (int top_id = 0; top_id < num_top; ++top_id) {
      AppendTop(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
      // Collect Input layer tops as Net inputs.
      if (layer_param.type() == "Input") {
        const int blob_id = blobs_.size() - 1;
        net_input_blob_indices_.push_back(blob_id);
        net_input_blobs_.push_back(blobs_[blob_id].get());
      }
    }
```

and in [examples/cpp_classification/classification.cpp][53]

```cpp
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
  Blob<float>* input_layer = net_->input_blobs()[0];

  int width = input_layer->width();
  int height = input_layer->height();
  float* input_data = input_layer->mutable_cpu_data();
  for (int i = 0; i < input_layer->channels(); ++i) {
    cv::Mat channel(height, width, CV_32FC1, input_data);
    input_channels->push_back(channel);
    input_data += width * height;
  }
}
```

Pay attention to `input_layer->mutable_cpu_data();`. It is
`mutable_cput_data()` instead of `cpu_data()` !!!

## Neuron Layer

[neuron_layer.hpp][70]

```cpp
template <typename Dtype>
class NeuronLayer : public Layer<Dtype> {
 public:
  explicit NeuronLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
};
```

[neuron_layer.cpp][71]

```cpp
template <typename Dtype>
void NeuronLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}
```

## Sigmod Layer

The sigmoid function is
$$
f(x) = \frac{1}{1+e^{-x}}
$$

The tanh function is
$$
g(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
= \frac{1 - e^{-2x}}{1+e^{-2x}}
= 1 - \frac{2e^{-2x}}{1+e^{-2x}}
$$

Note that
$$
\frac{1}{2} g(\frac{1}{2}x) + \frac{1}{2} =
\frac{1}{2} (1 - \frac{2e^{-x}}{1+e^{-x}}) + \frac{1}{2}
= 1 - \frac{e^{-x}}{1+e^{-x}}
= \frac{1}{1+e^{-x}}
$$

See the [code][72] below

```cpp
template <typename Dtype>
inline Dtype sigmoid(Dtype x) {
  return 0.5 * tanh(0.5 * x) + 0.5;
}
```

[forward pass][73]

```cpp
template <typename Dtype>
void SigmoidLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = sigmoid(bottom_data[i]);
  }
}
```

The derivative of a sigmoid function is
$$
\frac{\mathrm{d}f}{\mathrm{d}x} =
\frac{e^{-x}}{(1+e^{-x})^2} =
\frac{1+e^{-x}-1}{(1+e^{-x})^2} =
\frac{1}{1+e^{-x}} - \frac{1}{(1+e^{-x})^2} =
= f - f^2
= f(1-f)
$$

The [backward][74] pass is

```cpp
template <typename Dtype>
void SigmoidLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
      const Dtype sigmoid_x = top_data[i];
      bottom_diff[i] = top_diff[i] * sigmoid_x * (1. - sigmoid_x);
    }
  }
}
```

## tanh Layer
[std::tanh][75]

The [forward][76] pass

```cpp
template <typename Dtype>
void TanHLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = tanh(bottom_data[i]);
  }
}
```

The derivative of tanh is
$$
\frac{\mathrm{d}g}{\mathrm{d}x} =
\frac{(e^x+e^{-x})(e^x+e^{-x}) - (e^x-e^{-x})(e^x-e^{-x})}{(e^x+e^{-x})^2}
= 1 - g^2
$$

The [backward][77] pass is

```cpp
template <typename Dtype>
void TanHLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* top_data = top[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype tanhx;
    for (int i = 0; i < count; ++i) {
      tanhx = top_data[i];
      bottom_diff[i] = top_diff[i] * (1 - tanhx * tanhx);
    }
  }
}
```

## ReLu Layer
ReLU is short for Rectified Linear Unit.

```proto
// Message that stores parameters used by ReLULayer
message ReLUParameter {
  // Allow non-zero slope for negative inputs to speed up optimization
  // Described in:
  // Maas, A. L., Hannun, A. Y., & Ng, A. Y. (2013). Rectifier nonlinearities
  // improve neural network acoustic models. In ICML Workshop on Deep Learning
  // for Audio, Speech, and Language Processing.
  optional float negative_slope = 1 [default = 0];
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 2 [default = DEFAULT];
}
```

Pay attention to `negative_slope`; it is called **leaky**
rectified linear unit. Its usage is shown below.
The paper in the comment is available [here][79]

The [forward][78] pass is

```cpp
template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::max(bottom_data[i], Dtype(0))
        + negative_slope * std::min(bottom_data[i], Dtype(0));
  }
}
```

The [backward][80] pass is

```cpp
template <typename Dtype>
void ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slope = this->layer_param_.relu_param().negative_slope();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0)
          + negative_slope * (bottom_data[i] <= 0));
    }
  }
}
```

## PReLu
PReLu is short for Parametric ReLu, which was proposed by
[He Kaiming][81] in 2015.

Note that in leaky ReLu, the negative slope is fixed while
in the PReLu, the negative slope is a learnable parameter.

```proto
message PReLUParameter {
  // Parametric ReLU described in K. He et al, Delving Deep into Rectifiers:
  // Surpassing Human-Level Performance on ImageNet Classification, 2015.

  // Initial value of a_i. Default is a_i=0.25 for all i.
  optional FillerParameter filler = 1;
  // Whether or not slope parameters are shared across channels.
  optional bool channel_shared = 2 [default = false];
}
```

## Pooling Layer

```proto
message PoolingParameter {
  enum PoolMethod {
    MAX = 0;
    AVE = 1;
    STOCHASTIC = 2;
  }
  optional PoolMethod pool = 1 [default = MAX]; // The pooling method
...
```

```cpp
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
```

## Loss Layer
It takes two input:
    * index 0: predictions
    * index 1: ground truth

`LossLayer` is the base class for other kinds of losses.

[loss_layer.hpp][82]

```cpp
const float kLOG_THRESHOLD = 1e-20;

template <typename Dtype>
class LossLayer : public Layer<Dtype> {
    virtual inline int ExactNumBottomBlobs() const { return 2; }
```

[loss_layer.cpp][83]

```cpp
template <typename Dtype>
void LossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // LossLayers have a non-zero (1) loss by default.
  if (this->layer_param_.loss_weight_size() == 0) {
    this->layer_param_.add_loss_weight(Dtype(1));
  }
}

template <typename Dtype>
void LossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0))
      << "The data and label should have the same first dimension.";
  vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}
```
Note that it set `loss_weight[0]` to 1 and the output is a scalar.

## Negative Log Loss (neg loss)
It is `MultinomialLogisticLossLayer` in Caffe.

[multinomial_logistic_loss_layer.hpp][84]

```cpp
template <typename Dtype>
class MultinomialLogisticLossLayer : public LossLayer<Dtype> {
```

[multinomial_logistic_loss_layer.cpp][85]

```cpp
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  CHECK_EQ(bottom[1]->height(), 1);
  CHECK_EQ(bottom[1]->width(), 1);
}
```

Note that `bottom[1]` is the ground truth and contains
a batch of scalars. `bottom[1]->num()` is the batch size.

The [forward][86] pass

```cpp
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int dim = bottom[0]->count() / bottom[0]->num();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    int label = static_cast<int>(bottom_label[i]);
    Dtype prob = std::max(
        bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
    loss -= log(prob);
  }
  top[0]->mutable_cpu_data()[0] = loss / num;
}
```

It first identifies the label of the input and then extracts
the corresponding prediction of the label and take the negative
log of the prediction as the loss. Since after the soft max operation
the prediction is in the range `[0,1]`, if the prediction is correct,
i.e., it is near 1, then the negative log of the prediction is nearly 0,
which is contributes less to the total loss; if the prediction is false,
i.e., it is near 0, then the negative log of the prediction is positively huge
and contributes a lot to the final loss.

The final loss is averaged over the batch size and is saved
into `top[0]`.

[layer.hpp][88]

```cpp
template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype loss = 0;
  Reshape(bottom, top);
  switch (Caffe::mode()) {
  case Caffe::CPU:
    Forward_cpu(bottom, top);
    for (int top_id = 0; top_id < top.size(); ++top_id) {
      if (!this->loss(top_id)) { continue; }
      const int count = top[top_id]->count();
      const Dtype* data = top[top_id]->cpu_data();
      const Dtype* loss_weights = top[top_id]->cpu_diff();
      loss += caffe_cpu_dot(count, data, loss_weights);
    }
    break;
```

Note that `top[0]->cpu_diff()` is computed cumulatively above!
`this->loss(0)` has been initialized to 1 previously (in the base class `LossLayer`).

`top[0]->cpu_data()` contains the loss averaged over one batch and `loss_weights[0]`
is 1, so `top[0]->cpu_diff()` contains the same value as
`top[0]->cpu_data`, which is used for backward propagation.

[net.hpp][89]

```cpp
  Dtype ForwardBackward() {
    Dtype loss;
    Forward(&loss);
    Backward();
    return loss;
  }
```

[net.cpp][90]

```cpp
template <typename Dtype>
const vector<Blob<Dtype>*>& Net<Dtype>::Forward(Dtype* loss) {
  if (loss != NULL) {
    *loss = ForwardFromTo(0, layers_.size() - 1);
  } else {
    ForwardFromTo(0, layers_.size() - 1);
  }
  return net_output_blobs_;
}

template <typename Dtype>
Dtype Net<Dtype>::ForwardFromTo(int start, int end) {
  CHECK_GE(start, 0);
  CHECK_LT(end, layers_.size());
  Dtype loss = 0;
  for (int i = start; i <= end; ++i) {
    for (int c = 0; c < before_forward_.size(); ++c) {
      before_forward_[c]->run(i);
    }
    Dtype layer_loss = layers_[i]->Forward(bottom_vecs_[i], top_vecs_[i]);
    loss += layer_loss;
    if (debug_info_) { ForwardDebugInfo(i); }
    for (int c = 0; c < after_forward_.size(); ++c) {
      after_forward_[c]->run(i);
    }
  }
  return loss;
}
```

[solver.cpp][91]

```cpp
    Dtype loss = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
    }
    loss /= param_.iter_size();
```

The above loss is averaged over iterations. Note that
is has already been averaged over batches.

The [backward][87] pass

```cpp
template <typename Dtype>
void MultinomialLogisticLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int dim = bottom[0]->count() / bottom[0]->num();
    caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
    const Dtype scale = - top[0]->cpu_diff()[0] / num;
    for (int i = 0; i < num; ++i) {
      int label = static_cast<int>(bottom_label[i]);
      Dtype prob = std::max(
          bottom_data[i * dim + label], Dtype(kLOG_THRESHOLD));
      bottom_diff[i * dim + label] = scale / prob;
    }
  }
}
```

Note that `top[0]->cpu_diff()` has been initialized in `layer.hpp` (see above).

Since `bottom[1]` is the ground truth label, it first checks that
`propagate_down[1]` is false.

$$
f = \frac{1}{n} \sum_{i}^{n}(-\log y_i)
$$

where f is the total loss.

$$
\frac{\partial f}{\partial y_i} =
-\frac{1}{n} \cdot \frac{1}{y_i}
$$

$\frac{\partial \mathrm{loss}}{\partial f}$ is equal to
`top[0]->cpu_diff()[0]`

## Softmax Layer

```proto
// Message that stores parameters used by SoftmaxLayer, SoftmaxWithLossLayer
message SoftmaxParameter {
  enum Engine {
    DEFAULT = 0;
    CAFFE = 1;
    CUDNN = 2;
  }
  optional Engine engine = 1 [default = DEFAULT];

  // The axis along which to perform the softmax -- may be negative to index
  // from the end (e.g., -1 for the last axis).
  // Any other axes will be evaluated as independent softmaxes.
  optional int32 axis = 2 [default = 1];
}
```
Pay attention to `axis` above! Its default value is 1, which refers to
the number of channels.

[softmax_layer.hpp][92]

```cpp
template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype> {
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
```

[softmax_layer.hpp][94]

```cpp
template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  top[0]->ReshapeLike(*bottom[0]);
  vector<int> mult_dims(1, bottom[0]->shape(softmax_axis_));
  sum_multiplier_.Reshape(mult_dims);
  Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
  caffe_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  vector<int> scale_dims = bottom[0]->shape();
  scale_dims[softmax_axis_] = 1;
  scale_.Reshape(scale_dims);
}
```

If the shape of the bottom is `2x3x4x5` and
`softmax_axis` is 1, then

* `sum_multiplier_` is `[1, 1, 1]`, i.e., a vector of 3 scalars
* `outer_num_` is 2, which is the batch size
* `inner_num_` is 20, which is the product of 4 and 5
* `scale_` has the shape `2x1x4x5` (by fangjun: it should be 4x5)

Every channel is a vector of 20 numbers and the output is
3 channel vectors, each of which is a  vector of 20 numbers.

`out[0][0] = softmax(channel[0][0]; channel[0][0], channel[1][0], channel[2][0])`

`out[0][1] = softmax(channel[0][1]; channel[0][1], channel[1][1], channel[2][1])`


The [forward][95] pass

```cpp
template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = bottom[0]->shape(softmax_axis_);
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}
```

For the `subtraction`, `sum_multiplier` is of size
`channels x 1`, `scale_data` if of size `1 x inner_num_`,
`top_data` is of size `channels x inner_num_`. alpha is -1 and beta is 1.

`sum_multiplier * scale_data` is to replicate the rows of max values
and then subtract it from the top data.

For the `exponentiation`, `top_data` is first transposed
to a shape `inner_num x channels`, then it is multiplied with
`sum_multiplier` (shape is `channels x 1`) and the results
are saved into `inner_num`.

For the `division`, the numerator is in `top_data` and the
denominator is in `scale_data`


[math_functions][97]

```cpp
template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}
```

[cblas_sgemm][96] `C = alpha*A*B + beta*C`.
`A` is `MxK`, `B` is `KxN`, `C` is `MxN`.

[cblas_sgemv][98], `y = alpha * transpose(A) * x + beta * y`.
`A` is `MxN`.

The [backward][99] pass is

```cpp
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}
```

Use the chain rule to show that the above is indeed correct.

The following shows the backward propagation rule of the softmax layer
in Caffe.

$$
f_i = \frac{e^{x_i}}{\sum_j e^j}
$$

$$
\frac{\partial f_k}{\partial x_i}
= \frac{[i==k]e^k \sum_j e^j - e^i e^k}{(\sum_j e^j)^2}
= e^k(\frac{[i==k]}{\sum_j e^j} - \frac{e^i}{(\sum_j e^j)^2})
$$

$$
\frac{\partial \mathrm{loss}}{\partial x_i}
= \sum_k \frac{\partial \mathrm{loss}}{\partial f_k} \frac{\partial f_k}{x_i}
= \sum_k \frac{\partial \mathrm{loss}}{\partial f_k} e^k
(\frac{[i==k]}{\sum_j e^j} - \frac{e^i}{(\sum_j e^j)^2})
$$

$$
\frac{\partial \mathrm{loss}}{\partial x_i}
= \sum_k \frac{\partial \mathrm{loss}}{\partial f_k}e^k\frac{[i==k]}{\sum_j e^j}
- \sum_k \frac{\partial \mathrm{loss}}{\partial f_k}\frac{e^k}{\sum_j e^j}\frac{e^i}{\sum_j e^j}
$$

$$
\frac{\partial \mathrm{loss}}{\partial x_i}
= \frac{\partial \mathrm{loss}}{\partial f_i} f_i
- \sum_k \frac{\partial \mathrm{loss}}{\partial f_k} f_k f_i
= f_i (\frac{\partial \mathrm{loss}}{\partial f_i} - \sum_k \frac{\partial \mathrm{loss}}{\partial f_k} f_k)
$$

In the code, `bottom_diff` is $\frac{\partial \mathrm{loss}}{\partial f_k}$, which is copied from `top_diff`.

`compute dot(top_diff, top_data)` computes the inner product in the above euqation:
$$
\sum_k \frac{\partial \mathrm{loss}}{\partial f_k} f_k
$$

`subtraction` implements the expression in the
parenthesis in the above equation:
$$
\frac{\partial \mathrm{loss}}{\partial f_i} - \sum_k \frac{\partial \mathrm{loss}}{\partial f_k} f_k
$$

`elementwise multiplication` is the last step in the above equation:
$$
f_i (\frac{\partial \mathrm{loss}}{\partial f_i} - \sum_k \frac{\partial \mathrm{loss}}{\partial f_k} f_k)
$$


## SoftMax with Loss Layer

Loss here means negative log loss. The comment in the source code
called it as cross entropy loss. It is also known as
multinomial logistic loss.

Note that there is no cross entropy loss layer in caffe.
Caffe provides `MultinomialLogisticLayer`.

`SoftmaxWithLossLayer` combines `SoftMaxLayer`
and `MultinomialLogisticLossLayer`. The gradient
propagation is much simpler than
`SoftMaxLayer` and
`MultinomialLogisticLossLayer`.

Note that we cannot ignore labels in
`MultinomialLogisticLossLayer`
but we can ignore labels in `SoftmaxWithLossLayer`.

```cpp
template <typename Dtype>
class SoftmaxWithLossLayer : public LossLayer<Dtype> {
 public:
   /**
    * @param param provides LossParameter loss_param, with options:
    *  - ignore_label (optional)
    *    Specify a label value that should be ignored when computing the loss.
    *  - normalize (optional, default true)
    *    If true, the loss is normalized by the number of (nonignored) labels
    *    present; otherwise the loss is simply summed over spatial locations.
    */
  explicit SoftmaxWithLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SoftmaxWithLoss"; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 2; }

```

If it has two tops, then `top[1]` is the output of the softmax.
`top[0]` is always the loss.

```proto
// Message that stores parameters shared by loss layers
message LossParameter {
  // If specified, ignore instances with the given label.
  optional int32 ignore_label = 1;
  // How to normalize the loss for loss layers that aggregate across batches,
  // spatial dimensions, or other dimensions.  Currently only implemented in
  // SoftmaxWithLoss and SigmoidCrossEntropyLoss layers.
  enum NormalizationMode {
    // Divide by the number of examples in the batch times spatial dimensions.
    // Outputs that receive the ignore label will NOT be ignored in computing
    // the normalization factor.
    FULL = 0;
    // Divide by the total number of output locations that do not take the
    // ignore_label.  If ignore_label is not set, this behaves like FULL.
    VALID = 1;
    // Divide by the batch size.
    BATCH_SIZE = 2;
    // Do not normalize the loss.
    NONE = 3;
  }
  // For historical reasons, the default normalization for
  // SigmoidCrossEntropyLoss is BATCH_SIZE and *not* VALID.
  optional NormalizationMode normalization = 3 [default = VALID];
  // Deprecated.  Ignored if normalization is specified.  If normalization
  // is not specified, then setting this to false will be equivalent to
  // normalization = BATCH_SIZE to be consistent with previous behavior.
  optional bool normalize = 2;
}
```
It is possible to ignore one class via `ignore_label`.
It is impossible to ignore more than one class without changing
the caffe source code.

```cpp
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
}
```

Note that if `has_ignore_label` is set, `ignore_label_` saves
the ignored label. Labels must be continuous and counted from 0.

It uses `softmax_layer_` internally. The above code also sets up
the bottom and top vectors of the softmax layer.

`softmax_bottom_vec_` saves `bottom[0]`, which is the data.
`bottom[1]` saves labels.

The [forward][100] pass

```cpp
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}
```

If the output shape is `4x5x6x7`, it means the batch size is 4, which is
`outer_num_`; number of classes is `5`. The image size is `6x7` and each
pixel is classified into class 0,1,2,3 or 4. The output is a probability
map image with 5 channels.

Note the following code in the above

```cpp
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
```

```cpp
template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}
```

If the ignore label is not set, then
`LossParameter_NormalizationMode_FULL` is the same as
`LossParameter_NormalizationMode_VALID`.

If the ignore label is set,
`LossParameter_NormalizationMode_FULL` is `outer_number * inner_number`
and
`LossParameter_NormalizationMode_VALID` is number of labels that contributes
to the final loss.

The [backward][101] pass is

```cpp
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}
```

Note that the backward gradient propagation is much simpler than
the softmax layer!

Suppose the input feature has 3 components: $x_0$, $x_1$ and $x_2$.
After the softmax layer, we have
$$
y_0 = \frac{e^{x_0}}{e^{x_0} + e^{x_1} + e^{x_2}}
$$

$$
y_1 = \frac{e^{x_1}}{e^{x_0} + e^{x_1} + e^{x_2}}
$$

$$
y_2 = \frac{e^{x_2}}{e^{x_0} + e^{x_1} + e^{x_2}}
$$

If the label is 1, that is $y_1$ is the predicated output, the loss is

$$
\mathrm{loss} = -\log y_1
$$

$$
\frac{\partial \mathrm{loss}}{\partial y_0} = 0
$$

$$
\frac{\partial \mathrm{loss}}{\partial y_1} = - \frac{1}{y_1}
$$

$$
\frac{\partial \mathrm{loss}}{\partial y_2} = 0
$$

$$
\frac{\partial \mathrm{loss}}{\partial x_0}
$$

$$
\frac{\partial \mathrm{loss}}{\partial x_0} =
\frac{\partial \mathrm{loss}}{\partial y_0}
\frac{\partial y_0}{\partial x_0}
+
\frac{\partial \mathrm{loss}}{\partial y_1}
\frac{\partial y_1}{\partial x_0}
+
\frac{\partial \mathrm{loss}}{\partial y_2}
\frac{\partial y_2}{\partial x_0} =
-\frac{1}{y_1} \cdot (-y_0 y_1) = y_0
$$

$$
\frac{\partial \mathrm{loss}}{\partial x_1}
= \frac{\partial \mathrm{loss}}{\partial y_1}
\frac{\partial y_1}{\partial x_1}
= -\frac{1}{y_1}\cdot y_1(1-y_1) = -(1 - y_1)
= y_1 - 1
$$

$$
\frac{\partial \mathrm{loss}}{\partial x_2}
= y_2
$$


The line

```cpp
caffe_copy(prob_.count(), prob_data, bottom_diff);
```
first copies `y_1`, `y_2` and `y_3` to the gradient which are computed as above.


The line

```cpp
bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
```
computes `y_1 - 1`.

If the label is ignored, then there is no gradient for
`x_1`, `x_2` and `x_3`, which is accomplished by the line

```cpp
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        }
```

## Differences Between Snapshot and Weights
### Snapshot

[sovler.cpp][55]

```cpp
template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}
```

[sgd_solver.cpp][56]

```cpp
template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}
```

[tools/caffe.cpp][57]

```cpp
  if (FLAGS_snapshot.size()) {
    solver_param.clear_weights();
  } else if (FLAGS_weights.size()) {
    solver_param.clear_weights();
    solver_param.add_weights(FLAGS_weights);
  }

  if (FLAGS_snapshot.size()) {
    LOG(INFO) << "Resuming from " << FLAGS_snapshot;
    solver->Restore(FLAGS_snapshot.c_str());
  }
```

[solver.cpp][58]

```cpp
template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file) {
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
  }
}
```

[sgd_solver.cpp][59]

```cpp
template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
}
```

[net.cpp][60]

```cpp
template <typename Dtype>
void Net<Dtype>::CopyTrainedLayersFrom(const NetParameter& param) {
  int num_source_layers = param.layer_size();
  for (int i = 0; i < num_source_layers; ++i) {
    const LayerParameter& source_layer = param.layer(i);
    const string& source_layer_name = source_layer.name();
    int target_layer_id = 0;
    while (target_layer_id != layer_names_.size() &&
        layer_names_[target_layer_id] != source_layer_name) {
      ++target_layer_id;
    }
    if (target_layer_id == layer_names_.size()) {
      LOG(INFO) << "Ignoring source layer " << source_layer_name;
      continue;
    }
    DLOG(INFO) << "Copying source layer " << source_layer_name;
    vector<shared_ptr<Blob<Dtype> > >& target_blobs =
        layers_[target_layer_id]->blobs();
    CHECK_EQ(target_blobs.size(), source_layer.blobs_size())
        << "Incompatible number of blobs for layer " << source_layer_name;
    for (int j = 0; j < target_blobs.size(); ++j) {
      if (!target_blobs[j]->ShapeEquals(source_layer.blobs(j))) {
        Blob<Dtype> source_blob;
        const bool kReshape = true;
        source_blob.FromProto(source_layer.blobs(j), kReshape);
        LOG(FATAL) << "Cannot copy param " << j << " weights from layer '"
            << source_layer_name << "'; shape mismatch.  Source param shape is "
            << source_blob.shape_string() << "; target param shape is "
            << target_blobs[j]->shape_string() << ". "
            << "To learn this layer's parameters from scratch rather than "
            << "copying from a saved net, rename the layer.";
      }
      const bool kReshape = false;
      target_blobs[j]->FromProto(source_layer.blobs(j), kReshape);
    }
  }
}
```

Conclusion: snapshots save not only the weights of the network but also
the states of the solver. They are saved to two files, one for the network
weights and the other one for the solver state. For the sgd solver, the solver
state contains the current iteration number and the current step (for computing learning rate), and the weight gradient for momentum update (i.e., the history vector above).

It is useful for continuing the training from the last time point
where it was stopped for some reason. If the solver state is not
available and we only have the network weights at present, then it
is called fine tuning.


### Weights
[tools/caffe.cpp][61]

```cpp
  CHECK(!FLAGS_snapshot.size() || !FLAGS_weights.size())
      << "Give a snapshot to resume training or weights to finetune "
      "but not both.";

  if (FLAGS_snapshot.size()) {
    solver_param.clear_weights();
  } else if (FLAGS_weights.size()) {
    solver_param.clear_weights();
    solver_param.add_weights(FLAGS_weights);
  }
```

[solver.cpp][62]

```cpp
  for (int w_idx = 0; w_idx < param_.weights_size(); ++w_idx) {
    LoadNetWeights(net_, param_.weights(w_idx));
  }

// Load weights from the caffemodel(s) specified in "weights" solver parameter
// into the train and test nets.
template <typename Dtype>
void LoadNetWeights(shared_ptr<Net<Dtype> > net,
    const std::string& model_list) {
  std::vector<std::string> model_names;
  boost::split(model_names, model_list, boost::is_any_of(","));
  for (int i = 0; i < model_names.size(); ++i) {
    boost::trim(model_names[i]);
    LOG(INFO) << "Finetuning from " << model_names[i];
    net->CopyTrainedLayersFrom(model_names[i]);
  }
}
```

Conclusion: weights contain only the network weight and the solver state
is missing. It is useful for fine tuning.
Refer to [Fine-tuning CaffeNet for Style Recognition on “Flickr Style” Data][63].

## Convolution
It is actually correlation from the perspective of the image processing!
In addition, there is a bias term for the correlation, which does
not exist in image processing.

Note that the anchor of the kernel is not at the center but at the top
left corner!

Refer to [CS231N][66].


For a color image, there are several kinds of filtering:

- concatenate the channels. For an rgb image, the filter kernel must be
`nxnx3`. For instance, if the filter kernel is `5x5x3`, for every position
`(x,y)`, take `5x5` from the red channel, `5x5` from the green channel,
`5x5` from the blue channel, and concatenate them into a vector
containing 75 elements and perform an inner product with the filter
to get an output. Therefore, after the filter operation, we get a gray image.
If we have another filter of size `5x5x3`, the above operation is repeated
and we get a 2-channel output. If we have `M` filters of size `5x5x3`, then
the final output is an image with `M` channels.

### 1D convolution in Pytorch
Refer to [torch.nn.Conv1d][67]

```
class torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

### Dilated Convolution
Refer to [Dilated convolution animations][68].

It is also known as atrous convolution.

- if we have 1 filter with kernel size 3x3
    * stride 0, pad 0, the output is

# CS231N Notes
[CS231n Convolutional Neural Networks for Visual Recognition][102]

## Image Classification: Data-driven Approach, k-Nearest Neighbor, train/val/test splits
See <http://cs231n.github.io/classification/>

**Limitation**: the number of categories of the images to be classified
**MUST** be predefined. The classification result is a probability map
and is not an exact prediction but the probability of being each category.
Even if you feed it with an image not in the predefined categories, it still
gives you a result!

topics covered:

- NN: nearest neighbor classification
- kNN: k nearest neighbor classification
- how to choose k in kNN
    * k is called a hyperparameter
    * split training set into a validation set and another training set
    * cross validation

## Linear classification: Support Vector Machine, Softmax
See <http://cs231n.github.io/linear-classify/>

Topics covered:

- linear classifier as template matching
- image data preprocessing
- loss functions
    * SVM loss, hinge loss, squared hinge loss SVM (L2-SVM)
- regularization
- [cs229 SVM lecture notes][105], primal and dual, KKT condition, SMO
- [关于原问题与对偶问题，KKT条件以及其他][106] at zhihu.com

## Optimization: Stochastic Gradient Descent
See <http://cs231n.github.io/optimization-1/>

Topics covered:

- gradient descent
- stochastic gradient descent (SGD)
- batch stochastic gradient descent
- compute gradient numerically

## Backpropagation, Intuitions
See <http://cs231n.github.io/optimization-2/>

Topics covered:

- back propagation examples.

## Neural Networks Part 1: Setting up the Architecture
See <http://cs231n.github.io/neural-networks-1/>

Topics covered:

- introduce the neural network in terms of matrix vector product!
- different activation functions: sigmoid, tanh, ReLu, leaky ReLu, PReLu, MaxOut
- how to count the number of parameters of a neural network

## Neural Networks Part 2: Setting up the Data and the Loss
See <http://cs231n.github.io/neural-networks-2/>

Topics covered:

- data normalization
- weight initialization
- weight regularization
- batch normalization, drop out
- classification loss: hinge loss, cross entropy loss
- regression loss: L2

Refer to [batch_norm_layer.cpp][108]
and
[dropout_layer.cpp][109] in caffe.
Note that caffe use inverted scale in drop out. The parameter
[`dropout_ratio`][110] is the ratio to be removed. It is equal
to `1-p`, where `p` is the same meaning (i.e., present) as in the [paper][112].

## Neural Networks Part 3: Learning and Evaluation
See <http://cs231n.github.io/neural-networks-3/>

Topics covered:

- tricks while training the network
- gradient checker
- how to determine the learning rate
    * loss curve
    * accuracy curves of training/test
    * the ratio $\frac{||\delta w||}{||w||}$ should be about 1e-3
- gradient descent
    * mementum update
    * neterov mementum
    * adagrad: adaptive subgradient, refer to the [caffe][112] implementation.

```
history = history + gradient * gradient
detla_x = learning_rate * gradient / (sqrt(history) + eps)
x = x - delta_x
```
- RMSprop: refer to the [caffe][114] implementation

```
history = momentum * history + (1-momentum) * gradient * gradient
detla_x = learning_rate * gradient / (sqrt(history) + eps)
x = x - delta_x
```

Note that `momentum` here is [`rms_decay`][115] in caffe

- model ensembles: train the network multiple times and average the output
of trained models


Things to take way for this section:

- initial learning rate
- how to decrease the learning rate along training
- how to perform hyperparameter search

## Putting it together: Minimal Neural Network Case Study
See <http://cs231n.github.io/neural-networks-case-study/>

There is no convolution in this section!

It first implements a linear classifier with softmax cross entropy
loss to classifier 3 classes and finds that the accuracy rate is about 49%.
Then a hidden layer with 100 neuron is added and retrained, the final accuracy
is 98%.

There are 3 classes, each of which has 100 2-D points.

It does not show how to draw the decision boundary.

It is implemented in python and no framework is used.

## Convolutional Neural Networks: Architectures, Convolution / Pooling Layers
See <http://cs231n.github.io/convolutional-networks/>

For a color image of size `3x24x24` and the receptive field is of size `5x5`,
then the filter kernel is of size `3x5x5` and the output is `24x24`.

The anchor of the kernel is its top left element and is **NOT** at the center!


For an input of size `3x227x227` and 96 filters of size `3x11x11`, the number of
parameters is
$$
96\times 3 \times 11 times 11  + 96 = 34944
$$

We add 96 because every filter has additional bias, which is quiet different
from the image processing point of view!

Every filter forms a slice and pixels in this slice have the same filter, which
is called parameter sharing.

Note that this section uses actually cross-correlation,
but it considers the operation as convolution.

Pooling decreases the feature size resulting into fewer parameters, which
controls overfitting.

### Layer Patterns

```
input -> [conv ->ReLU]*N -> Pool?]*M -> [FC -> ReLU]*K -> FC
```

`Pool?` means pool is optional.

- 0 <= N <= 3
- M >= 0
- 0 <= K < 3

#### LeNet
see <https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt>

```
input: 64x1x28x28

conv1:
    20 kernels, 5x5, stride 1, pad 0
    output size: 64x20x24x24

pool1 (max):
    2x2, stride 2, pad 0
    output size: 64x20x12x12

conv2:
    50 kernels, 5x5, stride 1, pad 0
    output size: 64x50x8x8

pool2 (max):
    2x2, stride 2, pad 0
    output size: 64x50x4x4

FC (inner product)
    500
    output size: 64x500

ReLU
    output size: 64x500

FC (inner product)
    10
    output size: 64x10

Softmax
```

### VGG 16
refer to prototxt from <https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md>

```
input: 3x224x224
conv1_1:
    64 filters, 3x3, pad 1, stride 1
    output size: 64x224x224
relu1_1:
    output size: 64x224x224
conv1_2:
    64 filters, 3x3, pad 1, stride 1
    output size: 64x224x224
relu1_2:
    output size: 64x224x224
pool1:
    max, 2x2, stride 2, pad 0
    output size: 64x112x112
conv2_1:
    128 filters, 3x3, pad 1, stride 1
    output size: 128x112x112
relu2_1:
    output size: 128x112x112
conv2_2:
    128 filters, 3x3, pad 1, stride 1
    output size: 128x112x112
relu2_2:
    output size: 128x112x112
pool2:
    max, 2x2, stride 2, pad 0
    output size: 128x56x56
conv3_1:
    256 filters, 3x3, pad 1, stride 1
    output size: 256x56x56
relu3_1:
    output size: 256x56x56
conv3_2:
    256 filters, 3x3, pad 1, stride 1
    output size: 256x56x56
relu3_2:
    output size: 256x56x56
conv3_3:
    256 filters, 3x3, pad 1, stride 1
    output size: 256x56x56
relu3_3:
    output size: 256x56x56
pool3:
    max, 2x2, stride 2, pad 0
    output size: 256x28x28
conv4_1:
    512 filters, 3x3, pad 1, stride 1
    output size: 512x28x28
relu4_1:
    output size: 512x28x28
conv4_2:
    512 filters, 3x3, pad 1, stride 1
    output size: 512x28x28
relu4_2:
    output size: 512x28x28
conv4_3:
    512 filters, 3x3, pad 1, stride 1
    output size: 512x28x28
relu4_3:
    output size: 512x28x28
pool4:
    max, 2x2, stride 2, pad 0
    output size: 512x14x14
conv5_1:
    512 filters, 3x3, pad 1, stride 1
    output size: 512x14x14
relu5_1:
    output size: 512x14x14
conv5_2:
    512 filters, 3x3, pad 1, stride 1
    output size: 512x14x14
relu5_2:
    output size: 512x14x14
conv5_3:
    512 filters, 3x3, pad 1, stride 1
    output size: 512x14x14
relu5_3:
    output size: 512x14x14
pool5:
    max, 2x2, stride 2, pad 0
    output size: 512x7x7
FC6:
    4096
    output size: 4096x1
relu6:
    output size: 4096
drop6:
    dropout ratio: 0.5
    output size: 4096
FC7:
    4096
    output size: 4096
relu7:
    output size: 4096
drop7:
    dropout ratio: 0.5
    output size: 4096
FC8:
    1000
    output size: 1000
prob (softmax)
    outptu size: 1000
```


Parameters

- conv1-1: 64 x 3x3x3 + 64 = 1802
- conv1-2: 64 x 64x3x3 + 64 = 36928
- conv2-1: 128 x 64x3x3 + 128 = 73856
- conv2-2: 128 x 128x3x3 + 128 = 147584
- conv3-1: 256 x 128x3x3 + 256 = 294912
- conv3-2: 256 x 256x3x3 + 256 = 590080
- conv3-3: 256 x 256x3x3 + 256 = 590080
- conv4-1: 512 x 256x3x3 + 512 = 1180160
- conv4-2: 512 x 512x3x3 + 512 = 2359808
- conv4-3: 512 x 512x3x3 + 512 = 2359808
- conv5-1: 512 x 512x3x3 + 512 = 2359808
- conv5-2: 512 x 512x3x3 + 512 = 2359808
- conv5-3: 512 x 512x3x3 + 512 = 2359808
- FC6: 4096*512*7*7 + 4096     = 102764544
- FC7: 4096*4096+4096          = 16781312
- FC8: 1000*4096+1000          = 4097000

The above parameters do not contain spaces for gradient update
and momentum update.

# Datasets

- [The CIFAR-10 dataset][103], CIFAR-10 - Object Recognition in Images at [kaggle][104]


# todo
AlexNet, GoogleNet, ResNet, VGG, Inception

dropout, batch normalization, pooling

sgd

[CS230: Deep Learning][107]

[CSC321 Winter 2014 Introduction to Neural Networks and Machine Learning][113]
by Geoffrey	Hinton


# References
- [A step by step guide to Caffe][36]
- [Layers][65]
- [A guide to convolution arithmetic for deep learning][69], a paper, 2018, arxiv.org
- [Blobs, Layers, and Nets: anatomy of a Caffe model][93]
- <http://cs231n.github.io/neural-networks-case-study/>




[115]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/proto/caffe.proto#L225
[114]: https://github.com/BVLC/caffe/blob/master/src/caffe/solvers/rmsprop_solver.cpp
[113]: http://www.cs.toronto.edu/~tijmen/csc321/information.shtml
[112]: https://github.com/BVLC/caffe/blob/master/src/caffe/solvers/adagrad_solver.cpp
[111]: http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
[110]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/proto/caffe.proto#L698
[109]: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/dropout_layer.cpp
[108]: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/batch_norm_layer.cpp
[107]: http://cs230.stanford.edu/index.html
[106]: https://zhuanlan.zhihu.com/p/20599581
[105]: http://cs229.stanford.edu/notes/cs229-notes3.pdf
[104]: https://www.kaggle.com/c/cifar-10
[103]: http://www.cs.toronto.edu/~kriz/cifar.html
[102]: http://cs231n.github.io/classification/
[101]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/softmax_loss_layer.cpp#L118
[100]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/softmax_loss_layer.cpp#L89
[99]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/softmax_layer.cpp#L63
[98]: https://developer.apple.com/documentation/accelerate/1513065-cblas_sgemv?language=objc
[97]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/math_functions.cpp#L13
[96]: https://developer.apple.com/documentation/accelerate/1513264-cblas_sgemm?language=objc
[95]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/softmax_layer.cpp#L27
[94]: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/softmax_layer.cpp
[93]: http://caffe.berkeleyvision.org/tutorial/net_layer_blob.html
[92]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/layers/softmax_layer.hpp#L18
[91]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solver.cpp#L229
[90]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/net.cpp#L547
[89]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/net.hpp#L85
[88]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/layer.hpp#L424
[87]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/multinomial_logistic_loss_layer.cpp#L37
[86]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/multinomial_logistic_loss_layer.cpp#L20
[85]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/multinomial_logistic_loss_layer.cpp#L11
[84]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/layers/multinomial_logistic_loss_layer.hpp#L44
[83]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/loss_layer.cpp#L7
[82]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/layers/loss_layer.hpp#L32
[81]: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
[80]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/relu_layer.cpp#L22
[79]: https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
[78]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/relu_layer.cpp#L9
[77]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/tanh_layer.cpp#L22
[76]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/tanh_layer.cpp#L10
[75]: https://en.cppreference.com/w/cpp/numeric/math/tanh
[74]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/sigmoid_layer.cpp#L25
[73]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/sigmoid_layer.cpp#L14
[72]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/sigmoid_layer.cpp#L8
[71]: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/neuron_layer.cpp
[70]: https://github.com/BVLC/caffe/blob/master/include/caffe/layers/neuron_layer.hpp
[69]: https://arxiv.org/pdf/1603.07285.pdf
[68]: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md#dilated-convolution-animations
[67]: https://pytorch.org/docs/stable/nn.html#torch.nn.Conv1d
[66]: http://cs231n.github.io/convolutional-networks/
[65]: http://caffe.berkeleyvision.org/tutorial/layers.html
[64]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/examples/mnist/lenet_train_test.prototxt#L4
[63]: http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html
[62]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solver.cpp#L116
[61]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L168
[60]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/net.cpp#L735
[59]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L323
[58]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solver.cpp#L480
[57]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L222
[56]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L257
[55]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solver.cpp#L420
[54]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/layers/input_layer.cpp#L8
[53]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/examples/cpp_classification/classification.cpp#L176
[52]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/net.cpp#L98
[51]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/insert_splits.cpp#L12
[50]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/net.cpp#L46
[49]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/upgrade_proto.cpp#L1017
[48]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/upgrade_proto.cpp#L977
[47]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/upgrade_proto.cpp#L23
[46]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/io.cpp#L34
[45]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/upgrade_proto.cpp#L88
[44]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/adam_solver.cpp#L26
[43]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/nesterov_solver.cpp#L13
[42]: https://distill.pub/2017/momentum/
[41]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L11
[40]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L224
[39]: https://stats.stackexchange.com/questions/29130/difference-between-neural-net-weight-decay-and-learning-rate
[38]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L156
[37]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L88
[36]: http://shengshuyang.github.io/A-step-by-step-guide-to-Caffe.html
[35]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/proto/caffe.proto#L345
[34]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/proto/caffe.proto#L327
[33]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/net.cpp#L773
[32]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solver.cpp#L78
[31]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L73
[30]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/sgd_solvers.hpp#L18
[29]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/proto/caffe.proto#L216
[28]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/solvers/sgd_solver.cpp#L373
[27]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/sgd_solvers.hpp#L16
[26]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/solver.hpp#L42
[25]: https://github.com/BVLC/caffe/blob/master/include/caffe/solver_factory.hpp
[24]: https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet_solver.prototxt
[23]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L248
[22]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L229-L230
[21]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L236
[20]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L226
[19]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L215
[18]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L102
[17]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L29
[16]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L178
[15]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L176
[14]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L39
[13]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L126
[12]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L41
[11]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L46
[10]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L44
[9]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/io.cpp#L35
[8]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/io.cpp#L34
[7]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/util/upgrade_proto.cpp#L1119
[6]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L33
[5]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L166
[4]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L253
[3]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L429-L432
[2]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/tools/caffe.cpp#L435
[1]: https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/src/caffe/common.cpp#L45
