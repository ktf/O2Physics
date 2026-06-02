// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file     model.h
///
/// \author   Christian Sonnabend <christian.sonnabend@cern.ch>
///
/// \brief    A general-purpose class for ONNX models
///

#ifndef TOOLS_ML_MODEL_H_
#define TOOLS_ML_MODEL_H_

#include <Framework/Logger.h>

#include <onnx/onnx_pb.h>
#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace o2
{

namespace ml
{

class OnnxModel
{

 public:
  OnnxModel() = default;
  ~OnnxModel() = default;

  // Inferencing
  void initModel(const std::string&, const bool = false, const int = 0, const uint64_t = 0, const uint64_t = 0);

  // template methods -- best to define them in header
  template <typename T>
  T* evalModel(std::vector<Ort::Value>& input)
  {
    LOG(debug) << "Input tensor shape: " << printShape(input[0].GetTensorTypeAndShapeInfo().GetShape());
    // assert(input[0].GetTensorTypeAndShapeInfo().GetShape() == getNumInputNodes()); --> Fails build in debug mode, TODO: assertion should be checked somehow

    try {
      const Ort::RunOptions runOptions;
      std::vector<const char*> inputNamesChar(mInputNames.size(), nullptr);
      std::transform(std::begin(mInputNames), std::end(mInputNames), std::begin(inputNamesChar),
                     [&](const std::string& str) { return str.c_str(); });

      std::vector<const char*> outputNamesChar(mOutputNames.size(), nullptr);
      std::transform(std::begin(mOutputNames), std::end(mOutputNames), std::begin(outputNamesChar),
                     [&](const std::string& str) { return str.c_str(); });
      auto outputTensors = mSession->Run(runOptions, inputNamesChar.data(), input.data(), input.size(), outputNamesChar.data(), outputNamesChar.size());
      LOG(debug) << "Number of output tensors: " << outputTensors.size();
      if (outputTensors.size() != mOutputNames.size()) {
        LOG(fatal) << "Number of output tensors: " << outputTensors.size() << " does not agree with the model specified size: " << mOutputNames.size();
      }
      for (std::size_t i = 0; i < outputTensors.size(); i++) {
        LOG(debug) << "Output tensor shape: " << printShape(outputTensors[i].GetTensorTypeAndShapeInfo().GetShape());
        if ((outputTensors[i].GetTensorTypeAndShapeInfo().GetShape() != mOutputShapes[i]) && (mOutputShapes[i][0] != -1)) {
          LOG(fatal) << "Shape of tensor " << i << " does not agree with model specification! Output: " << printShape(outputTensors[i].GetTensorTypeAndShapeInfo().GetShape()) << " model: " << printShape(mOutputShapes[i]);
        }
      }
      T* outputValues = outputTensors.back().GetTensorMutableData<T>();
      return outputValues;
    } catch (const Ort::Exception& exception) {
      LOG(error) << "Error running model inference: " << exception.what();
    }
    return nullptr;
  }

  template <typename T>
  T* evalModel(std::vector<T>& input)
  {
    const int64_t size = input.size();
    assert(size % mInputShapes[0][1] == 0);
    std::vector<int64_t> inputShape{size / mInputShapes[0][1], mInputShapes[0][1]};
    std::vector<Ort::Value> inputTensors;
    Ort::MemoryInfo memInfo =
      Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    inputTensors.emplace_back(Ort::Value::CreateTensor<T>(memInfo, input.data(), size, inputShape.data(), inputShape.size()));
    LOG(debug) << "Input shape calculated from vector: " << printShape(inputShape);
    return evalModel<T>(inputTensors);
  }

  // For 2D inputs
  template <typename T>
  T* evalModel(std::vector<std::vector<T>>& input)
  {
    std::vector<Ort::Value> inputTensors;

    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (std::size_t iinput = 0; iinput < input.size(); iinput++) {
      [[maybe_unused]] int totalSize = 1;
      int64_t size = input[iinput].size();
      for (std::size_t idim = 1; idim < mInputShapes[iinput].size(); idim++) {
        totalSize *= mInputShapes[iinput][idim];
      }
      assert(size % totalSize == 0);

      std::vector<int64_t> inputShape{static_cast<int64_t>(size / totalSize)};
      for (std::size_t idim = 1; idim < mInputShapes[iinput].size(); idim++) {
        inputShape.push_back(mInputShapes[iinput][idim]);
      }

      inputTensors.emplace_back(Ort::Value::CreateTensor<T>(memInfo, input[iinput].data(), size, inputShape.data(), inputShape.size()));
    }

    return evalModel<T>(inputTensors);
  }

  // Reset session
  void resetSession()
  {
    mSession.reset(new Ort::Session{*mEnv, modelPath.c_str(), sessionOptions});
  }

  // Declaration of a raw graph input fed directly from an Arrow buffer.
  struct PreprocInput {
    enum class Type { TrackFloat,     // per-track float column           [N]
                      TrackInt32,     // per-track int32 column           [N]
                      TrackUint8,     // per-track uint8 column           [N]
                      TrackInt8,      // per-track int8 column            [N]
                      TrackBool,      // per-track bool mask              [N]
                      CollisionFloat, // per-collision float array        [C]
                      ScalarFloat };  // single scalar (e.g. mass)        [1]
    std::string name;
    Type type;
  };

  // Preprocessing recipe for one network feature (produces a [N] float tensor
  // that feeds column i of the decomposed first layer).
  struct PreprocFeature {
    enum class Op {
      Passthrough,     // feature = a                                   (a: TrackFloat)
      BroadcastScalar, // feature = Expand(a, shape(shapeRef))          (a: ScalarFloat)
      NClsSqrtRecip,   // feature = Sqrt(c0 / (float(a) - float(b)))    (a,b: Track int cols)
      Mod2,            // feature = Mod(Mod(a, c0) + c1, c2)            (a: TrackFloat)
      GatherNormWhere  // feature = Where(b<0, fb, Gather(a, b) / c0)   (a: CollisionFloat, b: TrackInt32)
    };
    Op op;
    std::string a;             // primary input
    std::string b;             // secondary input (NCls: 2nd col; Gather: index)
    std::string shapeRef;      // BroadcastScalar: [N] input to size the Expand
    std::string fallbackInput; // GatherNormWhere: scalar input for the b<0 fallback; "" => use c[1]
    std::string scaleInput;    // NCls: numerator scalar input; Gather: divisor scalar input; "" => use c[0]
    std::array<float, 3> c{};  // op constants
  };

  // Rebuild the model from scratch so the network reads its raw Arrow inputs
  // directly and performs all preprocessing + the first linear layer inside the
  // graph.  Each feature column is produced by a small preprocessing subgraph
  // (`features`) from the raw inputs (`inputDefs`), then the first linear layer is
  // decomposed
  //   layer0 = X @ W + b = sum_i (feat_i[N,1] @ W_row_i[1,H]) + b
  // so no [N, K] interleaving / Concat buffer is ever materialised.  The original
  // layers 1..N are copied verbatim on top of the decomposed layer-0 output.
  // Building a fresh model (rather than augmenting the existing one) is required:
  // the Model Editor can only add nodes, so the original layer-0 Gemm would
  // otherwise remain and collide on the layer-0 output name.
  // If maskInput is non-empty it names a bool [N] input; each feature is then
  // Compress'd by it so the matmul runs only on the selected (valid) rows and the
  // output is the compact set of selected tracks, in order.
  void setupColumnInputs(const std::vector<PreprocInput>& inputDefs,
                         const std::vector<PreprocFeature>& features,
                         const std::string& maskInput = "")
  {
    const int numFeatures = static_cast<int>(features.size());
    if (numFeatures != mInputShapes[0][1]) {
      LOG(fatal) << "setupColumnInputs: expected " << mInputShapes[0][1] << " features, got " << numFeatures;
      return;
    }

    onnx::ModelProto onnxModel;
    {
      std::ifstream ifs(modelPath, std::ios::binary);
      if (!ifs || !onnxModel.ParseFromIstream(&ifs)) {
        LOG(fatal) << "setupColumnInputs: failed to parse ONNX model from " << modelPath;
        return;
      }
    }
    const auto& og = onnxModel.graph();

    int opset = 0;
    for (const auto& oi : onnxModel.opset_import()) {
      if (oi.domain().empty()) {
        opset = static_cast<int>(oi.version());
      }
    }
    if (opset == 0) {
      opset = 13; // Unsqueeze with axes as input requires opset >= 13
    }

    auto findInit = [&](const std::string& name) -> const onnx::TensorProto* {
      for (int i = 0; i < og.initializer_size(); ++i) {
        if (og.initializer(i).name() == name) {
          return &og.initializer(i);
        }
      }
      return nullptr;
    };
    auto tensorFloats = [&](const onnx::TensorProto* t, int64_t n) {
      std::vector<float> v(n);
      if (t->raw_data().size() > 0) {
        std::memcpy(v.data(), t->raw_data().data(), n * sizeof(float));
      } else {
        for (int64_t i = 0; i < n; ++i) {
          v[i] = t->float_data(i);
        }
      }
      return v;
    };

    // --- locate the first linear layer (layer 0) ---
    const onnx::NodeProto* first = nullptr;
    for (int i = 0; i < og.node_size(); ++i) {
      const auto& n = og.node(i);
      if (n.op_type() == "Gemm" || n.op_type() == "MatMul") {
        first = &n;
        break;
      }
    }
    if (!first) {
      LOG(fatal) << "setupColumnInputs: no Gemm/MatMul layer found in model";
      return;
    }
    const std::string layer0Out = first->output(0); // pre-activation output we must reproduce

    const onnx::TensorProto* wT = findInit(first->input(1));
    if (!wT || wT->dims_size() != 2) {
      LOG(fatal) << "setupColumnInputs: first-layer weight initializer not found or not 2D";
      return;
    }
    bool transB = false;
    if (first->op_type() == "Gemm") {
      for (int i = 0; i < first->attribute_size(); ++i) {
        if (first->attribute(i).name() == "transB") {
          transB = (first->attribute(i).i() != 0);
        }
      }
    }
    const int K = transB ? static_cast<int>(wT->dims(1)) : static_cast<int>(wT->dims(0));
    const int H = transB ? static_cast<int>(wT->dims(0)) : static_cast<int>(wT->dims(1));
    if (K != numFeatures) {
      LOG(fatal) << "setupColumnInputs: first-layer K=" << K << " != numFeatures=" << numFeatures;
      return;
    }
    const std::vector<float> wData = tensorFloats(wT, static_cast<int64_t>(K) * H);

    std::vector<float> bData;
    if (first->op_type() == "Gemm" && first->input_size() >= 3 && !first->input(2).empty()) {
      if (const onnx::TensorProto* bT = findInit(first->input(2))) {
        bData = tensorFloats(bT, H);
      }
    }

    // --- build the new graph ---
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::Graph graph;

    auto addFloatInit = [&](const std::string& name, const std::vector<float>& data,
                            const std::vector<int64_t>& shape) {
      auto val = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
      std::memcpy(val.GetTensorMutableData<float>(), data.data(), data.size() * sizeof(float));
      graph.AddInitializer(name, val, false);
    };

    // axes = {1} for Unsqueeze (opset >= 13 takes axes as an input)
    {
      std::vector<int64_t> shape = {1};
      auto val = Ort::Value::CreateTensor<int64_t>(allocator, shape.data(), shape.size());
      val.GetTensorMutableData<int64_t>()[0] = 1;
      graph.AddInitializer("_col_axes", val, false);
    }

    auto addScalarF = [&](const std::string& name, float v) {
      const std::vector<int64_t> shape = {1};
      auto val = Ort::Value::CreateTensor<float>(allocator, shape.data(), shape.size());
      val.GetTensorMutableData<float>()[0] = v;
      graph.AddInitializer(name, val, false);
    };
    addScalarF("_decZeroF", 0.f); // unused placeholder kept for symmetry
    {
      const std::vector<int64_t> shape = {1};
      auto val = Ort::Value::CreateTensor<int32_t>(allocator, shape.data(), shape.size());
      val.GetTensorMutableData<int32_t>()[0] = 0;
      graph.AddInitializer("_decZeroI32", val, false);
    }

    // --- raw graph inputs (wrapped zero-copy from Arrow at inference time) ---
    std::vector<Ort::ValueInfo> inputs;
    inputs.reserve(inputDefs.size());
    std::vector<std::string> rawInputNames;
    rawInputNames.reserve(inputDefs.size());
    for (const auto& pin : inputDefs) {
      ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
      std::vector<int64_t> dims = {-1};
      std::vector<std::string> sym = {"N"};
      switch (pin.type) {
        case PreprocInput::Type::TrackFloat:
          et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
          break;
        case PreprocInput::Type::TrackInt32:
          et = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
          break;
        case PreprocInput::Type::TrackUint8:
          et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
          break;
        case PreprocInput::Type::TrackInt8:
          et = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
          break;
        case PreprocInput::Type::TrackBool:
          et = ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
          break;
        case PreprocInput::Type::CollisionFloat:
          et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
          sym = {"C"};
          break;
        case PreprocInput::Type::ScalarFloat:
          et = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
          dims = {1};
          sym = {""};
          break;
      }
      Ort::TensorTypeAndShapeInfo tInfo(et, dims, &sym);
      auto typeInfo = Ort::TypeInfo::CreateTensorInfo(tInfo.GetConst());
      inputs.emplace_back(pin.name, typeInfo.GetConst());
      rawInputNames.push_back(pin.name);
    }

    // --- preprocessing subgraph: produce one [N] float feature tensor per column ---
    const int64_t toFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    const int64_t fmodFlag = 1;
    std::vector<std::string> featNames(K);
    for (int i = 0; i < K; ++i) {
      const auto& f = features[i];
      const std::string p = "_pp" + std::to_string(i);
      switch (f.op) {
        case PreprocFeature::Op::Passthrough: {
          featNames[i] = f.a; // the raw input is already the [N] feature
          break;
        }
        case PreprocFeature::Op::BroadcastScalar: {
          Ort::Node shapeNode("Shape", "", p + "_shape", {f.shapeRef}, {p + "_shp"});
          graph.AddNode(shapeNode);
          Ort::Node expandNode("Expand", "", p + "_expand", {f.a, p + "_shp"}, {p});
          graph.AddNode(expandNode);
          featNames[i] = p;
          break;
        }
        case PreprocFeature::Op::NClsSqrtRecip: {
          std::vector<Ort::OpAttr> ca;
          ca.emplace_back("to", &toFloat, 1, ORT_OP_ATTR_INT);
          Ort::Node castA("Cast", "", p + "_castA", {f.a}, {p + "_a"}, ca);
          graph.AddNode(castA);
          std::vector<Ort::OpAttr> cb;
          cb.emplace_back("to", &toFloat, 1, ORT_OP_ATTR_INT);
          Ort::Node castB("Cast", "", p + "_castB", {f.b}, {p + "_b"}, cb);
          graph.AddNode(castB);
          Ort::Node subNode("Sub", "", p + "_sub", {p + "_a", p + "_b"}, {p + "_ncl"});
          graph.AddNode(subNode);
          std::string numer = f.scaleInput;
          if (numer.empty()) {
            addScalarF(p + "_c0", f.c[0]);
            numer = p + "_c0";
          }
          Ort::Node divNode("Div", "", p + "_div", {numer, p + "_ncl"}, {p + "_recip"});
          graph.AddNode(divNode);
          Ort::Node sqrtNode("Sqrt", "", p + "_sqrt", {p + "_recip"}, {p});
          graph.AddNode(sqrtNode);
          featNames[i] = p;
          break;
        }
        case PreprocFeature::Op::Mod2: {
          addScalarF(p + "_c0", f.c[0]);
          addScalarF(p + "_c1", f.c[1]);
          addScalarF(p + "_c2", f.c[2]);
          std::vector<Ort::OpAttr> m1a;
          m1a.emplace_back("fmod", &fmodFlag, 1, ORT_OP_ATTR_INT);
          Ort::Node mod1("Mod", "", p + "_mod1", {f.a, p + "_c0"}, {p + "_m1"}, m1a);
          graph.AddNode(mod1);
          Ort::Node addNode("Add", "", p + "_add", {p + "_m1", p + "_c1"}, {p + "_m1a"});
          graph.AddNode(addNode);
          std::vector<Ort::OpAttr> m2a;
          m2a.emplace_back("fmod", &fmodFlag, 1, ORT_OP_ATTR_INT);
          Ort::Node mod2("Mod", "", p + "_mod2", {p + "_m1a", p + "_c2"}, {p}, m2a);
          graph.AddNode(mod2);
          featNames[i] = p;
          break;
        }
        case PreprocFeature::Op::GatherNormWhere: {
          // Gather tolerates the -1 ambiguous index (negative indexing); the Where
          // below discards that value in favour of the fallback, so no Clip needed.
          Ort::Node gatherNode("Gather", "", p + "_gather", {f.a, f.b}, {p + "_g"});
          graph.AddNode(gatherNode);
          std::string denom = f.scaleInput;
          if (denom.empty()) {
            addScalarF(p + "_c0", f.c[0]);
            denom = p + "_c0";
          }
          Ort::Node divNode("Div", "", p + "_div", {p + "_g", denom}, {p + "_gn"});
          graph.AddNode(divNode);
          Ort::Node lessNode("Less", "", p + "_less", {f.b, "_decZeroI32"}, {p + "_amb"});
          graph.AddNode(lessNode);
          std::string fbName;
          if (!f.fallbackInput.empty()) {
            fbName = f.fallbackInput; // runtime scalar input
          } else {
            addScalarF(p + "_fb", f.c[1]);
            fbName = p + "_fb";
          }
          Ort::Node whereNode("Where", "", p + "_where", {p + "_amb", fbName, p + "_gn"}, {p});
          graph.AddNode(whereNode);
          featNames[i] = p;
          break;
        }
      }
    }

    // --- decompose the first linear layer over the produced feature tensors ---
    // feat_i [N] -> Unsqueeze [N,1] -> MatMul [N,1]x[1,H] -> partial_i [N,H]
    for (int i = 0; i < K; ++i) {
      std::vector<float> row(H);
      for (int j = 0; j < H; ++j) {
        row[j] = transB ? wData[static_cast<size_t>(j) * K + i] : wData[static_cast<size_t>(i) * H + j];
      }
      addFloatInit("_w_row_" + std::to_string(i), row, {1, static_cast<int64_t>(H)});

      // Optionally drop the filtered-out rows before the matmul (compact output).
      std::string decompIn = featNames[i];
      if (!maskInput.empty()) {
        const int64_t axis0 = 0;
        std::vector<Ort::OpAttr> ca;
        ca.emplace_back("axis", &axis0, 1, ORT_OP_ATTR_INT);
        decompIn = "_cf_" + std::to_string(i);
        Ort::Node compressNode("Compress", "", "_compress_" + std::to_string(i),
                               {featNames[i], maskInput}, {decompIn}, ca);
        graph.AddNode(compressNode);
      }

      const std::string unsq = "_col_unsq_" + std::to_string(i);
      Ort::Node unsqNode("Unsqueeze", "", "_unsqueeze_" + std::to_string(i),
                         {decompIn, "_col_axes"}, {unsq});
      graph.AddNode(unsqNode);
      Ort::Node matmulNode("MatMul", "", "_matmul_" + std::to_string(i),
                           {unsq, "_w_row_" + std::to_string(i)}, {"_partial_" + std::to_string(i)});
      graph.AddNode(matmulNode);
    }

    // sum the partials; the final op must produce layer0Out
    std::string acc = "_partial_0";
    for (int i = 1; i < K; ++i) {
      const bool last = (i == K - 1);
      const std::string out = (last && bData.empty()) ? layer0Out : ("_sum_" + std::to_string(i));
      Ort::Node addNode("Add", "", "_add_" + std::to_string(i),
                        {acc, "_partial_" + std::to_string(i)}, {out});
      graph.AddNode(addNode);
      acc = out;
    }
    if (!bData.empty()) {
      addFloatInit("_dec_bias", bData, {static_cast<int64_t>(H)});
      Ort::Node biasNode("Add", "", "_add_bias", {acc, "_dec_bias"}, {layer0Out});
      graph.AddNode(biasNode);
    } else if (K == 1) {
      Ort::Node idNode("Identity", "", "_id_layer0", {"_partial_0"}, {layer0Out});
      graph.AddNode(idNode);
    }

    // --- copy original layers 1..N: every node reachable backwards from the graph
    //     outputs, except the replaced first layer (and any node, e.g. an input
    //     Cast, that fed only into it and is now dead) ---
    std::set<std::string> live;
    for (int i = 0; i < og.output_size(); ++i) {
      live.insert(og.output(i).name());
    }
    std::vector<bool> keep(og.node_size(), false);
    bool changed = true;
    while (changed) {
      changed = false;
      for (int i = 0; i < og.node_size(); ++i) {
        if (keep[i] || &og.node(i) == first) {
          continue;
        }
        const auto& n = og.node(i);
        bool produces = false;
        for (int o = 0; o < n.output_size(); ++o) {
          if (live.count(n.output(o))) {
            produces = true;
            break;
          }
        }
        if (!produces) {
          continue;
        }
        keep[i] = true;
        changed = true;
        for (int in = 0; in < n.input_size(); ++in) {
          live.insert(n.input(in));
        }
      }
    }

    int copiedNodes = 0;
    std::set<std::string> neededInits;
    for (int i = 0; i < og.node_size(); ++i) {
      if (!keep[i]) {
        continue;
      }
      const auto& n = og.node(i);
      const std::vector<std::string> ins(n.input().begin(), n.input().end());
      const std::vector<std::string> outs(n.output().begin(), n.output().end());
      std::vector<Ort::OpAttr> attrs;
      for (int a = 0; a < n.attribute_size(); ++a) {
        const auto& at = n.attribute(a);
        switch (at.type()) {
          case onnx::AttributeProto::FLOAT: {
            const float v = at.f();
            attrs.emplace_back(at.name().c_str(), &v, 1, ORT_OP_ATTR_FLOAT);
            break;
          }
          case onnx::AttributeProto::INT: {
            const int64_t v = at.i();
            attrs.emplace_back(at.name().c_str(), &v, 1, ORT_OP_ATTR_INT);
            break;
          }
          case onnx::AttributeProto::FLOATS: {
            const std::vector<float> v(at.floats().begin(), at.floats().end());
            attrs.emplace_back(at.name().c_str(), v.data(), static_cast<int>(v.size()), ORT_OP_ATTR_FLOATS);
            break;
          }
          case onnx::AttributeProto::INTS: {
            const std::vector<int64_t> v(at.ints().begin(), at.ints().end());
            attrs.emplace_back(at.name().c_str(), v.data(), static_cast<int>(v.size()), ORT_OP_ATTR_INTS);
            break;
          }
          case onnx::AttributeProto::STRING: {
            const std::string& s = at.s();
            attrs.emplace_back(at.name().c_str(), s.data(), static_cast<int>(s.size()), ORT_OP_ATTR_STRING);
            break;
          }
          default:
            LOG(fatal) << "setupColumnInputs: unhandled attribute type " << at.type() << " on node " << n.name();
            return;
        }
      }
      Ort::Node node(n.op_type(), n.domain(),
                     n.name().empty() ? ("_copy_" + std::to_string(i)) : n.name(), ins, outs, attrs);
      graph.AddNode(node);
      ++copiedNodes;
      for (const auto& in : ins) {
        if (findInit(in)) {
          neededInits.insert(in);
        }
      }
    }

    for (const auto& name : neededInits) {
      const onnx::TensorProto* t = findInit(name);
      std::vector<int64_t> shape(t->dims().begin(), t->dims().end());
      int64_t n = 1;
      for (const auto d : shape) {
        n *= d;
      }
      if (t->data_type() != onnx::TensorProto::FLOAT) {
        LOG(fatal) << "setupColumnInputs: unsupported initializer dtype " << t->data_type() << " for " << name;
        return;
      }
      addFloatInit(name, tensorFloats(t, n), shape);
    }

    // graph outputs (copy name + tensor type/shape from the original model)
    std::vector<Ort::ValueInfo> outputs;
    for (int i = 0; i < og.output_size(); ++i) {
      const auto& vp = og.output(i);
      const auto& tt = vp.type().tensor_type();
      std::vector<int64_t> dims;
      std::vector<std::string> sym;
      for (int d = 0; d < tt.shape().dim_size(); ++d) {
        const auto& dd = tt.shape().dim(d);
        if (dd.has_dim_value()) {
          dims.push_back(dd.dim_value());
          sym.emplace_back("");
        } else {
          dims.push_back(-1);
          sym.push_back(dd.dim_param().empty() ? ("d" + std::to_string(d)) : dd.dim_param());
        }
      }
      Ort::TensorTypeAndShapeInfo tInfo(static_cast<ONNXTensorElementDataType>(tt.elem_type()), dims, &sym);
      auto typeInfo = Ort::TypeInfo::CreateTensorInfo(tInfo.GetConst());
      outputs.emplace_back(vp.name(), typeInfo.GetConst());
    }

    graph.SetInputs(inputs);
    graph.SetOutputs(outputs);

    // Declare the ONNX domain plus the ORT contrib domain: with graph
    // optimizations enabled ORT fuses Gemm+activation into com.microsoft FusedGemm
    // ops, which need an opset import for that domain in a Model-Editor-built model.
    Ort::Model model({{std::string(), opset}, {std::string("com.microsoft"), 1}});
    model.AddGraph(graph);
    mSession = std::make_shared<Ort::Session>(*mEnv, model, sessionOptions);

    mInputNames = rawInputNames;
    mInputShapes.assign(rawInputNames.size(), std::vector<int64_t>{-1});
    mNumFeatures = K;

    LOG(info) << "setupColumnInputs: rebuilt model with " << rawInputNames.size() << " raw inputs -> "
              << K << " preprocessed features, layer 0 decomposed (H=" << H << "), "
              << copiedNodes << " downstream nodes copied";
  }

  // Getters & Setters
  Ort::SessionOptions* getSessionOptions() { return &sessionOptions; } // For optimizations in post
  std::shared_ptr<Ort::Session> getSession()
  {
    return mSession;
  }
  int getNumInputNodes() const { return mInputShapes[0][1]; }
  bool hasColumnInputs() const { return mInputShapes.size() > 1 || (mInputShapes.size() == 1 && mInputShapes[0].size() == 1); }
  int getNumColumns() const { return static_cast<int>(mInputNames.size()); }
  int getNumFeatures() const { return mNumFeatures; }
  std::vector<std::vector<int64_t>> getInputShapes() const { return mInputShapes; }
  int getNumOutputNodes() const { return mOutputShapes[0][1]; }
  uint64_t getValidityFrom() const { return validFrom; }
  uint64_t getValidityUntil() const { return validUntil; }
  void setActiveThreads(const int);

 private:
  // Environment variables for the ONNX runtime
  std::shared_ptr<Ort::Env> mEnv = nullptr;
  std::shared_ptr<Ort::Session> mSession = nullptr;
  Ort::SessionOptions sessionOptions;

  // Input & Output specifications of the loaded network
  std::vector<std::string> mInputNames;
  std::vector<std::vector<int64_t>> mInputShapes;
  int mNumFeatures = 0;
  std::vector<std::string> mOutputNames;
  std::vector<std::vector<int64_t>> mOutputShapes;

  // Environment settings
  std::string modelPath;
  int activeThreads = 0;
  uint64_t validFrom = 0;
  uint64_t validUntil = 0;

  // Internal function for printing the shape of tensors
  std::string printShape(const std::vector<int64_t>&);
  bool checkHyperloop(const bool = true);
};

} // namespace ml

} // namespace o2

#endif // TOOLS_ML_MODEL_H_
