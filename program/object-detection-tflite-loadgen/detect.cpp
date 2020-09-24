/*
 * Copyright (c) 2020 cTuning foundation.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#include "includes/detect.hpp"

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"

#ifdef USE_EDGETPU

#include "includes/edgetpu.h"

std::unique_ptr<tflite::Interpreter> BuildEdgeTpuInterpreter(
      const tflite::FlatBufferModel& model,
      tflite::ops::builtin::BuiltinOpResolver resolver,
      edgetpu::EdgeTpuContext* edgetpu_context) {

    resolver.AddCustom(edgetpu::kCustomOp, edgetpu::RegisterCustomOp());

    std::unique_ptr<tflite::Interpreter> interpreter;

    if (tflite::InterpreterBuilder(model, resolver)(&interpreter) != kTfLiteOk) {
      std::cerr << "Failed to build interpreter." << std::endl;
    }

    // Bind given context with interpreter.
    interpreter->SetExternalContext(kTfLiteEdgeTpuContext, edgetpu_context);

    if (interpreter->AllocateTensors() != kTfLiteOk) {
      std::cerr << "Failed to allocate tensors." << std::endl;
    }
    return interpreter;
}

#endif


class Program {
public:
  Program () {


    settings = new BenchmarkSettings();

    session = new BenchmarkSession(settings);

    if (!settings->graph_file().c_str()) {
        throw string("Model file name is empty");
    }

    if (settings->batch_size() != 1)
        throw string("Only BATCH_SIZE=1 is currently supported");

    cout << endl << "Loading graph..." << endl;

    model = FlatBufferModel::BuildFromFile(settings->graph_file().c_str());

    if (!model)
        throw "Failed to load graph from file " + settings->graph_file();
    if (settings->verbose()) {
        cout << "Loaded model " << settings->graph_file() << endl;
        model->error_reporter();
        cout << "resolved reporter" << endl;
        cout << endl << "Number of threads: " << settings->number_of_threads() << endl;
    }


    ops::builtin::BuiltinOpResolver resolver;

#ifdef USE_EDGETPU
    edgetpu_context = edgetpu::EdgeTpuManager::GetSingleton()->OpenDevice();
    interpreter = BuildEdgeTpuInterpreter(*model, resolver, edgetpu_context.get());
#else
    InterpreterBuilder(*model, resolver)(&interpreter);
#endif

    if (!interpreter)
        throw string("Failed to construct interpreter");
    if (interpreter->AllocateTensors() != kTfLiteOk)
        throw string("Failed to allocate tensors");

    int input_size = interpreter->inputs().size();
    int output_size = interpreter->outputs().size();

    if (settings->verbose()) {
        cout << "tensors size: " << interpreter->tensors_size() << endl;
        cout << "nodes size: " << interpreter->nodes_size() << endl;
        cout << "number of inputs: " << input_size << endl;
        cout << "number of outputs: " << output_size << endl;
        cout << "input(0) name: " << interpreter->GetInputName(0) << endl;

        int t_size = interpreter->tensors_size();
        for (int i = 0; i < t_size; i++) {
            if (interpreter->tensor(i)->name)
                cout << i << ": " << interpreter->tensor(i)->name << ", "
                     << interpreter->tensor(i)->bytes << ", "
                     << interpreter->tensor(i)->type << ", "
                     << interpreter->tensor(i)->params.scale << ", "
                     << interpreter->tensor(i)->params.zero_point << endl;
        }
    }

    interpreter->SetNumThreads(settings->number_of_threads());

    if (settings->verbose()) PrintInterpreterState(interpreter.get());

    int input_index = interpreter->inputs()[0];
    int detection_boxes_id = interpreter->outputs()[0];
    int detection_classes_id = interpreter->outputs()[1];
    int detection_scores_id = interpreter->outputs()[2];
    auto input_type = interpreter->tensor(input_index)->type;

    switch (input_type) {
        case kTfLiteFloat32:

            benchmark.reset(new TFLiteBenchmark<float, float, InNormalize, OutCopy>(settings, interpreter.get(),
                                                                             input_index));
            break;

        case kTfLiteUInt8:
            benchmark.reset(new TFLiteBenchmark<uint8_t, float, InCopy, OutCopy>(settings, interpreter.get(),
                                                                                input_index));
            break;

        default:
            throw format("Unsupported type of graph's input: %d. "
                         "Supported types are: Float32 (%d), UInt8 (%d)",
                         int(input_type), int(kTfLiteFloat32), int(kTfLiteUInt8));
    }

    TfLiteIntArray *in_dims = interpreter->tensor(input_index)->dims;
    int in_num = in_dims->data[0];
    int in_height = in_dims->data[1];
    int in_width = in_dims->data[2];
    int in_channels = in_dims->data[3];

    if (in_height != settings->image_size_height() ||
        in_width != settings->image_size_width() ||
        in_channels != settings->num_channels())
        throw format("Dimensions of graph's input do not correspond to dimensions of input image (%d*%d*%d*%d)",
                     settings->batch_size(),
                     settings->image_size_height(),
                     settings->image_size_width(),
                     settings->num_channels());

    int frames = 1;
    TfLiteIntArray *detection_boxes_ptr = interpreter->tensor(detection_boxes_id)->dims;
    int boxes_count = detection_boxes_ptr->data[1];
    int boxes_length = detection_boxes_ptr->data[2];

    TfLiteIntArray *detection_classes_ptr = interpreter->tensor(detection_classes_id)->dims;
    int classes_count = detection_classes_ptr->data[1];

    TfLiteIntArray *detection_scores_ptr = interpreter->tensor(detection_scores_id)->dims;
    int scores_count = detection_scores_ptr->data[1];

    if (settings->get_verbosity_level()) {
        cout << format("Input tensor dimensions (NHWC): %d*%d*%d*%d", in_num, in_height, in_width, in_channels)
             << endl;
        cout << format("Detection boxes tensor dimensions: %d*%d*%d", frames, boxes_count, boxes_length)
             << endl;
        cout << format("Detection classes tensor dimensions: %d*%d", frames, classes_count) << endl;
        cout << format("Detection scores tensor dimensions: %d*%d", frames, scores_count) << endl;
        cout << format("Number of detections tensor dimensions: %d*1", frames) << endl;
    }
  }

  ~Program() {
      interpreter.reset();
#ifdef USE_EDGETPU
      edgetpu_context.reset();
#endif
  }

  //bool is_available_batch() {return session? session->get_next_batch(): false; }

  void LoadNextBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices) {
    auto vl = settings->get_verbosity_level();

    if( vl > 1 ) {
      cout << "LoadNextBatch([";
      for( auto idx : img_indices) {
        cout << idx << ' ';
      }
      cout << "])" << endl;
    } else if( vl ) {
      cout << 'B' << flush;
    }
    session->load_filenames(img_indices);
    benchmark->load_images( session );
    if( vl ) {
      cout << endl;
    }
  }

  void ColdRun() {
    auto vl = settings->get_verbosity_level();

    if( vl > 1 ) {
      cout << "Triggering a Cold Run..." << endl;
    } else if( vl ) {
      cout << 'C' << flush;
    }

    if (interpreter->Invoke() != kTfLiteOk)
      throw "Failed to invoke tflite";
  }

  ResultData* InferenceOnce(int img_idx) {
    benchmark->get_random_image( img_idx );
    if (interpreter->Invoke() != kTfLiteOk)
      throw "Failed to invoke tflite";
    return benchmark->get_next_result(img_idx);
  }

  void UnloadBatch(const std::vector<mlperf::QuerySampleIndex>& img_indices) {
    auto b_size = img_indices.size();

    auto vl = settings->get_verbosity_level();

    if( vl > 1 ) {
      cout << "Unloading a batch[" << b_size << "]" << endl;
    } else if( vl ) {
      cout << 'U' << flush;
    }

    benchmark->unload_images(b_size);
    //benchmark->save_results( );
  }

  const int available_images_max() { return settings->list_of_available_imagefiles().size(); }
  const int images_in_memory_max() { return settings->images_in_memory_max; }

  BenchmarkSettings *settings;
private:
  BenchmarkSession *session;
  unique_ptr<IBenchmark> benchmark;
  unique_ptr<tflite::Interpreter> interpreter;
  unique_ptr<tflite::FlatBufferModel> model;
#ifdef USE_EDGETPU
  shared_ptr<edgetpu::EdgeTpuContext> edgetpu_context;
#endif
};


class SystemUnderTestSingleStream : public mlperf::SystemUnderTest {
public:
  SystemUnderTestSingleStream(Program *_prg) : mlperf::SystemUnderTest() {
    prg = _prg;
    query_counter = 0;
  };

  ~SystemUnderTestSingleStream() override = default;

  const std::string& Name() const override { return name_; }

  void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override {

    ++query_counter;
    auto vl = prg->settings->get_verbosity_level();
    if( vl > 1 ) {
      cout << query_counter << ") IssueQuery([" << samples.size() << "]," << samples[0].id << "," << samples[0].index << ")" << endl;
    } else if ( vl ) {
      cout << 'Q' << flush;
    }

    std::vector<mlperf::QuerySampleResponse> responses;
    responses.reserve(samples.size());
    float* encoding_buffer[samples.size()];
    size_t encoding_buffer_len[samples.size()];
    int i=0;
    for (auto s : samples) {
      ResultData* predicted_class = prg->InferenceOnce(s.index);
      if( vl > 1 ) {
        cout << "Query image index: " << s.index << " -> Predicted class: " << *predicted_class->data() << endl << endl;
      } else if ( vl ) {
        cout << 'p' << flush;
      }

      /* This would be the correct way to pass in one integer index:
      */
//      int single_value_buffer[] = { (int)predicted_class };

      /* This conversion is subtly but terribly wrong
         yet we use it here in order to use Guenther's parsing script:
      */

      encoding_buffer[i] = new float[predicted_class->size()];
      memcpy(encoding_buffer[i], predicted_class->data(), predicted_class->size()*sizeof(float));
      encoding_buffer_len[i] = predicted_class->size();
      responses.push_back({s.id, uintptr_t(encoding_buffer[i]), encoding_buffer_len[i]*sizeof(float)});
      ++i;
    }
    mlperf::QuerySamplesComplete(responses.data(), responses.size());
    for( int i=0 ; i<samples.size() ; ++i) delete encoding_buffer[i];
  }

  void FlushQueries() override {
    auto vl = prg->settings->get_verbosity_level();
    if ( vl ) {
      cout << endl;
    }
  }

  void ReportLatencyResults(const std::vector<mlperf::QuerySampleLatency>& latencies_ns) override {

    size_t size = latencies_ns.size();
    uint64_t avg = accumulate(latencies_ns.begin(), latencies_ns.end(), uint64_t(0) )/size;

    std::vector<mlperf::QuerySampleLatency> sorted_lat(latencies_ns.begin(), latencies_ns.end());
    sort(sorted_lat.begin(), sorted_lat.end());

    cout << endl << "------------------------------------------------------------";
    cout << endl << "|            LATENCIES (in nanoseconds and fps)            |";
    cout << endl << "------------------------------------------------------------";
    size_t p50 = size * 0.5;
    size_t p90 = size * 0.9;
    cout << endl << "Number of queries run: " << size;
    cout << endl << "Min latency:                      " << sorted_lat[0]       << "ns  (" << 1e9/sorted_lat[0]         << " fps)";
    cout << endl << "Median latency:                   " << sorted_lat[p50]     << "ns  (" << 1e9/sorted_lat[p50]       << " fps)";
    cout << endl << "Average latency:                  " << avg                 << "ns  (" << 1e9/avg                   << " fps)";
    cout << endl << "90 percentile latency:            " << sorted_lat[p90]     << "ns  (" << 1e9/sorted_lat[p90]       << " fps)";

    if(!prg->settings->trigger_cold_run) {
      cout << endl << "First query (cold model) latency: " << latencies_ns[0]     << "ns  (" << 1e9/latencies_ns[0]       << " fps)";
    }
    cout << endl << "Max latency:                      " << sorted_lat[size-1]  << "ns  (" << 1e9/sorted_lat[size-1]    << " fps)";
    cout << endl << "------------------------------------------------------------ " << endl;
  }

private:
  std::string name_{"TFLite_SUT"};
  Program *prg;
  long query_counter;
};

class QuerySampleLibrarySingleStream : public mlperf::QuerySampleLibrary {
public:
  QuerySampleLibrarySingleStream(Program *_prg) : mlperf::QuerySampleLibrary() {
    prg = _prg;
  };

  ~QuerySampleLibrarySingleStream() = default;

  const std::string& Name() const override { return name_; }

  size_t TotalSampleCount() override { return prg->available_images_max(); }

  size_t PerformanceSampleCount() override { return prg->images_in_memory_max(); }

  void LoadSamplesToRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
    prg->LoadNextBatch(samples);
    return;
  }

  void UnloadSamplesFromRam( const std::vector<mlperf::QuerySampleIndex>& samples) override {
    prg->UnloadBatch(samples);
    return;
  }

private:
  std::string name_{"TFLite_QSL"};
  Program *prg;
};

void TestSingleStream(Program *prg) {
  SystemUnderTestSingleStream sut(prg);
  QuerySampleLibrarySingleStream qsl(prg);

  const std::string mlperf_conf_path = getenv_s("CK_ENV_MLPERF_INFERENCE_MLPERF_CONF");
  const std::string user_conf_path = getenv_s("CK_LOADGEN_USER_CONF");

  std::string model_name = getenv_opt_s("ML_MODEL_MODEL_NAME", "unknown_model");

  const std::string scenario_string = getenv_s("CK_LOADGEN_SCENARIO");
  const std::string mode_string = getenv_s("CK_LOADGEN_MODE");

  std::cout << "Path to mlperf.conf : " << mlperf_conf_path << std::endl;
  std::cout << "Path to user.conf : " << user_conf_path << std::endl;
  std::cout << "Model Name: " << model_name << std::endl;
  std::cout << "LoadGen Scenario: " << scenario_string << std::endl;
  std::cout << "LoadGen Mode: " << ( mode_string != "" ? mode_string : "(empty string)" ) << std::endl;

  mlperf::TestSettings ts;

  // This should have been done automatically inside ts.FromConfig() !
  ts.scenario = ( scenario_string == "SingleStream")    ? mlperf::TestScenario::SingleStream
              : ( scenario_string == "MultiStream")     ? mlperf::TestScenario::MultiStream
              : ( scenario_string == "MultiStreamFree") ? mlperf::TestScenario::MultiStreamFree
              : ( scenario_string == "Server")          ? mlperf::TestScenario::Server
              : ( scenario_string == "Offline")         ? mlperf::TestScenario::Offline : mlperf::TestScenario::SingleStream;

  if( mode_string != "")
    ts.mode   = ( mode_string == "SubmissionRun")       ? mlperf::TestMode::SubmissionRun
              : ( mode_string == "AccuracyOnly")        ? mlperf::TestMode::AccuracyOnly
              : ( mode_string == "PerformanceOnly")     ? mlperf::TestMode::PerformanceOnly
              : ( mode_string == "FindPeakPerformance") ? mlperf::TestMode::FindPeakPerformance : mlperf::TestMode::SubmissionRun;

  if (ts.FromConfig(mlperf_conf_path, model_name, scenario_string)) {
    std::cout << "Issue with mlperf.conf file at " << mlperf_conf_path << std::endl;
    exit(1);
  }

  if (ts.FromConfig(user_conf_path, model_name, scenario_string)) {
    std::cout << "Issue with user.conf file at " << user_conf_path << std::endl;
    exit(1);
  }

  mlperf::LogSettings log_settings;
  log_settings.log_output.prefix_with_datetime = false;
  log_settings.enable_trace = false;

  if (prg->settings->trigger_cold_run) {
    prg->ColdRun();
  }

  mlperf::StartTest(&sut, &qsl, ts, log_settings);
}



int main(int argc, char* argv[]) {
  try {
    Program *prg = new Program();
    TestSingleStream(prg);
    delete prg;
  }
  catch (const string& error_message) {
    cerr << "ERROR: " << error_message << endl;
    return -1;
  }
  return 0;
}





