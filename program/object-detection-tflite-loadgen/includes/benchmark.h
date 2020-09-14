/*
 * Copyright (c) 2020 dividiti.
 * See CK COPYRIGHT.txt for copyright details.
 *
 * SPDX-License-Identifier: BSD-3-Clause.
 * See CK LICENSE.txt for licensing details.
 */

#ifndef BENCHMARK_H
#define BENCHMARK_H

#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <memory>
#include <string.h>
#include <vector>
#include <map>
#include <cwctype>
#include <locale>

#include <xopenme.h>

#define DEBUG(msg) std::cout << "DEBUG: " << msg << std::endl;

namespace CK {

    enum _TIMERS {
        X_TIMER_SETUP,
        X_TIMER_TEST,

        X_TIMER_COUNT
    };

    enum _VARS {
        X_VAR_TIME_SETUP,
        X_VAR_TIME_TEST,
        X_VAR_TIME_IMG_LOAD_TOTAL,
        X_VAR_TIME_IMG_LOAD_AVG,
        X_VAR_TIME_CLASSIFY_TOTAL,
        X_VAR_TIME_CLASSIFY_AVG,
        X_VAR_TIME_NON_MAX_SUPPRESSION_TOTAL,
        X_VAR_TIME_NON_MAX_SUPPRESSION_AVG,
        X_VAR_TIME_GRAPH_AVG,
        X_VAR_TIME_GRAPH_TOTAL,

        X_VAR_COUNT
    };


/// Store named value into xopenme variable.
    inline void store_value_f(int index, const char *name, float value) {
        char *json_name = new char[strlen(name) + 6];
        sprintf(json_name, "\"%s\":%%f", name);
        xopenme_add_var_f(index, json_name, value);
        delete[] json_name;
    }

/// Dummy `sprintf` like formatting function using std::string.
/// It uses buffer of fixed length so can't be used in any cases,
/// generally use it for short messages with numeric arguments.
    template<typename ...Args>
    inline std::string format(const char *str, Args ...args) {
        char buf[1024];
        sprintf(buf, str, args...);
        return std::string(buf);
    }

//----------------------------------------------------------------------

    class Accumulator {
    public:
        void reset() { _total = 0, _count = 0; }

        void add(float value) { _total += value, _count++; }

        float total() const { return _total; }

        float avg() const { return _count > 0 ? _total / static_cast<float>(_count): 0.0f; }

    private:
        float _total = 0;
        int _count = 0;
    };


//----------------------------------------------------------------------

class BenchmarkSession {
public:
  BenchmarkSession(BenchmarkSettings* settings): _settings(settings) {
  }

  virtual ~BenchmarkSession() {}

  const std::vector<std::string>& load_filenames(std::vector<size_t> img_indices) {
    _filenames_buffer.clear();
    _filenames_buffer.reserve( img_indices.size() );
    idx2loc.clear();

    auto list_of_available_imagefiles = _settings->list_of_available_imagefiles();
    auto count_available_imagefiles   = list_of_available_imagefiles.size();

    int loc=0;
    for (auto idx : img_indices) {
      if(idx<count_available_imagefiles) {
        _filenames_buffer.emplace_back(list_of_available_imagefiles[idx].name);
        idx2loc[idx] = loc++;
      } else {
        std::cerr << "Trying to load filename[" << idx << "] when only " << count_available_imagefiles << " images are available" << std::endl;
        exit(1);
      }
    }

    return _filenames_buffer;
  }

  const std::vector<std::string>& current_filenames() const { return _filenames_buffer; }

  std::map<int,int> idx2loc;

private:
  BenchmarkSettings* _settings;
  std::vector<std::string> _filenames_buffer;
};



//----------------------------------------------------------------------

    template<typename TData>
    class StaticBuffer {
    public:
        StaticBuffer(int size, const std::string &dir) : _size(size), _dir(dir) {
            _buffer = new TData[size];
        }

        virtual ~StaticBuffer() {
            delete[] _buffer;
        }

        TData *data() const { return _buffer; }

        int size() const { return _size; }

    protected:
        const int _size;
        const std::string _dir;
        TData *_buffer;
    };

//----------------------------------------------------------------------

    class ImageData : public StaticBuffer<uint8_t> {
    public:
        ImageData(BenchmarkSettings *s) : StaticBuffer(
                s->image_size_height() * s->image_size_width() * s->num_channels(), s->images_dir()) {}

      void load(const std::string& filename, int vl) {
        auto path = _dir + '/' + filename;
        std::ifstream file(path, std::ios::in | std::ios::binary);
        if (!file) throw "Failed to open image data " + path;
        file.read(reinterpret_cast<char*>(_buffer), _size);
        if( vl > 1) {
          std::cout << "Loaded file: " << path << std::endl;
        } else if ( vl ) {
          std::cout << 'l' << std::flush;
        }
      }
    };

//----------------------------------------------------------------------

    class ResultData {
    public:
        ResultData(BenchmarkSettings *s) : _size(0) {
            _buffer = new float[s->detections_buffer_size()*7];
        }

        ~ResultData() {
            delete[] _buffer;
        }

        int size() const { return _size; }

        void set_size(int size) { _size = size; }

        float *data() const { return _buffer; }

    private:
        float *_buffer;
        int _size;
    };

//----------------------------------------------------------------------

    class IBenchmark {
    public:
      bool has_background_class = false;

      virtual ~IBenchmark() {}
      virtual void load_images(BenchmarkSession *session) = 0;
      virtual void unload_images(size_t num_examples) = 0;
      virtual ResultData* get_next_result(int img_idx) = 0;
      virtual void get_random_image(int img_idx) = 0;
    };


    template<typename TInData, typename TOutData, typename TInConverter, typename TOutConverter>
    class Benchmark : public IBenchmark {
    public:


      Benchmark(BenchmarkSettings *settings,
                  TInData *in_ptr,
                  TOutData *boxes_ptr,
                  TOutData *classes_ptr,
                  TOutData *scores_ptr,
                  TOutData *num_ptr): _settings(settings) {
            _in_ptr = in_ptr;
            _boxes_ptr = boxes_ptr;
            _classes_ptr = classes_ptr;
            _scores_ptr = scores_ptr;
            _num_ptr = num_ptr;
            _in_converter.reset(new TInConverter(settings));
            _out_converter.reset(new TOutConverter(settings));
      }

      void load_images(BenchmarkSession *_session) override {
        session = _session;
        auto vl = _settings->get_verbosity_level();

        const std::vector<std::string>& image_filenames = session->current_filenames();

        int length = image_filenames.size();
        _current_buffer_size = length;
        _in_batch = new std::unique_ptr<ImageData>[length];
        _out_batch = new std::unique_ptr<ResultData>[length];
        int i = 0;
        for (auto image_file : image_filenames) {
          _in_batch[i].reset(new ImageData(_settings));
          _out_batch[i].reset(new ResultData(_settings));
          _in_batch[i]->load(image_file, vl);
          i++;
        }
      }

      void unload_images(size_t num_examples) override {
        for(size_t i=0;i<num_examples;i++) {
          delete _in_batch[i].get();
          delete _out_batch[i].get();
        }
      }

      void get_random_image(int img_idx) override {
        _in_converter->convert(_in_batch[ session->idx2loc[img_idx] ].get(), _in_ptr);
      }

      ResultData* get_next_result(int img_idx) override {
        //int probe_offset = has_background_class ? 1 : 0;
        ResultData *next_result_ptr = _out_batch[_out_buffer_index++].get();
        int offset = 0;
        int size = next_result_ptr->size();

        _out_converter->convert(img_idx,
                                _boxes_ptr + offset * size * 4,
                                _classes_ptr + offset * size,
                                _scores_ptr + offset * size,
                                _num_ptr + offset,
                                next_result_ptr,
                                _settings->image_size_width(),
                                _settings->image_size_height(),
                                _settings->model_classes(),
                                _settings->correct_background());

        _out_buffer_index %= _current_buffer_size;
        return next_result_ptr;
      }

    private:
      BenchmarkSettings* _settings;
      BenchmarkSession* session;
      int _out_buffer_index = 0;
      int _current_buffer_size = 0;
      TInData* _in_ptr;
      TOutData *_boxes_ptr;
      TOutData *_classes_ptr;
      TOutData *_scores_ptr;
      TOutData *_num_ptr;
      std::unique_ptr<ImageData> *_in_batch;
      std::unique_ptr<ResultData> *_out_batch;
      std::unique_ptr<TInConverter> _in_converter;
      std::unique_ptr<TOutConverter> _out_converter;
    };

//----------------------------------------------------------------------

    class IinputConverter {
    public:
      virtual ~IinputConverter() {}
      virtual void convert(ImageData* source, void* target) = 0;
    };


//----------------------------------------------------------------------

    class InCopy : public IinputConverter {
    public:
      InCopy(BenchmarkSettings* s) {}

      void convert(ImageData* source, void* target) {
        uint8_t *uint8_target = static_cast<uint8_t *>(target);
        std::copy(source->data(), source->data() + source->size(), uint8_target);
      }
    };

//----------------------------------------------------------------------

    class InNormalize : public IinputConverter {
    public:
        InNormalize(BenchmarkSettings *s) :
                _normalize_img(s->normalize_img()), _subtract_mean(s->subtract_mean()) {
        }

    void convert(ImageData* source, void* target) {
      // Copy image data to target
      float *float_target = static_cast<float *>(target);
            float sum = 0;
            for (int i = 0; i < source->size(); i++) {
                float px = source->data()[i];
                if (_normalize_img)
                    px = (px / 255.0 - 0.5) * 2.0;
                sum += px;
                float_target[i] = px;
            }
            // Subtract mean value if required
            if (_subtract_mean) {
                float mean = sum / static_cast<float>(source->size());
                for (int i = 0; i < source->size(); i++)
                    float_target[i] -= mean;
            }
        }

    private:
        const bool _normalize_img;
        const bool _subtract_mean;
    };


//----------------------------------------------------------------------

    void box_to_output(float image_index,
                       float x1,
                       float y1,
                       float x2,
                       float y2,
                       float score,
                       int detected_class,
                       float *buffer) {

        buffer[0] = image_index;
        buffer[1] = y1;
        buffer[2] = x1;
        buffer[3] = y2;
        buffer[4] = x2;
        buffer[5] = score;
        buffer[6] = detected_class+1;
    }

    class OutCopy {
    public:
        OutCopy(BenchmarkSettings *s): _settings(s) {}

        void convert(int img_idx,
                     const float *boxes,
                     const float *classes,
                     const float *scores,
                     const float *num,
                     ResultData *target,
                     int src_width,
                     int src_height,
                     std::vector<std::string> model_classes,
                     bool correct_background) const {

            float *buffer = target->data();
            target->set_size(*num*7);

            for (int i = 0; i < *num; i++) {
                float y1 = boxes[i * 4];
                float x1 = boxes[i * 4 + 1];
                float y2 = boxes[i * 4 + 2];
                float x2 = boxes[i * 4 + 3];
                float score = scores[i];
                int detected_class = int(classes[i]);

                std::string class_name = detected_class < model_classes.size()
                                 ? model_classes[detected_class]
                                 : "unknown";

                box_to_output(img_idx, x1, y1, x2, y2, score, classes[i], buffer);
                buffer += 7;
            }
        }
    private:
        BenchmarkSettings *_settings;
    };

//----------------------------------------------------------------------

    class OutDequantize {
    public:
        OutDequantize(BenchmarkSettings *s): _settings(s) {}

        void convert(int img_idx,
                     const uint8_t *boxes,
                     const uint8_t *classes,
                     const uint8_t *scores,
                     const uint8_t *num,
                     ResultData *target,
                     FileInfo src,
                     std::vector<std::string> model_classes,
                     bool correct_background) const {
            float *buffer = target->data();
            target->set_size(*num*7);
            if (*num == 0) 
            {
                return;
            }

            for (int i = 0; i < *num; i++) {
                float y1 = boxes[i * sizeof(float)] * src.height / 255.0f;
                float x1 = boxes[i * sizeof(float) + 1] * src.width / 255.0f;
                float y2 = boxes[i * sizeof(float) + 2] * src.height / 255.0f;
                float x2 = boxes[i * sizeof(float) + 3] * src.width / 255.0f;
                float score = scores[i] / 255.0f;
                int detected_class = int(classes[i]);

                box_to_output(img_idx, x1, y1, x2, y2, score, detected_class, buffer);
                buffer += 7;
            }
        }
    private:
        BenchmarkSettings *_settings;
    };

} // namespace CK

#endif
