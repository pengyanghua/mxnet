/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */




#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include <chrono>
#include <set>
#include "ps/ps.h"
#include "mxnet/kvstore.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"
#include <fstream>
#include <unistd.h>
#include <ctime>
namespace mxnet {
namespace kvstore {

enum class CommandType {
  kController, kStopServer, kSyncMode, kSetGradientCompression, kGetParaInfo, kSetParams, kStartTraining, kRequestParaInfo, kGetStragInfo,kRedistribute
};  // further define kGetParaInfo

enum class DataHandleType {
  kDefaultPushPull, kCompressedPushPull, kRowSparsePushPull, kMoveParams
};

struct ParamInfo {
	double speed;
	int step;
	// server_id:<key,size><key,size>
	std::unordered_map<int, std::vector<std::pair<int, int>>> ps_kvs;
};

/*
std::string split implementation by using delimiter as a character.
*/
std::vector<std::string> SplitStr(const std::string strToSplit, char delimeter)
{
    std::stringstream ss(strToSplit);
    std::string item;
    std::vector<std::string> splittedStrings;
    while (std::getline(ss, item, delimeter))
    {
       splittedStrings.push_back(item);
    }
    return splittedStrings;
}

// convert ParamInfo to string
// string: "speed step server_id,key:size,key:size, server_id,key:size,key:size, "
std::string ParamInfoToString(const ParamInfo& param_info){
	std::string body;
	const auto& speed_str = std::to_string(param_info.speed);
	body.append(speed_str).append(" ");
	const auto& step_str = std::to_string(param_info.step);
	body.append(step_str).append(" ");
	LOG(INFO) << "Converting ParamInfo to string";
	for(const auto& server_kvs : param_info.ps_kvs){
		int server_id = server_kvs.first;
		body.append(std::to_string(server_id)).append(",");
		const std::vector<std::pair<int, int>>& kvs = server_kvs.second;
		for (const auto& elem : kvs){
			int key = elem.first; // the key after decoding
			int size = elem.second;
			auto key_size_str = std::to_string(key).append(":").append(std::to_string(size));
			body.append(key_size_str).append(",");
		}
		body.append(" ");
	}
	return body;
}
/*ParamInfo StringToParamInfo_Scaling(const std::string& body){
	ParamInfo param_info;
	std::vector<std::string> ps_metas = SplitStr(body,' ');
	param_info.speed = std::stod(ps_metas.at(1));
	param_info.step = std::stoi(ps_metas.at(2));
	std::vector<std::string> key_map_metas = SplitStr(ps_metas.at(0),'_');
	for(int i=0;i<key_map_metas.size();i++){
		auto& keymap = key_map_metas.at(i);
		if(keymap.size()){
			std::vector<std::string> keymap_strs = SplitStr(keymap,',');
			key_map_[std::stoi(keymap_strs.at(0))] = std::stoi(keymap_strs.at(1)); 
		}
	}

    for(size_t i=3; i<ps_metas.size(); i++){
            auto& ps_meta = ps_metas.at(i);
            if (ps_meta.size()){ // the last elem is empty str
                std::vector<std::string> key_size_strs = SplitStr(ps_meta, ',');
                int server_id = std::stoi(key_size_strs.at(0));
                for (size_t j=1; j<key_size_strs.size(); j++){
                        auto& key_size_str = key_size_strs.at(j);
                        if(key_size_str.size()){ // the last elem is empty str ,
                                std::vector<std::string> key_size = SplitStr(key_size_str, ':');
                                int key = std::stoi(key_size.at(0));
                                int size = std::stoi(key_size.at(1));
                                param_info.ps_kvs[server_id].push_back(std::make_pair(key,size));
                        }
                }
            }
    }
    return param_info;	
}*/

// convert string to ParamInfo
ParamInfo StringToParamInfo(const std::string& body){
	ParamInfo param_info;
	std::vector<std::string> ps_metas = SplitStr(body, ' ');
	param_info.speed = std::stod(ps_metas.at(0));
	param_info.step = std::stoi(ps_metas.at(1));

	LOG(INFO) << "Converting string to ParamInfo";

    for(size_t i=2; i<ps_metas.size(); i++){
	    auto& ps_meta = ps_metas.at(i);
	    if (ps_meta.size()){ // the last elem is empty str
	    	std::vector<std::string> key_size_strs = SplitStr(ps_meta, ',');
	    	int server_id = std::stoi(key_size_strs.at(0));
	    	for (size_t j=1; j<key_size_strs.size(); j++){
	    		auto& key_size_str = key_size_strs.at(j);
	    		if(key_size_str.size()){ // the last elem is empty str ,
	    			std::vector<std::string> key_size = SplitStr(key_size_str, ':');
	    			int key = std::stoi(key_size.at(0));
	    			int size = std::stoi(key_size.at(1));
	    			param_info.ps_kvs[server_id].push_back(std::make_pair(key,size));
	    		}
	    	}
	    }
    }
    return param_info;
}

// update node_ids_
void UpdateMetas(int scaling_server_id){
	  CHECK(scaling_server_id);
	  LOG(INFO) << "Updating Postoffice::node_ids_ information";
	  std::cerr<<"Node_id: "<<ps::MyID()<<" num_servers before update: "<<
		ps::NumServers()<<" scaling_server_id"<< scaling_server_id<<std::endl;
	  if (scaling_server_id > 0){
		  ps::Postoffice::Get()->UpdateNodeIDs(scaling_server_id, true);
	  } else {
		  ps::Postoffice::Get()->UpdateNodeIDs(-scaling_server_id, false);
	  }
	  std::cerr<<"num_servers after: "<<ps::NumServers()<<std::endl;
}

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    resp_scaling.sender = 0;
    LOG(INFO) << "CREATE SERVER, RESP_SCALING.SENDER "<<resp_scaling.sender;
    ps_server_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::RequestCommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandleEx, this, _1, _2, _3));
//    sync_mode_ = false;
	sync_mode_ = true;
    gradient_compression_ = std::make_shared<GradientCompression>();
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
  }

  ~KVStoreDistServer() {
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;
  };

  void RequestCommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
	LOG(INFO) << "Server " << ps::MyID() << " gets command " << recved.head;
	bool respHere = true;
    CommandType recved_type = static_cast<CommandType>(recved.head);
    if (recved_type == CommandType::kStopServer) {
      LOG(INFO) << "Server " << ps::MyID() << " is exiting";
      app->Response(recved);
      respHere = false;
      exec_.Stop();
    } else if (recved_type == CommandType::kSyncMode) {
      LOG(INFO) << "Server " << ps::MyID() << " is set to synchronous training";
      sync_mode_ = true;
      //app->Response(recved);
      //std::cerr << "Send response to command type2 " << recved.head;
    } else if (recved_type == CommandType::kSetGradientCompression) {
      gradient_compression_->DecodeParams(recved.body);
    } else if (recved_type == CommandType::kGetParaInfo) { // get parameters information, e.g., size, iteration
      GetParaInfo(recved, app);
      respHere = false;
    } else if (recved_type == CommandType::kSetParams) {
      SetParams(recved, app);
      respHere = false;
    } else {
      // this uses value 0 for message id from frontend
      // let the main thread to execute ctrl, which is necessary for python
      LOG(INFO) << "Server " << ps::MyID() << " is set optimizer";
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    }
    if (respHere) {
    	app->Response(recved);
    	std::cerr << "Send response to command type " << recved.head << std::endl;
    }
  }

  void DataHandleEx(const ps::KVMeta& req_meta,
                    const ps::KVPairs<real_t>& req_data,
                    ps::KVServer<real_t>* server) {
    DataHandleType recved_type = static_cast<DataHandleType>(req_meta.cmd);
    if (recved_type == DataHandleType::kRowSparsePushPull) {
      DataHandleRowSparse(req_meta, req_data, server);
    } else if (recved_type == DataHandleType::kCompressedPushPull) {
      DataHandleCompressed(req_meta, req_data, server);
    } else if (recved_type == DataHandleType::kMoveParams) {
      // handle parameter movement request and response here
      DataHandleMoveParams(req_meta, req_data, server);
    } else {
    	AdvancedDataHandleDefault(req_meta, req_data, server);
    }
    return;
  }

  inline void ApplyUpdates(const int key, MergeBuf *merged, NDArray *stored,
                           ps::KVServer<real_t>* server) {
	// no problem with this if condition when scaling workers
    if (merged->request.size() == (size_t) ps::NumWorkers()) {
      // let the main thread to execute updater_, which is necessary for python
      if (updater_) {
        exec_.Exec([this, key, merged, stored](){
            CHECK(updater_);
            updater_(key, merged->array, stored);
          });
      } else {
        // if no updater, just copy
        CopyFromTo(merged->array, stored);
      }
      if (log_verbose_)  {
        LOG(INFO) << "sync response to " << merged->request.size() << " workers";
      }
      for (const auto& req : merged->request) {
        server->Response(req);
      }
      merged->request.clear();

      //Waits until all previous write operations on the array are finished.
      //This method guarantees that all previous write operations
      //that pushed into the backend engine for execution are actually finished.
      stored->WaitToRead();
      // add a counter here to calculate the iteration for sync updates
      // e.g., counter[key] += 1
      steps_[key] += 1;
      SpeedoMeter(steps_[key]);
      if (steps_[key] == start_scaling_step){
    	  num_pull_key_[key] = 0;
      }
      //CheckIsScaling(key, server);
      std::cerr << "************************************";	
      std::cerr << "Update parameters on server " << ps::MyID() << std::endl;

    } else {
      merged->array.WaitToRead();
    }
  }

  void DecodeRowIds(const ps::SArray<ps::Key> &keys, int64_t *indices,
                    const int64_t master_key, const int64_t num_rows) {
    indices[0] = 0;
    for (int64_t i = 1; i <= num_rows; i++) {
      int key = DecodeKey(keys[i]);
      auto row_id = key - master_key;
      indices[i - 1] = row_id;
    }
  }

  void DataHandleRowSparse(const ps::KVMeta& req_meta,
                       const ps::KVPairs<real_t>& req_data,
                       ps::KVServer<real_t>* server) {
    int master_key = DecodeKey(req_data.keys[0]);
    auto num_rows = req_data.keys.size() - 1;
    auto& stored = store_[master_key];
    if (req_meta.push) {
      CHECK_GT(req_data.lens.size(), 0) << "req_data.lens cannot be empty";
      CHECK_EQ(req_data.lens[0], 0);
      real_t* data = req_data.vals.data();
      if (stored.is_none()) {
        if (log_verbose_) LOG(INFO) << "initial push: " << master_key;
        // initialization
        CHECK_GT(num_rows, 0) << "init with empty data is not supported";
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        size_t ds[] = {num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        CHECK_EQ(req_data.vals.size(), num_rows * unit_len);
        TBlob recv_blob(data, dshape, cpu::kDevMask);  // NOLINT(*)
        NDArray recved = NDArray(recv_blob, 0);
        stored = NDArray(kRowSparseStorage, dshape, Context());
        Engine::Get()->PushAsync(
          [recved, stored](RunContext ctx, Engine::CallbackOnComplete on_complete) {
            NDArray rsp = stored;
            stored.CheckAndAlloc({mshadow::Shape1(recved.shape()[0])});
            mshadow::Stream<cpu> *s = ctx.get_stream<cpu>();
            op::PopulateFullIdxRspImpl(s, &rsp);
            mshadow::Copy(rsp.data().FlatTo1D<cpu, float>(),
                          recved.data().FlatTo1D<cpu, float>(), s);
            on_complete();
          }, recved.ctx(), {recved.var()}, {stored.var()},
          FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
        stored.WaitToRead();
        server->Response(req_meta);
        return;
      }
      // synced push
      if (sync_mode_) {
        if (log_verbose_) LOG(INFO) << "sync push: " << master_key << " " << req_data.keys;
        auto& merged = merge_buf_[master_key];
        if (merged.array.is_none()) {
          merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
        }
        if (num_rows == 0) {
          // reset to zeros
          if (merged.request.size() == 0) {
            merged.array = NDArray(kRowSparseStorage, stored.shape(), Context());
          } else {
            // nothing to aggregate
          }
          merged.request.push_back(req_meta);
          ApplyUpdates(master_key, &merged,  &stored, server);
          return;
        }
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);
        // data
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob(data, dshape, cpu::kDevMask); // NOLINT(*)
        // row_sparse NDArray
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);

        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          NDArray out(kRowSparseStorage, stored.shape(), Context());
          std::vector<Engine::VarHandle> const_vars;
          const_vars.push_back(recved.var());
          const_vars.push_back(merged.array.var());
          // accumulate row_sparse gradients
          // TODO(haibin) override + operator for row_sparse NDArray
          // instead of calling BinaryComputeRspRsp directly
          using namespace mshadow;
          Engine::Get()->PushAsync(
            [recved, merged, out](RunContext ctx, Engine::CallbackOnComplete on_complete) {
              op::ElemwiseBinaryOp::ComputeEx<cpu, mshadow::op::plus>(
                {}, {}, {recved, merged.array}, {kWriteTo}, {out});
              on_complete();
            }, recved.ctx(), const_vars, {out.var()},
            FnProperty::kNormal, 0, PROFILER_MESSAGE_FUNCNAME);
          CopyFromTo(out, &merged.array, 0);
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(master_key, &merged,  &stored, server);
      } else {
        // async push
        if (log_verbose_) LOG(INFO) << "async push: " << master_key;
        if (num_rows == 0) {
          server->Response(req_meta);
          return;
        }
        auto unit_len = req_data.lens[1];
        CHECK_GT(unit_len, 0);
        // indices
        std::vector<int64_t> indices(num_rows);
        DecodeRowIds(req_data.keys, indices.data(), master_key, num_rows);
        TBlob idx_blob(indices.data(), mshadow::Shape1(num_rows), cpu::kDevMask);
        size_t ds[] = {(size_t) num_rows, (size_t) unit_len};
        TShape dshape(ds, ds + 2);
        TBlob recv_blob(data, dshape, cpu::kDevMask); // NOLINT(*)
        NDArray recved(kRowSparseStorage, stored.shape(), recv_blob, {idx_blob}, 0);
        exec_.Exec([this, master_key, &recved, &stored](){
            CHECK(updater_);
            updater_(master_key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {
      // pull
      if (log_verbose_) LOG(INFO) << "pull: " << master_key;
      ps::KVPairs<real_t> response;
      if (num_rows == 0) {
        std::vector<int> lens(req_data.keys.size(), 0);
        response.keys = req_data.keys;
        response.lens.CopyFrom(lens.begin(), lens.end());
        server->Response(req_meta, response);
        return;
      }
      CHECK(!stored.is_none()) << "init " << master_key << " first";
      auto shape = stored.shape();
      auto unit_len = shape.ProdShape(1, shape.ndim());
      const float* data = stored.data().dptr<float>();
      auto len = unit_len * num_rows;
      // concat values
      response.vals.resize(len);
      #pragma omp parallel for
      for (size_t i = 1; i <= num_rows; i++) {
        int key = DecodeKey(req_data.keys[i]);
        int64_t row_id = key - master_key;
        const auto src = data + row_id * unit_len;
        auto begin = (i - 1) * unit_len;
        auto end = i * unit_len;
        response.vals.segment(begin, end).CopyFrom(src, unit_len);
      }
      // setup response
      response.keys = req_data.keys;
      std::vector<int> lens(req_data.keys.size(), unit_len);
      lens[0] = 0;
      response.lens.CopyFrom(lens.begin(), lens.end());
      server->Response(req_meta, response);
    }
  }

  void DefaultStorageResponse(int key, const NDArray& stored,
                              const ps::KVMeta& req_meta,
                              const ps::KVPairs<real_t> &req_data,
                              ps::KVServer<real_t>* server) {
	// worker 0: push, pull then start training
	// other worker: pull then start training
	// vers on servers: not count the first from worker 0
    ps::KVPairs<real_t> response;
    CHECK(!stored.is_none()) << "init " << key << " first";
    auto len = stored.shape().Size();
    response.keys = req_data.keys;
    response.lens = {len};
    // TODO(mli) try to remove this CopyFrom
    response.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
    auto step = steps_[key];
    response.vers = {step};
    server->Response(req_meta, response);

    //LOG(INFO) << "Server " << ps::MyID() << " sends back parameters with key " << key << " vers: " << step;

    if (step == start_scaling_step){
    	num_pull_key_[key] ++;
    	if (num_pull_key_[key] == ps::NumWorkers()){
    		end_pulls_++;
    		if (end_pulls_ == store_.size()) { // pull for all keys is finished
    			num_pull_key_.clear();
    			end_pulls_ = 0;
    			MoveParams(server);
    		}
    	}
    }
  }

  void DataHandleCompressed(const ps::KVMeta& req_meta,
                            const ps::KVPairs<real_t> &req_data,
                            ps::KVServer<real_t>* server) {
    if (req_meta.push) {
      // there used several WaitToRead, this is because \a recved's memory
      // could be deallocated when this function returns. so we need to make sure
      // the operators with \a NDArray are actually finished

      // first for dummy key which represents original size of array, whose len is 0
      CHECK_EQ(req_data.keys.size(), (size_t)2);
      CHECK_EQ(req_data.lens.size(), (size_t)2);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[1]);

      int original_size = DecodeKey(req_data.keys[0]);
      int key = DecodeKey(req_data.keys[1]);
      auto& stored = store_[key];

      size_t ds[] = {(size_t)req_data.lens[1]};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*) req_data.vals.data(), // NOLINT(*)
                      dshape, cpu::kDevMask);
      NDArray recved = NDArray(recv_blob, 0);

      NDArray decomp_buf = decomp_buf_[key];
      dshape = TShape{(int64_t) original_size};

      if (decomp_buf.is_none()) {
        decomp_buf = NDArray(dshape, Context());
      }

      if (stored.is_none()) {
        stored = NDArray(dshape, Context());
        gradient_compression_->Dequantize(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
        }
        if (merged.request.size() == 0) {
          gradient_compression_->Dequantize(recved, &merged.array, 0);
        } else {
          gradient_compression_->Dequantize(recved, &decomp_buf, 0);
          merged.array += decomp_buf;
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(key, &merged, &stored, server);
      } else {
        // async push
        gradient_compression_->Dequantize(recved, &decomp_buf, 0);
        exec_.Exec([this, key, &decomp_buf, &stored]() {
          CHECK(updater_);
          updater_(key, decomp_buf, &stored);
        });
        server->Response(req_meta);
        stored.WaitToRead();
      }
    } else {       // pull
      CHECK_EQ(req_data.keys.size(), (size_t)1);
      CHECK_EQ(req_data.lens.size(), (size_t)0);
      int key = DecodeKey(req_data.keys[0]);
      DefaultStorageResponse(key, store_[key], req_meta, req_data, server);
    }
  }

  // can only handle request with 1 key
  void DataHandleDefault(const ps::KVMeta& req_meta,
                         const ps::KVPairs<real_t> &req_data,
                         ps::KVServer<real_t>* server) {
    CHECK_EQ(req_meta.cmd, static_cast<int>(DataHandleType::kDefaultPushPull));
    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1) << "ERROR: Node " << ps::MyID()
    << " has keys " << req_data.keys[0] << " with len " << req_data.lens[0] <<
	",  " << req_data.keys[1] << " with len " << req_data.lens[1] << " total value size: " << req_data.vals.size();
    if (req_meta.push) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    int key = DecodeKey(req_data.keys[0]);
    auto& stored = store_[key];

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      LOG(INFO) << "Server " << ps::MyID() << " received gradients with key " << key;

      size_t ds[] = {(size_t)req_data.lens[0]};
      TShape dshape(ds, ds + 1);
      TBlob recv_blob((real_t*)req_data.vals.data(), // NOLINT(*)
                      dshape, cpu::kDevMask);	//cpu
      NDArray recved = NDArray(recv_blob, 0); // create NDArray that shares data with TBlob, 0 is dev id
      if (stored.is_none()) {
        // initialization
        stored = NDArray(dshape, Context());
        CopyFromTo(recved, &stored, 0);
        server->Response(req_meta);
        stored.WaitToRead();
      } else if (sync_mode_) {
        // synced push
        auto& merged = merge_buf_[key];
        if (merged.array.is_none()) {
          merged.array = NDArray(dshape, Context());
        }
        if (merged.request.size() == 0) {
          CopyFromTo(recved, &merged.array, 0);
        } else {
          merged.array += recved;
        }
        merged.request.push_back(req_meta);
        ApplyUpdates(key, &merged, &stored, server);
      } else {
        // async push
        exec_.Exec([this, key, &recved, &stored](){
            CHECK(updater_);
            updater_(key, recved, &stored);
          });
        server->Response(req_meta);
        stored.WaitToRead();
        // add a counter here to calculate the iteration for async updates
        // e.g., counter[key] += 1
      }
    } else {
      DefaultStorageResponse(key, stored, req_meta, req_data, server);
    }
  }

  void PushHandle(const ps::KVMeta& req_meta,
		  const std::vector<ps::KVPairs<real_t>>& kvpairs,
		  ps::KVServer<real_t>* server){
	// merge/update
	std::vector<int> decoded_keys;
	bool apply_updates = false;
	bool no_reply = false;

	for(const ps::KVPairs<real_t> &kvpair : kvpairs){
		auto steps = steps_[kvpair.keys[0]];
		int key = 0;
		key = DecodeKey(kvpair.keys[0]);
		decoded_keys.push_back(key);
		auto& stored = store_[key];
		size_t ds[] = {(size_t)kvpair.lens[0]};
		TShape dshape(ds, ds + 1);
		TBlob recv_blob((real_t*)kvpair.vals.data(), dshape, cpu::kDevMask);	//cpu
		// create NDArray that shares data with TBlob, 0 is dev id
		NDArray recved = NDArray(recv_blob, 0);
		if (stored.is_none()) {
		  // initialization
		  stored = NDArray(dshape, Context());
		  CopyFromTo(recved, &stored, 0);
		} else if (sync_mode_) {
		  // synced push
		  auto& merged = merge_buf_[key];
		  if (merged.array.is_none()) {
			merged.array = NDArray(dshape, Context());
		  }
		  if (merged.request.size() == 0) {
			if(recved.shape()!=merged.array.shape()){
				std::cerr<<"MERGE ERROR"<<ps::MyID()
				<<" key:"<<key<<" recved "<<recved.shape()
				<<" kv.lens:"<<kvpair.lens[0]
				<<" merged"<<merged.array.shape()<<std::endl;
			}

			CopyFromTo(recved, &merged.array, 0);
		  } else {
			merged.array += recved;
		  }
		  merged.request.push_back(req_meta); // add for each key
		  if (merged.request.size() == (size_t) ps::NumWorkers()) {
				if (updater_) {
				  exec_.Exec([this, key, &merged, &stored](){
					  CHECK(updater_); updater_(key, merged.array, &stored);});
				} else {
				  // if no updater, just copy
				  CopyFromTo(merged.array, stored);
				}
				apply_updates = true;
		  } else {
			  no_reply = true;
		  }
		} else {
		  // async push
		  exec_.Exec([this, key, &recved, &stored](){
			  CHECK(updater_); updater_(key, recved, &stored);
			});
		}
	}

	// wait to read and send response
	if (apply_updates && !no_reply){ // apply gradients
		bool first_time = true;
		for(const int key : decoded_keys){
			auto& merged = merge_buf_[key];
			if(first_time){ // reply once
				for (const auto& req : merged.request) {
					 server->Response(req);
				}
				first_time = false;
			}
			merged.request.clear(); // clear for each key
			store_[key].WaitToRead();
			steps_[key] += 1;
			SpeedoMeter(steps_[key]);
			if (steps_[key] == start_scaling_step){
				num_pull_key_[key] = 0;
			}
		}
	} else if (!apply_updates && !no_reply){  // initialization or async
		server->Response(req_meta);
		for(const int key: decoded_keys){
			store_[key].WaitToRead();
		}
	} else if (!apply_updates && no_reply){  // merge with no update
		for(const int key: decoded_keys){
			merge_buf_[key].array.WaitToRead();
		}
	}
  }

  void PullHandle(const ps::KVMeta& req_meta,
		  const std::vector<ps::KVPairs<real_t>>& kvpairs,
		  ps::KVServer<real_t>* server){
	std::vector<int> decoded_keys;
	ps::KVPairs<real_t> response;

	int tot_size = 0; // count total value size
	for(const ps::KVPairs<real_t> &kvpair : kvpairs){
	  response.keys.push_back(kvpair.keys[0]);
	  int key = DecodeKey(kvpair.keys[0]);
	  decoded_keys.push_back(key);
	  auto len = store_[key].shape().Size();
	  tot_size += len;
	}

	ps::SArray<real_t> vals(tot_size);
	response.vals = vals;
	real_t* p_vals = response.vals.data();
	for(const int key: decoded_keys){ // copy data for response
	  auto& stored = store_[key];
	  CHECK(!stored.is_none()) << "init " << key << " first";
	  auto len = stored.shape().Size();
	  memcpy(p_vals, static_cast<const float*>(stored.data().dptr_), len*sizeof(real_t));
	  p_vals += len;
	  response.lens.push_back(len);
//	  std::cerr<<"Server "<<ps::MyID()<<" receive pull for key "<<key<<" size "<<len<<std::endl;
	  if(key==-1){
		time_t now = time(0);
		char* dt = ctime(&now);
		std::cerr<<"Assume occuring straggler, sleeping 1s, time:"<<dt<<std::endl;
		sleep(1);		
	  }
	  auto step = steps_[key];
	  response.vers.push_back(step);
	  if(false&&start_scaling_step>0){
		std::cerr<<"Receiving Pull:";
                  std::cerr<<"Server "<<ps::MyID()<<" step["<<key<<"] "<<step<<
                        " start_scaling_step "<<start_scaling_step<<" num_pull_key_ "
                        << num_pull_key_[key]<<" from "<<req_meta.sender<<std::endl;		
	  }
	  if (step == start_scaling_step){ // move parameter when scaling
		  num_pull_key_[key] ++;
//		  std::cerr<<"Server "<<ps::MyID()<<" step["<<key<<"] "<<step<<
//			" start_scaling_step "<<start_scaling_step<<" num_pull_key_ "
//			<< num_pull_key_[key]<<" from "<<req_meta.sender<<std::endl;
		  if (num_pull_key_[key] == ps::NumWorkers()){
			  end_pulls_++;
//			  std::cerr<<" end_pull:"<<end_pulls_<<" pull_size:"<<pull_size<<std::endl;
			  if (end_pulls_ == pull_size) { // pull for all keys is finished
				  num_pull_key_.clear();
				  end_pulls_ = 0;
//				  std::cerr<<"endl_pulls_["<<end_pulls_<<"] = store_.size[" <<store_.size()<<"]"<<std::endl;
				  finish_pull = true;				
				  MoveParams(server);
			  }
		  }
	  }
	}
	if(is_init == false){
	  int total_size = 0;
	  for (const auto& elem: store_){
		  int key = elem.first; // the key after decoding
		  int size = elem.second.shape().Size();
		  total_size += size;
	  }
	  if(total_size>overal_size){
		overal_size = total_size;
	  } else if (total_size == overal_size){
	 	is_init=true;	
		std::ofstream myfile;
        	char buf[80];
         	getcwd(buf,80);
         	 //std::cerr<<"Buf of getcwd is:"<<buf<<std::endl;
        	std::string wd;
          	wd = std::string(buf) + std::string("/results/");
		std::string file_name = wd+std::string("keyTimeRecorder_server")+".csv";
	  	myfile.open(file_name,std::ios_base::app);
	  	myfile<<"Server,"<<ps::MyID()<<", total_size,"<<total_size<<std::endl;
	  	myfile.close();
	 }
	}
	server->Response(req_meta, response);
  }

  // handle push/pull with multiple keys
  void AdvancedDataHandleDefault(const ps::KVMeta& req_meta,
                         const ps::KVPairs<real_t> &req_data,
                         ps::KVServer<real_t>* server) {
	CHECK_EQ(req_meta.cmd, static_cast<int>(DataHandleType::kDefaultPushPull));
	// do some check
	CHECK_GE(req_data.keys.size(), (size_t)1); // handle multiple keys
	if (req_meta.push) CHECK_EQ(req_data.lens.size(), req_data.keys.size());

	// slice the KVPairs into multiple one
	std::vector<ps::KVPairs<real_t>> kvpairs;
	int begin = 0;
	for(size_t i=0; i<req_data.keys.size(); i++){
		//LOG(INFO) << "server: " << ps::MyID() << " is_push: " << req_meta.push 
		//<< " key: " << req_data.keys[i]<<" sender "<<req_meta.sender;
		sender_temp = req_meta.sender;
		int key = DecodeKey(req_data.keys[i]);
		sender_temp = 0;	
		ps::KVPairs<real_t> kvpair;
		kvpair.keys.push_back(req_data.keys[i]);
		if (req_meta.push)
		{
			kvpair.lens.push_back(req_data.lens[i]);
			kvpair.vals = req_data.vals.segment(begin, begin+req_data.lens[i]);
			begin += req_data.lens[i];
		}
		kvpairs.push_back(kvpair);
	}

	if (req_meta.push) {
		PushHandle(req_meta, kvpairs, server);
	} else {
		PullHandle(req_meta, kvpairs, server);
	}
  }


  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MapServerIDToRank(ps::MyID())];
    if(key-kr.begin()>100000||key-kr.begin()<0){
	std::cerr<<"Decode ERROR origin: "<<key<<" kr: "<<
		kr.begin()<<" after: "<<key-kr.begin()
		<<" by "<<ps::MyID()<<" num_servers: "<<ps::NumServers()
		<<" from "<<sender_temp<<std::endl;
	return DecodeKey_Scaling(key);
    } 
    return key - kr.begin();
  }
  int DecodeKey_Scaling(ps::Key key){
    int add = 0;
    if(previous_scaling_server>0){
	add=-1;
    }else{add=1;}
    auto kr_scaling = ps::Postoffice::Get()->GetServerKeyRanges_Scaling(add);
    auto key_temp = key;
    for(int i = 0; i< ps::NumServers()+1; i++){
	auto id = 8 + i*2; 
	auto rank = ps::MapServerIDToRank(id); 
        auto temp = kr_scaling[i];
        key_temp = key - temp.begin();
	if(key_temp>=0 &&key_temp<=100000){break;}
    }
    if(key_temp>=0 && key_temp<=100000){
    	return key_temp;
    }else{
	LOG(ERROR)<<"DECODE FAIL! "<<add;
	return 0;
    }
  }
  // send parameter information on this server to the scheduler
  void GetParaInfo(const ps::SimpleData& recved, ps::SimpleApp* app){
	  // response message format: "speed step key:size key:size key:size"
	  // improve this code using self-defined data structure and serialization
	  CHECK(!recved.body.size()) << "Request body of kGetParaInfo should be empty";
	  if (!store_.size()){
		  app->Response(recved); // no parameters on this server
		  LOG(INFO) << "No parameter info on Server " << ps::MyID();
		  return;
	  }
	  ParamInfo param_info;
	  param_info.speed = update_speed_;
	  param_info.step = last_step_;
	  int server_id = ps::MyID();
	  for (const auto& elem: store_){
		  int key = elem.first; // the key after decoding
		  int size = elem.second.shape().Size();
		  param_info.ps_kvs[server_id].push_back(std::make_pair(key,size));
	  }
	  std::string body = ParamInfoToString(param_info);
	  LOG(INFO) << "Parameter info on Server " << ps::MyID() << ": " << body;
	  app->Response(recved, body);
	  // app->Response(recved) in the caller function
  }

  // update parameters on this server based on the assignment from server
  void SetParams(const ps::SimpleData& recved, ps::SimpleApp* app){
	  LOG(INFO) << "Server " << ps::MyID() << " received new parameter assignment.";
	  ParamInfo param_info = StringToParamInfo(recved.body);
	  // the new server will not get the new assignment
	  // at the specified step, each server sends parameters to others and acknowledges params from others
	  // then response to the scheduler
	  start_scaling_step = param_info.step;
	  end_scaling_step = start_scaling_step;
	  scaling_server_id_ = int(param_info.speed);
	  previous_scaling_server = scaling_server_id_;
	  pull_size = store_.size();
	  // no need mutex lock since DataHandleEx and RequestCommandHandle is called by same thread
	  move_out_key_dests.clear();
	  move_out_keys.clear();
	  move_in_key_dests.clear();
	  // figure out the keys sent to others
	  for(const auto server_kvs : param_info.ps_kvs){
		  int server_id = server_kvs.first;
		  for (const auto kv_pair : server_kvs.second) {
			  int key = kv_pair.first;
			  int size = kv_pair.second;
			  if(store_.count(key)){ // the key is on this server
				  if (server_id != ps::MyID()) {
					  // need to sent out the key to other servers
					  std::cerr<<"Server "<<ps::MyID()<<" send key "<<key<<" to server "<<server_id<<" sotre_.count "<<store_.count(key)<<std::endl;
					  move_out_key_dests[key] = server_id;
					  move_out_keys.insert(key);
				  }
			  } else if (server_id == ps::MyID()) {
				std::cerr<<"Server "<<server_id<<" Move In Key Assignment "<< key <<std::endl;
				move_in_key_dests[key] = server_id;
			  }
		  }
	  }
	  // save recved to resp_scaling
	  resp_scaling.head = recved.head;
	  resp_scaling.sender = recved.sender;
	  resp_scaling.timestamp = recved.timestamp;
	  resp_scaling.body = "";
  }

  // send the key to other servers if necessary
  void MoveParams(ps::KVServer<real_t>* server){
      struct timeval tv;
      gettimeofday(&tv,NULL);
      time_t interval = 1000000*tv.tv_sec+tv.tv_usec;
      std::ofstream myfile;
      myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
      myfile<<"stage 3, start: "<<interval<<", ";
      myfile.close();
	  
	  // first update node_ids_ to enable communication between servers
	  LOG(INFO) << "Server " << ps::MyID() << " starts moving parameters out...";
	  if(scaling_server_id_!=-2){
	    std::cerr<<"Call for UpdateMetas in Server:"<<ps::MyID()<<std::endl;
	    UpdateMetas(scaling_server_id_);
	  }
	  for (auto& key_dest : move_out_key_dests){
		  int key = key_dest.first;
		  int dest = key_dest.second;
		  ps::KVPairs<real_t> kvs;
		  auto& stored = store_[key];
		  auto len = stored.shape().Size();
		  kvs.keys.push_back(key);
		  kvs.lens.push_back(len);
		  kvs.vers.push_back(steps_[key]);
		  kvs.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
		  LOG(INFO) << "Server " << ps::MyID() << " is moving the parameters with key "
				  << key << " to Server " << dest << "steps_"<<steps_[key];
		  server->Send(static_cast<int>(DataHandleType::kMoveParams), kvs, dest);
		  // delete from store_
		  store_.erase(key);
		  if (sync_mode_) merge_buf_.erase(key);
	  }
	  FinishParamMove();
  }

  // handle parameter movement request and response
  void DataHandleMoveParams(const ps::KVMeta& req_meta,
          const ps::KVPairs<real_t> &req_data,
          ps::KVServer<real_t>* server) {
	  // check if request or response
	  if (req_meta.request){
		  // push parameter here
		    CHECK_EQ(req_data.keys.size(), (size_t)1);
		    CHECK_EQ(req_data.lens.size(), (size_t)1);
		    CHECK_EQ(req_data.vers.size(), (size_t)1);
		    CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
		    int key = req_data.keys[0];
		    int ver = req_data.vers[0];
		    LOG(INFO) << "Server " << ps::MyID() << " received parameters with key " << key << " vers: " << ver;
		    move_in_key_dests.erase(key);
		    auto& stored = store_[key];
		    // copy parameters to store_
		    size_t ds[] = {(size_t)req_data.lens[0]};
		    TShape dshape(ds, ds + 1);
		    TBlob recv_blob((real_t*)req_data.vals.data(), // NOLINT(*)
		                    dshape, cpu::kDevMask);	//cpu
		    NDArray recved = NDArray(recv_blob, 0); // create NDArray that shares data with TBlob, 0 is dev id
		    CHECK(stored.is_none()) << "Parameters with key " << key
		    		<< " are already existed on server " << ps::MyID();
		    stored = NDArray(dshape, Context());
		    CopyFromTo(recved, &stored, 0);
		    stored.WaitToRead();
		    steps_[key] = ver;

		    // send response back
		    ps::KVPairs<real_t> kvs;
		    kvs.keys.push_back(key);
		    server->Response(req_meta, kvs);
	  } else {
		  // get a response showing the parameter finished movement
		  	CHECK_EQ(req_data.keys.size(), (size_t)1);
		    int key = req_data.keys[0];
		    LOG(INFO) << "Server " << ps::MyID() << " finished parameter move for key " << key;
		    CHECK_EQ(req_meta.sender, move_out_key_dests[key]);
		    move_out_key_dests.erase(key);
	  }
	  FinishParamMove();
  }

  void FinishParamMove(){
	  std::cerr << "In FinishParamMove function, server "<< ps::MyID()
		<< " resp_scaling: sender "<<resp_scaling.sender<<" head "<<
		resp_scaling.head<<" out_size "<<move_out_key_dests.size()
		<<" in_size "<<move_in_key_dests.size()<<std::endl;
	//Check param movement
	  ParamInfo param_info;
          param_info.speed = update_speed_;
          param_info.step = last_step_;
          int server_id = ps::MyID();
          for (const auto& elem: store_){
                  int key = elem.first; // the key after decoding
                  int size = elem.second.shape().Size();
                  param_info.ps_kvs[server_id].push_back(std::make_pair(key,size));
          }
          std::string body = ParamInfoToString(param_info);
          //LOG(INFO) << "Parameter info on Server " << ps::MyID() << ": " << body;

	  // for the new node, do not send ack.
	  if (finish_pull&&(resp_scaling.sender > 0) && !(move_out_key_dests.size()) && !(move_in_key_dests.size())){
		  LOG(INFO) << "Server " << ps::MyID()<< " finished all parameter move";
	  	  LOG(INFO) << "resp_scaling: sender-"<<resp_scaling.sender
			<< " head-"<<resp_scaling.head;
		  // use call back here for further optimization
		  static_cast<ps::SimpleApp*>(ps_server_)->Response(resp_scaling);
		  LOG(INFO) << "Server " << ps::MyID() << " sent acknowledgment to the scheduler";
		  finish_pull = false;
		  resp_scaling.sender = 0;
		  start_scaling_step = -1;
		  scaling_server_id_ = 0;

		  // write finish
	      const char* workdir = std::getenv("WORK_DIR");
	      if (workdir == NULL){
	        	LOG(ERROR) << "Environment variable WORK_DIR is not set.";
	      } else {
			  std::string fn = std::string(workdir)+"SCALING.txt";
			  std::ofstream file;
			  file.open(fn);
			  file << "FINISH\n";
			  file.close();
	    	}

		  // the first worker sends stop command to this server
		  //if (-scaling_server_id_ == ps::MyID()){
		  //	  exec_.Stop();
		  //  exit(0);
			  // how to stop postoffice: scheduler send a barrier done?
		  //}
	  }

          int total_size = 0;
          for (const auto& elem: store_){
                  int key = elem.first; // the key after decoding
                  int size = elem.second.shape().Size();
                  total_size += size;
          }	
		std::ofstream myfile;
                char buf[80];
                getcwd(buf,80);
                 //std::cerr<<"Buf of getcwd is:"<<buf<<std::endl;
                std::string wd;
                wd = std::string(buf) + std::string("/results/");
                std::string file_name = wd+std::string("keyTimeRecorder_server")+".csv";
                myfile.open(file_name,std::ios_base::app);
                myfile<<"Server,"<<ps::MyID()<<", total_size,"<<total_size<<std::endl;
                myfile.close();

  }

  /*
   * measure parameter update speed
   */
  void SpeedoMeter(int step){
	  CHECK_GT(step, 0);
	  if(step > last_step_){ // first key entering new iteration
		  last_step_ = step;
		  if (step == 1) {  // first time calling SpeedoMeter
			  start_time = std::chrono::system_clock::now();
		  }
	      if (step % disp_freq_ == 0){
	    	  auto end_time = std::chrono::system_clock::now();
	    	  std::chrono::duration<double> diff = end_time - start_time;
	    	  update_speed_ = disp_freq_ / (diff.count());
			  start_time = std::chrono::system_clock::now();
	    	  LOG(INFO) << "Server: " << ps::MyID() << " Speed: " << update_speed_ << " batches/sec";
	      }
	  }
  }

  /*
   * Temp server key range when just finishing DEC_SERVER 
   */


  /**
   * variables for speedometer
   */
  const int disp_freq_ = 10;  // measure speed every 5 iteration
  int last_step_ = 0;
  double update_speed_;
  // start scaling at this iteration
  int start_scaling_step = -1;
  int end_scaling_step = -1;
  std::chrono::time_point<std::chrono::system_clock> start_time;
  //key:dest_server
  std::unordered_map<int,int> move_out_key_dests;
  std::unordered_map<int,int> move_in_key_dests;
  std::set<int> move_out_keys;
  ps::SimpleData resp_scaling; // no effective if sender==0
  //resp_scaling.sender = 0;
  int scaling_server_id_;
  int previous_scaling_server;
  // counter for the pull times for each key
  std::unordered_map<int, int> num_pull_key_;
  size_t end_pulls_ = 0;
  bool finish_pull = false;
  int pull_size = 0;
  int sender_temp = 0;
  bool is_init = false;
  int overal_size = 0;
  /**
   * \brief user defined mode for push
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  /**
   * \brief store_ contains the value at kvstore for each key
   */
  std::unordered_map<int, NDArray> store_;

  /**
   * \brief merge_buf_ is a buffer used if sync_mode is true. It represents
   * values from different workers being merged. The store will be updated
   * to this value when values from all workers are pushed into this buffer.
   */
  std::unordered_map<int, MergeBuf> merge_buf_;

  /**
   * \brief decomp_buf_ is a buffer into which compressed values are
   * decompressed before merging to the store. used when compress_!='none'
   */
  std::unordered_map<int, NDArray> decomp_buf_;

  Executor exec_;
  ps::KVServer<float>* ps_server_;

  // whether to LOG verbose information
  bool log_verbose_;

  /**
   * \brief gradient compression object.
   * starts with none, used after SetGradientCompression sets the type
   * currently there is no support for unsetting gradient compression
   */
  std::shared_ptr<kvstore::GradientCompression> gradient_compression_;

  /**
   * track the update times (i.e., steps) of each key, this counter also migrates with the key together
   */
  std::unordered_map<int, int> steps_;


};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
