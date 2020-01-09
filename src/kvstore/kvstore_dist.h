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

/**
 * Copyright (c) 2015 by Contributors
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef kvstore_dist.h
#define kvstore_dist.h
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include <algorithm>
#include <utility>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include "./kvstore_local.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"
#include "./kvstore_dist_scheduler.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#include <chrono>
#include <thread>
#include <sys/time.h>
#endif
int global_node_id = 0;
namespace mxnet {
namespace kvstore {

/**
 * \brief distributed kvstore
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class KVStoreDist : public KVStoreLocal {
 public:
  explicit KVStoreDist(bool use_device_comm)
      : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) { 
//      Engine::Get()->SetNodeID(ps::MyID());
//      std::cerr<<"Setting Node ID for engine: "<<Engine::Get()->GetNodeID()<<std::endl;
      using namespace std::placeholders;
      ps_worker_ = new ps::KVWorker<real_t>(0);
      // @yhpeng: set request handle to process command from scheduler
      static_cast<ps::SimpleApp*>(ps_worker_)->set_request_handle(
              std::bind(&KVStoreDist::RequestCommandHandle, this, _1, _2));

      ps::StartAsync("mxnet\0");
      if (!ps::Postoffice::Get()->is_recovery() && !ps::Postoffice::Get()->is_added_node()) {
        ps::Postoffice::Get()->Barrier(
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      } // if INC_WORKER, do not send barrier
    }
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 10);
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    if (IsWorkerNode()) {
      if (barrier_before_exit_) {
        Barrier();
        if (ps::MapWorkerIDToRank(ps::MyID()) == 0) { //since the first worker may be shutdown during scaling
          // stop the executor at servers
          SendCommandToServers(static_cast<int>(CommandType::kStopServer), "");
        }
      }
      ps::Finalize(barrier_before_exit_);
      delete ps_worker_;
    }
  }

  void set_updater(const Updater& updater) override {
    CHECK(updater) << "invalid updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_updater(updater);
    } else {
      updater_ = updater;
    }
  }

  void SetGradientCompression(const std::vector<std::pair<std::string, std::string> >
                              & kwargs) override {
    KVStoreLocal::SetGradientCompression(kwargs);
    if (get_rank() == 0) {
      SendCommandToServers(static_cast<int>(CommandType::kSetGradientCompression),
                           gradient_compression_->EncodeParams());
    }
  }

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }

  void SendCommandToServers(int cmd_id, const std::string& cmd_body, int server=ps::kServerGroup) override {
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, server));
    if (cmd_id == 0) { // save the optimizer
    	optimizer_str = cmd_body;
    }
  }



  int get_group_size() const override { return ps::NumWorkers(); }

  int get_rank() const override { return ps::MyRank(); }

  int get_num_dead_node(int node_id, int timeout) const override {
    int number = 0;
    auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
    const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
    std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
    for (int r : dead_nodes) {
      if (watch_set.find(r) != watch_set.end()) number++;
    }
    return number;
  }

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    // yhpeng: start server/scheduler here
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    } else {
    	CHECK(IsSchedulerNode()) << "Node " << ps::MyID() << " is not a scheduler node";
    	scheduler_ = new KVStoreDistScheduler();
    	CHECK(scheduler_) << "Failed to initialize scheduler";
    	LOG(INFO) << "Scheduler start...";
    }
    LOG(INFO) <<"---------------------A NEW NODE STARTS-----------------------------";
    ps::StartAsync("mxnet_server\0");
    if (!ps::Postoffice::Get()->is_recovery() && !ps::Postoffice::Get()->is_added_node()) {
      ps::Postoffice::Get()->Barrier(
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    } // yhpeng: do not send Barrier if INC_SERVER
    // new server needs to set optimizer!!!!!!
    if (server_) server_->Run();
    if(!ps::Postoffice::Get()->is_deled_node()){
    	ps::Finalize();
    } else {
    	ps::Finalize(false); // no barrier
    }
    if (server_) {
      delete server_;
    }
    if (scheduler_){
    	delete scheduler_;
    	scheduler_ = nullptr;
    }
    server_ = nullptr;
  }

 private:
  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
//    int step = 0;
  };

  struct ComprPSKV {
    PSKV push;
    PSKV pull;
  };

  /**
   * \brief cache all key partitions
   *
   * `ps_kv_` is used for pushes and pulls without gradient compression
   * `compr_ps_kv_` is used for gradient compression. It contains different
   * pskv for push and pull because sizes would be different in both cases.
   * Note: `ps_kv_[k]` for some key k may not be the same as `compr_ps_kv_[k].pull`
   * This is because sharding may cause slightly different divisions when size is
   * not perfectly divisible.
   */
  std::unordered_map<int, PSKV> ps_kv_; // only used when encoding keys
  std::unordered_map<int, ComprPSKV> compr_ps_kv_;

  /**
   * \brief serialize access to ps_kv_ or push_ps_kv_/pull_ps_kv_ while encoding keys
   */
  std::mutex mu_;

  // @yhpeng: process command message from scheduler
  void RequestCommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
	  LOG(INFO) << "Worker " << ps::MyID() << " gets command " << recved.head;
      CommandType recved_type = static_cast<CommandType>(recved.head);
      if (recved_type == CommandType::kSetParams) {
        SetParams(recved, app);
      } else if (recved_type == CommandType::kGetStragInfo){
	LOG(INFO)<<" Receiving request for strag info!";
	GetStragInfo(recved,app);
      }	else if (recved_type == CommandType::kStartTraining) {
    	SetTraining(recved, app);
      } else if (recved_type == CommandType::kRequestParaInfo){
	LOG(INFO) << "Received kRequestParaInfo! " << recved.body<<recved.body.size();
	if(recved.body.size()>1){
		SetParams_Scaling(recved, app);
	}
	initialed = true;
      } else {
    	  std::cerr << "Invalid command to workers: " << recved.head;
      }
    }

std::string KeyMapInfoToString(){
	std::string body;
	for(auto it=key_map_.cbegin();it!=key_map_.cend();++it){
		body.append(std::to_string(it->first)).append(",");
		body.append(std::to_string(it->second)).append("_");
	}
	return body;
}

ParamInfo StringToParamInfo_Scaling(const std::string& body){
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
}

  void SetParams_Scaling(const ps::SimpleData& recved, ps::SimpleApp* app){
          LOG(INFO) << "Worker " << ps::MyID() << " received new parameter assignment";
          ParamInfo param_info = StringToParamInfo_Scaling(recved.body);
          start_scaling_step_ = param_info.step;
          end_scaling_step_ = start_scaling_step_;
          scaling_server_id_ = int(param_info.speed);
          scal_ps_kvs_ = std::move(param_info.ps_kvs);
          LOG(INFO) << "scaling_server_id_ " << scaling_server_id_;
  }


  void SetParams(const ps::SimpleData& recved, ps::SimpleApp* app){
	  LOG(INFO) << "Worker " << ps::MyID() << " received new parameter assignment";
	  ParamInfo param_info = StringToParamInfo(recved.body);
	  start_scaling_step_ = param_info.step;
	  end_scaling_step_ = start_scaling_step_;
	  scaling_server_id_ = int(param_info.speed);
	  scal_ps_kvs_ = std::move(param_info.ps_kvs);
	  LOG(INFO) << "scaling_server_id_ " << scaling_server_id_;
  }
  std::string ConvertStragInfo(){
	StragDetect();
	return "NUll straggler information, which is to be done.";
  }
  void GetStragInfo(const ps::SimpleData& recved, ps::SimpleApp* app){
	//TODO
	std::string body;
	body = ConvertStragInfo();
	LOG(INFO) << "Straggler info on worker " << ps::MyID() << ":" << body;
	app->Response(recved, body);	
  }

  int end_scaling_step_ = -1;
  int start_scaling_step_ = -1;
  int scaling_server_id_ = 0;
  std::unordered_map<int, std::vector<std::pair<int, int>>> scal_ps_kvs_;
  std::string optimizer_str;

  void UpdatePSKVs(){
	  // run the following code after the specified iteration
//	  LOG(INFO)<<"CHECK scaling server id:"<<scaling_server_id_;
	  CHECK(scaling_server_id_);
//	  LOG(INFO)<<"pass check scaling_server_id_";
//	  if(scaling_server_id_!=-2){
	    std::cerr<<"Call UpdateMetas for worker:"<<ps::MyID()<<std::endl;
	    UpdateMetas(scaling_server_id_);
//	  }
	  mu_.lock();
//	  LOG(INFO)<<" FINISH LOCK, prepare clear existing ps_kv_, size:"<<ps_kv_.size();
	  
	  if(!ps_kv_.empty())ps_kv_.clear();
//	  LOG(INFO)<<"Begin sorting pskvs.";
	  // should be new krs since num_server is changed by UpdateMeatas()
	  auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
	  using Elem = std::pair<int, std::vector<std::pair<int, int>>>;
	  std::vector<Elem> sorted_ps_kvs(scal_ps_kvs_.begin(), scal_ps_kvs_.end());
	  std::sort(sorted_ps_kvs.begin(), sorted_ps_kvs.end(), [](const Elem& a, const Elem& b) {
	  		  	          return a.first < b.first;
	  		  	        });
	  // must sort first since the keys need to be push to pskv.keys in order due to kv_app.h DefaultSlicer()
	  for (auto& server_kvs : sorted_ps_kvs){
		using Elem1 = std::pair<int,int>;
                std::sort(server_kvs.second.begin(),server_kvs.second.end(),[](const Elem1& a, const Elem1& b){
                        return a.first < b.first;               
                        });
		
	  }
//	  LOG(INFO)<<"FINISH SORTING BEFORE UPDATE PSKVS!";
	  for (const auto& server_kvs : sorted_ps_kvs){
		  int server_id = server_kvs.first;
		  int rank = ps::MapServerIDToRank(server_id);
//		  LOG(INFO) <<"Worker "<<ps::MyID()<< " second_size"<< server_kvs.second.size()<<
//			" for Server "<<server_id;
		  //using Elem1 = std::pair<int,int>;
		  auto temp_kvs = server_kvs;
		  for(const auto& kvs : server_kvs.second) {
//			  LOG(INFO) << "key: " << kvs.first << " size: " << kvs.second
//					  << " server_id: " << server_id << " org_key: " << key_map_[kvs.first];

			  int key = kvs.first;
			  int size = kvs.second;
			  ps::Key ps_key = krs[rank].begin() + key;
//			  LOG(INFO) << "begin: " << krs[rank].begin() << " end: " << krs[rank].end();
			  CHECK_LT(ps_key, krs[rank].end());
//			  std::cerr<<"FINISH CHECK_LT";
			  int org_key = key_map_[key];
//			  std::cerr<<"FINISH get org_key:"<<org_key;
			  PSKV& pskv = ps_kv_[org_key];
//			  std::cerr<<"FINISH get pskv.keys.size:"<<pskv.keys.size();
			  pskv.keys.push_back(ps_key);
//			  std::cerr<<"FINISH push_back ps_key:"<<ps_key;
			  pskv.lens.push_back(size);
//			  std::cerr<<"FINISH push_back lens:"<<size<<std::endl;
			  pskv.size += size;
//			  LOG(INFO)<<"FINISH THIS KEY'S UPDATE, pskv.size:"<<pskv.size;
		  }
	  }
	  for (auto& key_pskv : ps_kv_){
		  int org_key = key_pskv.first;
		  for(const auto& kv : key_pskv.second.keys){
//			  LOG(INFO) << "org-key: " << org_key << " encoded keys: " << kv;
		  }
	  }
	  mu_.unlock();
  }

  ps::SimpleData server_done_msg; // no effective if sender==0
  bool server_done = false;

  void SetTraining(const ps::SimpleData& recved, ps::SimpleApp* app){
	  LOG(INFO) << "Worker " << ps::MyID() << " received kStartTraining command";
	  server_done = true;
	  server_done_msg.head = recved.head;
	  server_done_msg.sender = recved.sender;
	  server_done_msg.timestamp = recved.timestamp;
	  server_done_msg.body = "";
  }

  void SetOptimizer(){
	  if (scaling_server_id_ > 0||scaling_server_id_==-2){
		  UpdatePSKVs();
		  LOG(INFO)<<"Finish UpdatePSKVs ";
		  std::string tname = type();
		  bool sync_mode = false;
		  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
		  auto has = [tname](const std::string& pattern) {
			return tname.find(pattern) != std::string::npos;
		  };
		  if (has("_sync")) sync_mode=true;
		  if (IsWorkerNode() && ps::MapWorkerIDToRank(ps::MyID()) == 0)
		  {
			  if (sync_mode) {
				if(scaling_server_id_!=-2){
				  LOG(INFO)<<"SendCommandToServers "<<scaling_server_id_;
				  SendCommandToServers(static_cast<int>(kvstore::CommandType::kSyncMode),
						  "", scaling_server_id_);
				}
			  }
			  LOG(INFO) << "Finish setting sync ";
			  // set optimizer
			  CHECK(optimizer_str.size());
			if(scaling_server_id_!=-2){
			  SendCommandToServers(static_cast<int>(kvstore::CommandType::kController),
					  optimizer_str, scaling_server_id_);
			  LOG(INFO) << "Finish setting optimizer ";
			}
		  }
	  } else {
		  if (IsWorkerNode() && ps::MapWorkerIDToRank(ps::MyID()) == 0) {
			 // stop the executor at servers
			 SendCommandToServers(static_cast<int>(CommandType::kStopServer), "", -scaling_server_id_);
		  }

		  UpdatePSKVs();
	  }
		//record update of PSKV
                std::ofstream myfile;
                char buf[80];
                getcwd(buf,80);
                std::string wd;
                wd = std::string(buf) + std::string("/results/");
                std::string file_name = wd+std::string("keyTimeRecorder_")+std::to_string(ps::MyID())+".csv";
                myfile.open(file_name,std::ios_base::app);
		myfile<<"***Update PSKV!***";
//            myfile<<"from,"<<msg.meta.sender<<", size,"<<size<<", avg propagation time,"<<it->second.p_time/it->second.num
                myfile.close();
	  
	  // send ack to scheduler
	  CHECK_GT(server_done_msg.sender,0);
	  if(IsWorkerNode()&&ps::MapWorkerIDToRank(ps::MyID())==0){
		std::string body = KeyMapInfoToString();
		static_cast<ps::SimpleApp*>(ps_worker_)->Response(server_done_msg,body);
	  }else{
		  static_cast<ps::SimpleApp*>(ps_worker_)->Response(server_done_msg);
	  }
 	  server_done_msg.sender = 0;
	  scaling_server_id_ = 0;
	  start_scaling_step_ = -1;
  }

  void Scaling_SetOptimizer(){
          if (scaling_server_id_ > 0||scaling_server_id_==-2){
/*		  for(int i=0;i<=60;i++){
			key_map_[i]=i;
		  }
  */                UpdatePSKVs();

                  std::string tname = type();
                  bool sync_mode = false;
                  std::transform(tname.begin(), tname.end(), tname.begin(), ::tolower);
                  auto has = [tname](const std::string& pattern) {
                        return tname.find(pattern) != std::string::npos;
                  };
                  if (has("_sync")) sync_mode=true;
                  if (IsWorkerNode() && ps::MapWorkerIDToRank(ps::MyID()) == 0)
                  {
                          if (sync_mode) {
				if(scaling_server_id_!=-2)
                                  SendCommandToServers(static_cast<int>(kvstore::CommandType::kSyncMode),
                                                  "", scaling_server_id_);
                          }
                          LOG(INFO) << "Finish setting sync ";
                          // set optimizer
                          CHECK(optimizer_str.size());
			if(scaling_server_id_!=-2){
                          SendCommandToServers(static_cast<int>(kvstore::CommandType::kController),
                                          optimizer_str, scaling_server_id_);
                          LOG(INFO) << "Finish setting optimizer ";
			}
                  }
          } else {
                  if (IsWorkerNode() && ps::MapWorkerIDToRank(ps::MyID()) == 0) {
                         // stop the executor at servers
                         SendCommandToServers(static_cast<int>(CommandType::kStopServer), "", -scaling_server_id_);
                  }

                  UpdatePSKVs();
          }

          // send ack to scheduler
//          CHECK_GT(server_done_msg.sender,0);
//          static_cast<ps::SimpleApp*>(ps_worker_)->Response(server_done_msg);
          server_done_msg.sender = 0;
          scaling_server_id_ = 0;
          start_scaling_step_ = -1;
  }


  // map modified keys to the original key
  std::unordered_map<int, int> key_map_;

  /**
   * \brief convert to keys in ps
   */
  inline PSKV& EncodeDefaultKey(int key, size_t size, bool is_push) {
	// first check steps_ to see if block
//	std::cerr << "worker: " << ps::MyID() << " key: " << key << " steps_[key] " << steps_[key]
//			<< " push: " << is_push << " start_scaling_step_ " 
//			<< start_scaling_step_ << " lens:" << size << std::endl;
	int currentMax = 0;
	for (auto it = steps_.begin();it!=steps_.end();it++){
		if(it->second>currentMax){
			currentMax = it->second;
		}
	}
	if (global_node_id == 0){
	    global_node_id = ps::MyID();
	    Engine::Get()->SetNodeID(ps::MyID());
	    std::ofstream myfile;
	    myfile.open("node_id.txt",std::ios::app);
	    myfile << "Set Engine Node ID: "<<Engine::Get()->GetNodeID()<<std::endl;
	    myfile.close();
	    std::cerr<<"Set Engine Node ID: "<<Engine::Get()->GetNodeID()<<std::endl;
//	    std::string global_node="GLOBAL_NODE_ID="+global_node_id;
//	    putenv(const_cast<char*>(global_node.c_str()));
	}
	if(scaling_server_id_==-2){
//		std::cerr<<"Check blocking conditions: is_push "<<is_push<<", currentMax "
//		<<currentMax<<", scaling step "<<start_scaling_step_<<", key "<<key<<std::endl;
	}	
	if (is_push==1&&currentMax>=start_scaling_step_&&start_scaling_step_!=-1&&steps_[key]==start_scaling_step_){
		// holding
		auto tic = std::chrono::system_clock::now(); // count overhead

		CHECK(is_push) << "Must be push request when blocking";
		LOG(INFO) << "Worker " << ps::MyID() << " starts blocking push with key " << key;
		while(!server_done){ // wait for server done
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		server_done = false;
		auto tic_1 = std::chrono::system_clock::now();
//		LOG(INFO)<<"server_done set false, begin to enter into SetOptimizer.";
		SetOptimizer();
		std::cerr << "Worker " << ps::MyID() << " unblocks push with key " << key << std::endl;
		start_scaling_step_ = -1;

		auto toc = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = toc - tic;
		std::chrono::duration<double> diff1 = toc - tic_1;
	  	std::ofstream myfile;
		myfile.open("/home/net/test/overhead.txt",std::ofstream::out | std::ofstream::app);
		if(myfile.is_open()){
			myfile << " stage 4:"<<diff1.count()<<"\n";
			myfile << "num_servers: "<<ps::NumServers()<<" \n";
			myfile << "scaling overhead for worker "<<ps::MyID()<<":"<<diff.count()<<"\n";
			myfile.close();
		}
		LOG(INFO) << "SCALING OVERHEAD: " << diff.count() << " seconds";
		for(auto it=key_map_.cbegin(); it!=key_map_.cend(); ++it){
			LOG(INFO) << "key_map_[" << it->first << "]=" << it->second;
		}
	}

    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    if (!pskv.keys.empty()) {
//	std::cerr<<"Key "<<key<<" after:"<<pskv.keys[0]<<" before_size:"<<size<<" size:"<<pskv.size<<" is_push:"<<is_push;
      CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
    } else {
    	// run once for each key during parameter initialization
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (size < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        key_map_[key] = key; // @yhpeng
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(size);
        pskv.size = size;
          if(key>=50){
//                std::cerr<<"Worker "<<ps::MyID()<<" bigarray_bound_: "<<bigarray_bound_<<" Key["
//		<<key<<"]="<< key<<"send to "<<server<<" len:"
  //              << size<<std::endl;
          }

      } else {
        // partition it to all servers
        pskv.size = 0;
        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
            static_cast<size_t>(round(static_cast<double>(size)/num_servers*(i+1))) -
            static_cast<size_t>(round(static_cast<double>(size)/num_servers*i));

          // @yhpeng: after decoding, multiple servers may get the same key
          // to make it convenient for parameter migration, let them different
          // comment out ps::Key ps_key = krs[i].begin() + key;
          // usually the number of keys is less then 65536.
          // map 'key' to 'key+65536+key_map.size()'
          int new_key = key + (1<<16) + key_map_.size();
          key_map_[new_key] = key;
          ps::Key ps_key = krs[i].begin() + new_key;
          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          pskv.lens.push_back(part_size);
          pskv.size += part_size;
	  if(key>=50){
		std::cerr<<"Key["<<key<<"]="<< new_key<<"send to "<<i<<"th server, len:"
		<< part_size<<std::endl;
	  }
        }
        CHECK_EQ(static_cast<size_t>(pskv.size), size);
      }
    }
    return pskv;
  }

  void InitImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values) override {
    //std::cerr<<"In InitiImpl, Worker "<<ps::MyID()<<" rank "<<get_rank()<<" keys "<<keys.size()<<std::endl;
    if(initialed==false&&ps::Postoffice::Get()->is_added_node()){
	SendCommandToServers(static_cast<int>(kvstore::CommandType::kRequestParaInfo),"",1);
	while(!initialed){
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
	}
	if(scaling_server_id_!=0){
		LOG(INFO) << "Begin UpdatePSKV!";
		Scaling_SetOptimizer();
		start_scaling_step_=-1;
//                LOG(INFO) << "SCALING OVERHEAD: " << diff.count() << " seconds";
                for(auto it=key_map_.cbegin(); it!=key_map_.cend(); ++it){
                        LOG(INFO) << "key_map_[" << it->first << "]=" << it->second;
                }
	}else{LOG(INFO) << "Finish worker Initial scaling!";}
    }
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].storage_type(), values[i].shape(), values[i].dtype());
    }
    if (get_rank() == 0) {
     std::cerr<<"Worker "<<ps::MyID()<<" begin init push for keys "<<keys[0]<<std::endl;
      Push_(keys, values, 0, false);
      // wait until the push is finished
      for (const int key : keys) {
        comm_buf_[key].WaitToWrite();
        compr_buf_[key].WaitToWrite();
      }
    } else {
      // do nothing
    }
    std::cerr<<"is_recovery "<<ps::Postoffice::Get()->is_recovery()<<
	" is_added_node "<<ps::Postoffice::Get()->is_added_node()<<std::endl;
    if (!ps::Postoffice::Get()->is_recovery() && !ps::Postoffice::Get()->is_added_node()) {
	std::cerr<<"Worker "<<ps::MyID()<<" is barrier to wait for param init."<<std::endl;
      Barrier(); // barrier to wait for parameter initialization
    }
  }

  void PushImpl(const std::vector<int>& keys,
                const std::vector<NDArray>& values,
                int priority) override {
    Push_(keys, values, priority, true);
  }

  void PullImpl(const std::vector<int>& keys,
                const std::vector<NDArray*>& values,
                int priority) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairsPull(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      const auto storage_type = grouped_vals[i][0]->storage_type();
      CHECK_EQ(storage_type, kDefaultStorage)
               << "Expected stype of value to be kDefaultStorage";
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(grouped_vals[i][0]->shape(), pinned_ctx_,
                           true, grouped_vals[i][0]->dtype());
      }
	pull_dsteps_[key] += 1;
/*          if(key>=50||key<=10){
                std::cerr <<"Worker "<<ps::MyID()<<" Key[" << key << "] Pulling "
                <<  " steps " << steps_[key]
                <<" pull_dsteps "<<pull_dsteps_[key]<<" pull_cbsteps "<<pull_cbsteps_[key]<<
        	" start_scaling_step" <<start_scaling_step_<<std::endl;
          }
*/

      auto pull_from_servers = [this, key, recv_buf](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = recv_buf.shape().Size();
	pull_cbsteps_[key] +=1;

          if(true||key>=59||key<=1){
               // std::cerr<<"Worker "<<ps::MyID()<<" key "<<key
                //<<" pull_cbsteps_ "<<pull_cbsteps_[key]<<" steps "<<steps_[key]<<" size "<<size<<std::endl;
          }

        PSKV& pskv = (gradient_compression_->get_type() == CompressionType::kNone) ?
                      EncodeDefaultKey(key, size, false) :
                      EncodeCompressedKey(key, size, false);
#if MKL_EXPERIMENTAL == 1
        mkl_set_tblob_eager_mode(recv_buf.data());
#endif
        real_t* data = recv_buf.data().dptr<real_t>();
        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<real_t>(data, size, false);
        // issue pull
        int cmd = (gradient_compression_->get_type() != CompressionType::kNone) ?
                  static_cast<int>(DataHandleType::kCompressedPushPull) :
                  static_cast<int>(DataHandleType::kDefaultPushPull);

        CHECK_NOTNULL(ps_worker_)->ZPull(
          pskv.keys, vals, &pskv.lens, &steps_[key], cmd, [vals, cb](){ delete vals; cb(); });
      };
      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {recv_buf.var()},
          FnProperty::kNormal,
          priority,
          PROFILER_MESSAGE("KVStoreDistDefaultStoragePull"));
      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
    }

  }

  //ps::SArray<int> vers_arr;// size 1

  void PullRowSparseImpl(const std::vector<int>& keys,
                         const std::vector<std::pair<NDArray*, NDArray>>& val_rowids,
                         int priority = 0) override {
    std::vector<int> uniq_keys;
    std::vector<std::vector<std::pair<NDArray*, NDArray>>> grouped_val_rowids;
    GroupKVPairsPullRsp(keys, val_rowids, &uniq_keys, &grouped_val_rowids);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];
      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = comm_buf_[key];
      auto& grouped_val_rowid = grouped_val_rowids[i];
      const auto storage_type = grouped_val_rowid[0].first->storage_type();
      CHECK_EQ(storage_type, kRowSparseStorage)
               << "expected kRowSparseStorage, but got " << storage_type;
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(storage_type, grouped_val_rowid[0].first->shape(),
                           pinned_ctx_, true, grouped_val_rowid[0].first->dtype());
      }
      auto &target_val_rowids = grouped_val_rowids[i];
      const size_t num_vals = target_val_rowids.size();
      size_t num_rows = 0;
      // TODO(haibin) refactor this for loop
      for (size_t i = 0; i < num_vals; i++) {
        auto &row_id = target_val_rowids[i].second;
        NDArray indices(row_id.shape(), pinned_ctx_, false, mshadow::kInt64);
	std::cerr << "PullRowSparseImpl ---- SHAPE OF ROW_ID IS: " << row_id.shape() << " SHAPE OF INDICES: " << indices.shape() << std::endl;
	LOG(INFO) << "PullRowSparseImpl ---- SHAPE OF ROW_ID IS: " << row_id.shape() << " SHAPE OF INDICES: " << indices.shape();
        CopyFromTo(row_id, &indices, 0);
        Unique(&indices, priority);
        target_val_rowids[i].second = indices;
        num_rows += indices.shape().Size();
      }
      if (num_vals > 1) {
        // TODO(haibin) aggregate over all unique indices
        LOG(FATAL) << "RowSparsePull with multiple values is not implemented yet";
      } else {
        auto& indices = target_val_rowids[0].second;
        PullRowSparse_(key, recv_buf, indices, priority);
        comm_->BroadcastRowSparse(key, recv_buf, grouped_val_rowid, num_vals == 1, priority);
      }
    }
  }

  void Push_(const std::vector<int>& keys,
             const std::vector<NDArray>& values,
             int priority,
             bool do_merge) {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairsPush(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devices
      int key = uniq_keys[i];
      const auto& vals = grouped_vals[i];
      NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];

      const auto storage_type = merged.storage_type();
      auto &comm_buf = comm_buf_[key];
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
        // Start of a push doesn't guarantee that the previous pushes are completed.
        // This shouldn't affect training of networks though because training involves
        // a sequence of push, pull, then push. This imposes ordering that the
        // second push happens after the first pull, and the pull happens after first push.
        comm_buf = merged;  // avoid memory copy
      } else {
        if (comm_buf.is_none()) {
          if (storage_type == kDefaultStorage) {
            comm_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
          } else {
            comm_buf = NDArray(storage_type, merged.shape(), pinned_ctx_, true, merged.dtype());
          }
        }
        CopyFromTo(merged, &comm_buf);
      }

      // push to servers
      if (storage_type == kDefaultStorage) {
        if (gradient_compression_->get_type() == CompressionType::kNone) {

        if(key >= 0 && push_dsteps_[key]!=0){
  //              std::cerr<<"check push_callback";
//                << " global_max_steps "<<global_max_steps<<" currentMax "
//                <<currentMax<<" start_scaling_step_ "<<start_scaling_step_;
    //            std::cerr<<" push_dsteps "<<push_dsteps_[key]<<" push_call "<<push_cbsteps_[key];

//           while(true){
           while(start_scaling_step_!=-1){
                int currentMax = 0;
                for (auto it = steps_.begin();it!=steps_.end();it++){
                        if(it->second>currentMax){
                                currentMax = it->second;
                        }
                }
//		std::cerr<<"**************Check if push delay for 2 rounds**********"
//		std::cerr<<"check push_callback"
//		<< " global_max_steps "<<global_max_steps<<" currentMax "
//		<<currentMax<<" start_scaling_step_ "<<start_scaling_step_;
//		std::cerr<<" push_dsteps "<<push_dsteps_[key]<<" push_call "<<push_cbsteps_[key];
//                if(global_max_steps<currentMax||start_scaling_step_-currentMax>1){
//		if(global_max_steps<currentMax){
		if(push_dsteps_[key]==push_cbsteps_[key]&&pull_dsteps_[key]==pull_cbsteps_[key]){
/*	                std::cerr<<"check push_callback"
	                << " global_max_steps "<<global_max_steps<<" currentMax "
        	        <<currentMax<<" start_scaling_step_ "<<start_scaling_step_;
                	std::cerr<<" push_dsteps "<<push_dsteps_[key]<<" push_call "<<push_cbsteps_[key];
*/                        global_max_steps = currentMax;
                        break;
                }else{
                        std::this_thread::sleep_for(std::chrono::milliseconds(1000));             
	        }
           }
        }

          PSKV& pskv = EncodeDefaultKey(key, comm_buf.shape().Size(), true);
	 // std::cerr<<"PushDefault Test!"<<std::endl;
          PushDefault(key, comm_buf, pskv, priority);
        } else {
          // Note: gradient compression uses `do_merge` as proxy to
          // detect whether the push is initialization of a key or not.
          // is_active is false when push is initialization of key
          bool is_active = do_merge;
          PSKV &pskv = EncodeCompressedKey(key, comm_buf.shape().Size(), is_active);
          // Returns push_pskv if active, else pull_pskv
          // we want inactive gc to send uncompressed gradients,
          // but sharded in the same way as later pushes would when gc becomes active
          if (is_active) {
            PushCompressed(key, comm_buf, pskv, priority);
          } else {
            PushDefault(key, comm_buf, pskv, priority);
          }
        }
      } else if (storage_type == kRowSparseStorage) {
        CHECK(gradient_compression_->get_type() == CompressionType::kNone)
          << "Gradient compression for row sparse storage type is not supported";
        PushRowSparse(key, comm_buf, priority);
      } else {
        LOG(FATAL) << "unknown storage type";
      }
    }
  }

  void PushCompressed(int key, const NDArray& comm_buf, const PSKV& pskv, int priority) {
    auto &small_buf = compr_buf_[key];
    auto &res_buf = residual_[key];
    size_t original_size = comm_buf.shape().Size();

    // Init the small buffer and residual_ buffer for quantize
    if (small_buf.is_none()) {
      small_buf = NDArray(TShape{pskv.size}, comm_buf.ctx(), false, comm_buf.dtype());
      res_buf = NDArray(TShape{(int64_t) original_size}, comm_buf.ctx(),
                        false, comm_buf.dtype());
      res_buf = 0;
    }
    gradient_compression_->Quantize(comm_buf, &small_buf, &res_buf, priority);
    auto push_to_servers =
      [this, key, pskv, small_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
        size_t size = small_buf.shape().Size();
        real_t* data = small_buf.data().dptr<real_t>();
#if MKL_EXPERIMENTAL == 1
        mkl_set_tblob_eager_mode(small_buf.data());
#endif
        // do push. false means no delete
        ps::SArray<real_t> vals(data, size, false);
        CHECK_NOTNULL(ps_worker_)->ZPush(
          pskv.keys, vals, pskv.lens,
          static_cast<int>(DataHandleType::kCompressedPushPull), [cb]() { cb(); });
      };
    // acquire locks on both comm_buf and small_buf so that
    // pull (which uses comm_buf) for the same key waits till push finishes
    Engine::Get()->PushAsync(
      push_to_servers,
      pinned_ctx_,
      {small_buf.var(), comm_buf.var()},
      {},
      FnProperty::kNormal,
      priority,
      PROFILER_MESSAGE("KVStoreDistCompressedPush"));
  }

  void PushDefault(int key, const NDArray &send_buf, const PSKV& pskv, int priority) {
	push_dsteps_[key] += 1;
        //LOG(INFO) <<"---------------------Enter into function PushDefault-----------------------------";
	//std::cerr <<"key is:"<<key<<std::endl;
          if((key>=55||key<=5)&&start_scaling_step_>0){
	        int currentMax = 0;
        	for (auto it = steps_.begin();it!=steps_.end();it++){
                	if(it->second>currentMax){
                        	currentMax = it->second;
                	}
       		}

                std::cerr <<"Worker "<<ps::MyID()<<" Key[" << key << "] = " 
		<< pskv.keys[0] << " steps " << steps_[key] 
		<<" push_dsteps "<<push_dsteps_[key]<<" push_cbsteps "<<push_cbsteps_[key]<<" currentMax "
	<<currentMax<< " start_scaling_step" <<start_scaling_step_<<std::endl;

          }
/*
	if(key == 0){
	   while(true){
		int currentMax = 0;
                for (auto it = steps_.begin();it!=steps_.end();it++){
                        if(it->second>currentMax){
                                currentMax = it->second;
                        }
                }
		if(global_max_steps<currentMax){
			global_max_steps = currentMax;
			break;
		}else{
                        std::this_thread::sleep_for(std::chrono::milliseconds(10));		     }
	   }
	}*/
	if(ps::MyID()==11&&steps_[key]==5&&key==0){
		std::cerr<<"*****************************************"<<std::endl;
//		std::cerr<<"Worker 11 sleep!"<<std::endl;
//		std::this_thread::sleep_for(std::chrono::milliseconds(100000000000));
	}	
    auto push_to_servers =
        [this, key, pskv, send_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
	  push_cbsteps_[key]+=1;
	  if(true||key>=59||key<=1){
//		time_t now = time(0);
//		char* dt = ctime(&now);
		struct timeval tv;
		gettimeofday(&tv,NULL);
		auto current_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
	//	std::cerr<<"Worker "<<ps::MyID()<<" key "<<key
	//	<<" push_cbsteps_ "<<push_cbsteps_[key]<<" steps "<<steps_[key]<<
	//	" size "<<pskv.lens[0]<<"  time: "<<current_time<<std::endl;
	  }
          // convert to ps keys
          size_t size = send_buf.shape().Size();
          real_t* data = send_buf.data().dptr<real_t>();
#if MKL_EXPERIMENTAL == 1
          mkl_set_tblob_eager_mode(send_buf.data());
#endif
          // do push. false means no delete
          ps::SArray<real_t> vals(data, size, false);

          CHECK_NOTNULL(ps_worker_)->ZPush(
              pskv.keys, vals, pskv.lens,
              static_cast<int>(DataHandleType::kDefaultPushPull), [cb]() { cb(); });
        };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistDefaultPush"));
  }

  // push row sparse gradient
  void PushRowSparse(int key, const NDArray &send_buf, int priority) {
    using namespace rowsparse;
    auto push_to_servers = [this, key, send_buf]
                           (RunContext rctx, Engine::CallbackOnComplete cb) {
#if MKL_EXPERIMENTAL == 1
      mkl_set_tblob_eager_mode(send_buf.data());
#endif
      real_t* data = send_buf.data().dptr<real_t>();
      const int64_t num_rows = send_buf.aux_shape(kIdx)[0];
      const auto offsets = send_buf.aux_data(kIdx).dptr<int64_t>();
      const auto unit_len = send_buf.shape().ProdShape(1, send_buf.shape().ndim());
      const int64_t size = num_rows * unit_len;

       // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, send_buf.shape()[0]);
      if (this->log_verbose_) {
        LOG(INFO) << "worker " << ps::MyID() << " push lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      ps::SArray<real_t> vals(data, size, false);
      CHECK_NOTNULL(ps_worker_)->ZPush(pskv.keys, vals, pskv.lens,
                                       static_cast<int>(DataHandleType::kRowSparsePushPull),
                                       [cb]() { cb(); });
    };
    Engine::Get()->PushAsync(
        push_to_servers,
        pinned_ctx_,
        {send_buf.var()},
        {},
        FnProperty::kNormal,
        priority,
        PROFILER_MESSAGE("KVStoreDistRowSparsePush"));
  }


  // pull row sparse weight into `recv_buf` based on indices given by `indices`
  void PullRowSparse_(const int key, const NDArray& recv_buf,
                      const NDArray& indices, int priority) {
    using namespace rowsparse;
    auto pull_from_servers = [this, key, recv_buf, indices]
      (RunContext rctx, Engine::CallbackOnComplete cb) {
      // allocate memory for the buffer
      size_t num_rows = indices.shape().Size();
      recv_buf.CheckAndAlloc({mshadow::Shape1(num_rows)});
#if MKL_EXPERIMENTAL == 1
      mkl_set_tblob_eager_mode(recv_buf.data());
#endif
      real_t* data = recv_buf.data().dptr<real_t>();
      const auto offsets = indices.data().dptr<int64_t>();
      const auto unit_len = recv_buf.shape().ProdShape(1, recv_buf.shape().ndim());
      const int64_t size = num_rows * unit_len;
      // convert to ps keys in row sparse format
      PSKV& pskv = EncodeRowSparseKey(key, size, num_rows, offsets,
                                      unit_len, recv_buf.shape()[0]);
      if (this->log_verbose_) {
        LOG(INFO) << "worker " << ps::MyID() << " pull lens: " << pskv.lens << " keys: "
                  << pskv.keys << " size: " << size;
      }
      auto vals = new ps::SArray<real_t>(data, size, false);
      // copy indices to recv_buf. this needs to be done before ZPull
      // because after pull is done, the callback function returns and locks are released.
      // at this point, later functions may access the indices variable while copy happens
      mshadow::Copy(recv_buf.aux_data(kIdx).FlatTo1D<cpu, int64_t>(),
                    indices.data().FlatTo1D<cpu, int64_t>());
      CHECK_NOTNULL(ps_worker_)->ZPull(pskv.keys, vals, &pskv.lens, &steps_[key],
                                       static_cast<int>(DataHandleType::kRowSparsePushPull),
                                       [vals, cb]() { delete vals; cb(); });
    };
    CHECK_NOTNULL(Engine::Get())->PushAsync(
      pull_from_servers,
      pinned_ctx_,
      {indices.var()},
      {recv_buf.var()},
      FnProperty::kNormal,
      priority,
      PROFILER_MESSAGE("KVStoreDistRowSparsePull"));
  }

  /**
   * \brief check if the keys are all unique
   */
  void CheckUnique(const std::vector<int>& keys) {
    auto keys_copy = keys;
    auto last = std::unique(keys_copy.begin(), keys_copy.end());
    CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
             static_cast<size_t>(keys.size()));
  }

  /**
   * \brief Convert to keys in ps for compressed values
   * Divides original array into equal parts for each server
   * Populates both push and pull pskv on first call
   */
  inline PSKV& EncodeCompressedKey(int key, size_t original_size, bool is_push) {
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    int num_servers = krs.size();
    CHECK_GT(num_servers, 0);

    // represents size of data to be sent
    size_t compr_size = gradient_compression_->GetCompressedSize(original_size);

    mu_.lock();
    PSKV& pskv = (is_push) ? compr_ps_kv_[key].push : compr_ps_kv_[key].pull;
    mu_.unlock();

    if (!pskv.keys.empty()) {
      size_t size = (is_push) ? compr_size : original_size;
      CHECK_EQ(static_cast<size_t >(pskv.size), size)<< "The value size can't be changed";
    } else {
      // populate both pull and push pskvs
      // push pskv has sizes corresponding to compressed data
      // pull pskv has decompressed sizes for parts in push_pskv
      mu_.lock();
      PSKV& pull_pskv = compr_ps_kv_[key].pull;
      PSKV& push_pskv = compr_ps_kv_[key].push;
      mu_.unlock();

      if (original_size < bigarray_bound_) {
        // a simple heuristic for load balancing
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        // meta info
        push_pskv.keys.push_back(krs[server].begin() + original_size);
        push_pskv.lens.push_back(0);
        // data
        push_pskv.keys.push_back(ps_key);
        pull_pskv.keys.push_back(ps_key);
        push_pskv.lens.push_back(compr_size);
        pull_pskv.lens.push_back(original_size);
        push_pskv.size = compr_size;
        pull_pskv.size = original_size;
      } else {
        // partition it to all servers
        push_pskv.size = 0;
        pull_pskv.size = 0;

        for (int i = 0; i < num_servers; ++i) {
          size_t part_compr, part_orig;
          if (i == num_servers-1) {
            part_compr = compr_size - push_pskv.size;
            part_orig = original_size - pull_pskv.size;
          } else {
            part_compr =
              static_cast<size_t> (round(static_cast<double>(compr_size)/num_servers*(i+1))) -
              static_cast<size_t> (round(static_cast<double>(compr_size)/num_servers*(i)));
            part_orig = part_compr * gradient_compression_->GetCompressionFactor();
          }

          // meta info
          ps::Key ps_key_dummy = krs[i].begin() + part_orig;
          CHECK_LT(ps_key_dummy, krs[i].end());
          push_pskv.keys.push_back(ps_key_dummy);
          push_pskv.lens.push_back(0);

          // data
          ps::Key ps_key = krs[i].begin() + key;
          CHECK_LT(ps_key, krs[i].end());
          push_pskv.keys.push_back(ps_key);
          pull_pskv.keys.push_back(ps_key);
          // push_pskv stores lengths of compressed blocks
          push_pskv.lens.push_back(part_compr);
          // pull_pskv stores lengths of original data
          pull_pskv.lens.push_back(part_orig);
          push_pskv.size += part_compr;
          pull_pskv.size += part_orig;
        }
        CHECK_EQ(static_cast<size_t>(push_pskv.size), compr_size);
        CHECK_EQ(static_cast<size_t>(pull_pskv.size), original_size);
        CHECK_EQ(push_pskv.lens.size(), num_servers*2);
        }
      }
    return pskv;
  }

  // Note: this encoding method for row sparse keys doesn't allow cross-layer batching
  inline PSKV& EncodeRowSparseKey(const int key, const int64_t size, const int64_t num_rows,
                                  const int64_t *offsets, const size_t unit_len,
                                  const int64_t total_num_rows) {
    using namespace common;
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();
    pskv.keys.clear();
    pskv.lens.clear();
    // TODO(haibin) cache this information
    auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
    int num_servers = krs.size();
    CHECK_GT(num_servers, 0);

    if (total_num_rows * unit_len >= bigarray_bound_) {
      pskv.size = 0;
      int64_t start_row = 0;
      // parition it to all servers
      for (int i = 0; i < num_servers; ++i) {
        ps::Key master_key = krs[i].begin() + key;
        pskv.keys.push_back(master_key);
        pskv.lens.push_back(0);
        if (offsets && size > 0) {
          // calculate partition ranges
          int64_t part_num_rows =
            llround(static_cast<double>(total_num_rows) / num_servers * (i + 1)) -
            llround(static_cast<double>(total_num_rows) / num_servers * i);
          auto end_row = start_row + part_num_rows;
          // search for offsets in [start_row, end_row)
          auto lb = std::lower_bound(offsets, offsets + num_rows, start_row);
          auto ub = std::upper_bound(offsets, offsets + num_rows, end_row - 1);
          for (auto offset = lb; offset < ub; offset++) {
            ps::Key ps_key = krs[i].begin() + key + (*offset - start_row);
            CHECK_LT(ps_key, krs[i].end());
            pskv.keys.push_back(ps_key);
            pskv.lens.push_back(unit_len);
            pskv.size += unit_len;
          }
          start_row = end_row;
        }
      }
      CHECK_EQ(static_cast<size_t>(pskv.size), size);
    } else {
      // send it to a single random picked server
      int server = (key * 9973) % num_servers;
      ps::Key master_key = krs[server].begin() + key;
      pskv.keys.push_back(master_key);
      pskv.lens.push_back(0);
      for (int64_t i = 0; i < num_rows; i++) {
        ps::Key ps_key = krs[server].begin() + key + offsets[i];
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(unit_len);
      }
      pskv.size = size;
    }
    return pskv;
  }


  //@yrchen
  struct Opr_symbol{
    std::string Op;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<int> keys;
  };
  struct Opr_train{
    std::string opr_name;
    std::string opr_symbol_name;
    std::vector<int> keys;
    time_t start_time;
    time_t end_time;
    time_t runtime;
  };
  struct Key_res{
    int key;
    int server;
    time_t time;
  };
  struct Opr_Key{
    std::string opr_name;
    std::string opr_symbol_name;
    std::vector<int> keys;
    int key=-1;
    int origin_key;
    int server;
    time_t start_time;
    time_t end_time;
    time_t runtime;
  };
  /*
  bool compare(struct Opr_Key a, struct Opr_Key b){
    return a.start_time < b.start_time;
  }*/

  std::vector<Opr_train> opr_train;
  std::vector<Key_res> key_res;

  std::vector<std::string> key_extract(std::string filename){
        std::ifstream myfile;
	LOG(INFO)<<"Open key file:"<<filename;
        myfile.open(filename);
        std::vector<std::string> keys;
        if(myfile.is_open()){
                std::string line;
                while(std::getline(myfile,line)){
                        keys.push_back(line.substr(0,line.size()-1));
                }
        }
        return keys;
  }

  void merge_trace(){
    std::ifstream myfile;
    char buf[80];
    getcwd(buf,80);
    std::string wd;
    wd = std::string(buf) + std::string("/results/");
    std::string file_name = wd+std::string("key_response_")+std::to_string(ps::MyID())+".csv";  
//    myfile.open("results/key_response_9.csv");
    myfile.open(file_name);
    if(myfile.is_open()){
      std::string line;
      while(std::getline(myfile,line)){
        //std::cerr<<line<<std::endl;
        char* pattern = ",";
        size_t initpos = 0;
        size_t pos = line.find(pattern);
	std::string server_str = line.substr(0,pos);
	int server_num = std::stoi(server_str.substr(12,server_str.size()-1));
	initpos = pos + 1;
	pos = line.find(pattern,initpos);
        std::string key_str = line.substr(initpos,pos);                        
        std::string time = line.substr(pos+1,line.size()-1);
        //std::cerr<<"key string:"<<key_str.substr(4,key_str.size()-1)<<std::endl;
        int key = std::stoi(key_str.substr(4,key_str.size()-1));
        Key_res temp;
        temp.key = key;
	temp.server = server_num;
        temp.time = std::stol(time); 
        key_res.push_back(temp);
      }
    }
    myfile.close();
    std::remove(file_name.c_str());
//    std::remove("results/key_response_9.csv");
    time_t init_time = key_res[0].time;
    std::vector<Opr_Key> opr_keys;
    for(auto key:key_res){
    //	std::cerr<<"key:"<<key.key<<", start_time:"<<key.time<<std::endl;
      struct Opr_Key opr_key;
      opr_key.key = key.key;
      auto it = key_map_.find(key.key);
      if(it != key_map_.end()){
	opr_key.origin_key = it->second;
      }
      opr_key.start_time = key.time - init_time;
      opr_key.server = key.server;
      opr_keys.push_back(opr_key);
    }	
    for(auto opr:opr_train){
      struct Opr_Key opr_key;
      opr_key.opr_name = opr.opr_name;
      opr_key.opr_symbol_name = opr.opr_symbol_name;
      opr_key.keys = opr.keys;
      opr_key.start_time = opr.start_time - init_time;
      opr_key.end_time = opr.end_time - init_time;
      opr_key.runtime = opr.runtime;
      opr_keys.push_back(opr_key);
    }
    using Elem = struct Opr_Key;
    std::sort(opr_keys.begin(),opr_keys.end(),[](const Elem& a, const Elem& b){
		return a.start_time < b.start_time;
	});
    for(int i = 0; i < opr_keys.size(); i++){
      struct Opr_Key opr_key = opr_keys[i];
      if(opr_key.key!=-1){
       // std::cerr<<"key "<<opr_key.key<<", orgin_key "<<opr_key.origin_key
	//<<", from server "<<opr_key.server<<", start time:"<<opr_key.start_time<<", waiting time:";
        time_t wait_t;
        for(int j = i; j<opr_keys.size(); j++){
          struct Opr_Key opr_key_temp = opr_keys[j];
          if(opr_key_temp.key==-1){
            auto it = std::find(opr_key_temp.keys.begin(),opr_key_temp.keys.end(),opr_key.origin_key);
            if(it==opr_key_temp.keys.end()){
              //std::cerr<<"No waiting time,";
              wait_t = 0;
            }else{
             // std::cerr<<" found in "<<j<<",";
              for(int k = i; k > 0; k--){
                struct Opr_Key opr_key_pre = opr_keys[k];
                if(opr_key_pre.key==-1){
                  wait_t = opr_key.start_time - opr_key_pre.end_time;
		  auto it = wait_time.find(opr_key.key);
		  if(it != wait_time.end()){
			it->second.total_count += wait_t;
			it->second.number += 1;
		  } else{
			struct counter cnt;
			cnt.origin_key = opr_key.origin_key;
			cnt.total_count = wait_t;
			cnt.number = 1;
			wait_time.insert(std::make_pair(opr_key.key,cnt));
		  }
                  break;
                }
              }
              //break;
            }
            break;
          }
        }
       // std::cerr<<wait_t<<std::endl;	
      }else{
        //std::cerr<<opr_key.opr_name<<","<<opr_key.opr_symbol_name<<", start:"<<opr_key.start_time<<", end:"<<opr_key.end_time;
        for(auto key:opr_key.keys){
          //std::cerr<<", "<<key;
        }
       // std::cerr<<"\n";
      }
    }	
  }

  void output_wait_time(){
	std::ofstream myfile;
        char buf[80];
        getcwd(buf,80);
        std::string wd;
        wd = std::string(buf) + std::string("/results/");
        std::string file_name = wd+std::string("wait_time__")+std::to_string(ps::MyID())+".csv";
        myfile.open(file_name,std::ios_base::app);
	if(myfile.is_open()){
		myfile << "----Monitor waiting time for keys----\n";
		for(auto it=wait_time.begin(); it!=wait_time.end(); ++it){
			if(it->second.number!=0||it->second.total_count!=0){
				myfile<<"key,"<<it->first<<", origin_key,"<<it->second.origin_key
				<<", total waiting time,"
				<<it->second.total_count<<", number,"<<it->second.number
				<<", avg,"<<it->second.total_count/it->second.number<<"\n";
				it->second.number =  0; it->second.total_count = 0;
			}
		}
	}
	myfile.close();	
  }

  size_t split(const std::string &txt, std::vector<std::string> &strs, char ch)
  {
      size_t pos = txt.find( ch );
      size_t initialPos = 0;
      strs.clear();

      // Decompose statement
      while( pos != std::string::npos ) {
          strs.push_back( txt.substr( initialPos, pos - initialPos ) );
          initialPos = pos + 1;

          pos = txt.find( ch, initialPos );
      }

      // Add the last one
      strs.push_back( txt.substr( initialPos, std::min( pos, txt.size() ) - initialPos + 1 ) );

      return strs.size();
  }  

  void LoadStragFile(std::vector<Opr_symbol> opr_symbol){
    std::ifstream myfile;
    char buf[80];
    getcwd(buf,80);
    std::string wd;
    wd = std::string(buf) + std::string("/results/");
    std::string file_name = wd+std::string("operator_")+std::to_string(ps::MyID())+".csv";
    LOG(INFO)<<"Prepare to open stragfile:"<<file_name;
    myfile.open(file_name);
    //myfile.open("results/operator_9.csv");
  //	vector<Opr_train> opr_train;
    if(myfile.is_open()){
      LOG(INFO)<<"FINISH OPEN STRAGFILE";
      int index = 0;
      std::string line;
      while(std::getline(myfile,line)){
        //std::cerr<<line<<std::endl;
        char* pattern = ",";
        size_t initpos = 0;
        size_t pos = line.find(pattern);
        std::string node_id = line.substr(0,pos);
        initpos = pos + 1;
        pos = line.find(pattern,initpos);
        std::string opr_name = line.substr(initpos,pos-initpos);
      //	std::cerr<<"opr_name:"<<opr_name;
        initpos = pos + 1;
        pos = line.find(pattern, initpos);
      //	std::cerr<<pos<<std::endl;
        time_t start_time = std::stol(line.substr(initpos,pos-initpos));
      //	std::cerr<<"start time:"<<start_time;
        initpos = pos + 1;
        pos = line.find(pattern, initpos);
        time_t end_time = std::stol(line.substr(initpos,pos-initpos));
        std::vector<std::string> opr_sub;
        split(opr_name.substr(1,opr_name.size()-1),opr_sub,' ');
        std::vector<int> dependent_keys;
        std::size_t found = opr_name.find("backward");
        if(found==std::string::npos){
          for(int i = 0; i<opr_sub.size();i++){
          auto temp_symbol = opr_symbol[index];
          if(temp_symbol.keys.size()!=0){
            for(int j=0; j<temp_symbol.keys.size();j++){
              dependent_keys.push_back(temp_symbol.keys[j]);
            }
          }
          index += 1;
          //std::cerr<<index<<" "<<opr_symbol.size()<<std::endl;
          if (index >= opr_symbol.size()){index=0;}
          }
        }else{index=0;}
        struct Opr_train temp;
        temp.opr_name = opr_name;
        temp.keys = dependent_keys;
        temp.start_time = start_time;
        temp.end_time = end_time;
        temp.runtime = end_time - start_time;
        opr_train.push_back(temp);
      }
    }
    myfile.close();
    LOG(INFO)<<"PREPARE TO REMOVE STRAGFILE.";
    std::remove(file_name.c_str());
    //std::remove("results/operator_9.csv");
    std::cerr<<"finish loading strag file.\n";
    /*
    for(auto it:opr_train){
      std::cerr<<"name:"<<it.opr_name;
      std::cerr<<" runtime:"<<it.runtime;
      std::cerr<<" dependent keys:";
      for(int i=0;i<it.keys.size();i++){
        std::cerr<< " "<<it.keys[i];
      }
      std::cerr<<"\n";
    }
    */
  }  


  void StragDetect(){
      std::vector<Opr_symbol> opr_symbol;
      std::ifstream myfile;
      char buf[80];
      getcwd(buf,80);
      std::string wd;
      wd = std::string(buf);
      std::string file_name = wd+std::string("/resnet-50-symbol.txt");
      myfile.open(file_name);
      //myfile.open("debug_str.txt");
      if(myfile.is_open()){
	LOG(INFO)<<"Open file for symbol.";
        std::string line;
        while(std::getline(myfile,line)){
          if(line!="--------------------"){
            continue;
          }else{
            std::string opr_name;
            getline(myfile,opr_name);
            char* pattern = ",";
            size_t pos = opr_name.find(pattern);
            std::string Op = opr_name.substr(3,pos-3);
            std::string name = opr_name.substr(pos+7,opr_name.size());
            getline(myfile,opr_name);
            std::vector<std::string> inputs;
            std::string line;
            while(true){
              std::getline(myfile,line);

              if(line[0]=='A' or line[0]=='V'){
                break;
              }else{
                inputs.push_back(line);
              }
            }
            struct Opr_symbol temp;
            temp.Op = Op;
            temp.name = name;
            temp.inputs = inputs;
            opr_symbol.push_back(temp);
          }
        }
      }
      myfile.close();
      std::string symbol_file = wd + std::string("/resnet-50.csv");
      auto keys = key_extract(symbol_file);
      LOG(INFO)<<"Open file for key names";	
      int count = 0;
      for(auto &opr:opr_symbol){
         for(auto &input:opr.inputs){
          count+=1;
          if(count<10){
              //std::cerr<<"input:"<<input<<std::endl;
          }
          for(int i=0;i<keys.size();i++){
              if(count<10){
          //	std::cerr<<"***********key:********************:"<<keys[i]<<std::endl;
              }
            auto pos = input.find(keys[i]);
            if(pos!=input.npos&&input[pos-1]=='='){
              opr.keys.push_back(i);
          //		std::cerr<<" key:"<<i<<std::endl;
            }
          }
        }
      }
    //	std::cerr<<"Enter to print\n"<<std::endl;
     /*
      for(struct Opr_symbol &opr:opr_symbol){
        std::cerr<<"opr:"<<opr.Op<<", name:"<<opr.name<<", input:\n"<<std::endl;
        for(int i=0;i<opr.inputs.size();i++){
          std::cerr<<opr.inputs[i]<<std::endl;
        }
        std::cerr<<"keys:\n"<<std::endl;
        for(int i=0;i<opr.keys.size();i++){
          std::cerr<<opr.keys[i]<<std::endl;
        }
      }*/                           
      LoadStragFile(opr_symbol);
      LOG(INFO)<<"FINISH load stragfile.";
      merge_trace();	  
      output_wait_time();  
  }


  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;
  /**
   * \brief the server handle
   */
  int step_push_default = 0;
  KVStoreDistServer* server_;
  /**
   * the scheduler handle
   */
  KVStoreDistScheduler* scheduler_;
  /**
   * \brief threshold for partition
   */
  size_t bigarray_bound_;
  /**
   * \brief buffer for non-compressed data.
   * When gradient compression is active, this is used
   * for the data in pull and for original data in push
   */
  std::unordered_map<int, NDArray> comm_buf_;
  /** \brief the version of the pulled key-value*/
  bool initialed = false;
  std::unordered_map<int, int> steps_;
  std::unordered_map<int, int> push_dsteps_;
  std::unordered_map<int, int> push_cbsteps_;
  std::unordered_map<int, int> pull_dsteps_;
  std::unordered_map<int, int> pull_cbsteps_;
  int global_max_steps = 0;  
  struct counter{
	int origin_key = -1;
	double total_count = 0;
	int number = 0;
  };
  std::unordered_map<int, struct counter> wait_time;
  /**
   * \brief buffer for compressed data
   * Used when gradient compression is active and action
   * is push
   */
  std::unordered_map<int, NDArray> compr_buf_;
  /**
   * \brief residual buffer to accumulate quantization error
   * during gradient compression
   */
  std::unordered_map<int, NDArray> residual_;
  bool log_verbose_;
/*
  //@yrchen:  
  struct Opr_symbol{
    std::string Op;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<int> keys;
  };
  struct Opr_train{
    std::string opr_name;
    std::string opr_symbol_name;
    std::vector<int> keys;
    time_t start_time;
    time_t end_time;
    time_t runtime;
  };
  struct Key_res{
    int key;
    time_t time;
  };
  struct Opr_Key{
    std::string opr_name;
    std::string opr_symbol_name;
    std::vector<int> keys;
    int key=-1;
    time_t start_time;
    time_t end_time;
    time_t runtime;
  };

  bool compare(struct Opr_Key a, struct Opr_Key b){
    return a.start_time < b.start_time;
  }

  std::vector<Opr_train> opr_train;
  std::vector<Key_res> key_res;
  */  
};

}  // namespace kvstore
//  int global_node_id = 0;
}  // namespace mxnet


#endif  // MXNET_KVSTORE_KVSTORE_DIST_H_
#endif
