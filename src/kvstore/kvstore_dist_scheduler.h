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

/*
 * server scaling:
 * (1) adding worker failed
 * (2) adding server sometimes failed with a low probability. Need to find out why.
 * (3) async training
 * (4) add multiple nodes?
 */


#ifndef MXNET_KVSTORE_KVSTORE_DIST_SCHEDULER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SCHEDULER_H_
#include <stdlib.h>
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include <chrono>
#include "ps/ps.h"
#include "mxnet/kvstore.h"
#include "./kvstore_dist_server.h"
#include "../operator/tensor/elemwise_binary_op-inl.h"
#include "../operator/tensor/init_op.h"
#include <sys/time.h>
namespace mxnet {
namespace kvstore {


class KVStoreDistScheduler {
 public:
	KVStoreDistScheduler() {
    using namespace std::placeholders;
    ps_scheduler_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_scheduler_)->set_request_handle(
        std::bind(&KVStoreDistScheduler::RequestCommandHandle, this, _1, _2));
    static_cast<ps::SimpleApp*>(ps_scheduler_)->set_response_handle(
            std::bind(&KVStoreDistScheduler::ResponseCommandHandle, this, _1, _2));
    log_verbose_ = dmlc::GetEnv("MXNET_KVSTORE_DIST_ROW_SPARSE_VERBOSE", false);
  }

  ~KVStoreDistScheduler() {
    delete ps_scheduler_;
  }

 private:

  void RequestCommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
	CommandType recved_type = static_cast<CommandType>(recved.head);
	LOG(INFO) << "The scheduler gets request command " << recved.head<<" "<<ps::kGetStragSignal;
	
    if (recved.head == ps::kIncServerSignal || recved.head == ps::kDecServerSignal) {
    	// start scaling
    	LOG(INFO) << "Get Server scaling signal from Van";
    	HandleServerScalingSig(recved, app);
    } else if(recved_type == CommandType::kRequestParaInfo){
	LOG(INFO) << "Get Worker Request Para Info";
	app->Response(recved);
	if(end_scaling_server_!=-1){
		last_scaling = key_map+" "+last_scaling;
		LOG(INFO)<<"Sending last_scaling to Initial Worker: "<<last_scaling;
		app->Request(static_cast<int>(CommandType::kRequestParaInfo), last_scaling, recved.sender);
	}else{app->Request(static_cast<int>(CommandType::kRequestParaInfo),"",recved.sender);}	
    } else if (recved.head == ps::kGetStragSignal|| recved.head == 103){
//	app->Response(recved);
	LOG(INFO)<<" Preparing to get strag info!";
	app->Request(static_cast<int>(CommandType::kGetStragInfo),"",ps::kWorkerGroup);
    } else {
    	LOG(WARNING) << "Unknown type of request command to scheduler!";
    }
    //app->Response(recved);
  }

  void ResponseCommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
	LOG(INFO) << "The scheduler gets response for command " << recved.head;
	CommandType recved_type = static_cast<CommandType>(recved.head);
	if (recved_type == CommandType::kGetParaInfo) {
		if(ReAssign==true){CollectParamInfoRe(recved, app);}
		else {
		      struct timeval tv;
		      gettimeofday(&tv,NULL);
		      time_t interval = 1000000*tv.tv_sec+tv.tv_usec;
		      std::ofstream myfile;
		      myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
		      myfile<<"stage 2, start: "<<interval<<", ";
		      myfile.close();
		      CollectParamInfo(recved, app);
		}
	} else if (recved_type == CommandType::kGetStragInfo){
		CollectStragInfo(recved,app);

	} else if (recved_type == CommandType::kSetParams){ // only server responses
		NotifyWorker(recved, app);
	} else if (recved_type == CommandType::kStartTraining){
		std::cerr<<" Scheduler receives command kStartTraining!"<<std::endl;
		EndScaling(recved, app);
	} else {
		LOG(WARNING) << "Unknown type of command to scheduler: " << recved.head;
	}
  }

  void HandleServerScalingSig(const ps::SimpleData& recved, ps::SimpleApp* app){
	if (scaling_server_id||scaling_server_id==-2) {
		LOG(WARNING)<<"The scaling of server " << std::abs(scaling_server_id)
		<< " is in process, can not accept another scaling!!!";
		return;
	}
	scaling_server_id = std::stoi(recved.body); // remember to clear after scaling out
	end_scaling_server_ = scaling_server_id;
	// clear existing info
	speeds_.clear();
	steps_.clear();
	ps_kvs_.clear();
	start_scaling_step = -1;
	tic = std::chrono::system_clock::now();

	// not include inc server since Postoffice::node_ids_ has not been updated on scheduler
	app->Request(static_cast<int>(CommandType::kGetParaInfo), "", ps::kServerGroup);
	barrier_group[ps::kServerGroup] = 0;
	LOG(INFO) << "Sent out kGetParaInfo request to servers.";
  }

  void HandleKeyRedistribute(ps::SimpleApp* app){
	steps_.clear();
	ps_kvs_.clear();
	tic = std::chrono::system_clock::now();
	app->Request(static_cast<int>(CommandType::kGetParaInfo),"",ps::kServerGroup);
	barrier_group[ps::kServerGroup] = 0;
	LOG(INFO) << "Send out kGetParaInfo request to servers for key redistribution.";
  }
  void CollectParamInfoRe(const ps::SimpleData& recved, ps::SimpleApp* app){
          // response message format: "speed step key:size key:size key:size"
          LOG(INFO) << "Received kGetParaInfo response from Node "
                        << recved.sender << " with Body " << recved.body;

          barrier_group[ps::kServerGroup] ++;
          if (recved.body.size()){
                  int server_id = recved.sender; // not server rank, but server node id
                  ParamInfo param_info = StringToParamInfo(recved.body);
                  speeds_[server_id] = param_info.speed;
                  steps_[server_id] = param_info.step;
                  CHECK_EQ(param_info.ps_kvs.size(),1);
                  ps_kvs_[server_id] = std::move(param_info.ps_kvs[server_id]);
          }
	
          if (barrier_group[ps::kServerGroup] == ps::NumServers()){
                  barrier_group[ps::kServerGroup] = 0;
                  CHECK_EQ(speeds_.size(), steps_.size());
                  CHECK_EQ(speeds_.size(), ps_kvs_.size());
                  LOG(INFO) << "Get kGetParaInfo responses from all servers.";
                  // calculate new parameter assignment across servers
                  ReAssignPSKVs();
                  //RandAssignPSKVs();
		  //RoundRobinAssignPSKVs();
		  // estimate when to start scaling
                  EstScalingStep();
                  // send new parameter assignment to servers and workers
                  ReSendParaAssign(app);
          }
		
  }
  void CollectParamInfo(const ps::SimpleData& recved, ps::SimpleApp* app){
	  // response message format: "speed step key:size key:size key:size"
	  LOG(INFO) << "Received kGetParaInfo response from Node "
	    		<< recved.sender << " with Body " << recved.body;

	  barrier_group[ps::kServerGroup] ++;
	  if (recved.body.size()){
		  int server_id = recved.sender; // not server rank, but server node id
		  ParamInfo param_info = StringToParamInfo(recved.body);
		  speeds_[server_id] = param_info.speed;
		  steps_[server_id] = param_info.step;
		  CHECK_EQ(param_info.ps_kvs.size(),1);
		  ps_kvs_[server_id] = std::move(param_info.ps_kvs[server_id]);
	  }
	  //check if received responses from all servers
	  if (barrier_group[ps::kServerGroup] == ps::NumServers()){
		  barrier_group[ps::kServerGroup] = 0;
		  CHECK_EQ(speeds_.size(), steps_.size());
		  CHECK_EQ(speeds_.size(), ps_kvs_.size());
		  LOG(INFO) << "Get kGetParaInfo responses from all servers.";
		  // calculate new parameter assignment across servers
		  AssignPSKVs();
		  // estimate when to start scaling
		  EstScalingStep();
		  // send new parameter assignment to servers and workers
		  struct timeval tv;
      		  gettimeofday(&tv,NULL);
      		  time_t interval = 1000000*tv.tv_sec+tv.tv_usec;
      		  std::ofstream myfile;
      		  myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
      		  myfile<<"stage 2, end: "<<interval<<"\n ";
      		  myfile.close();

		  SendParaAssign(app);
	  }
  }

  void CollectStragInfo(const ps::SimpleData& recved, ps::SimpleApp* app){
	  // response message format: "speed step key:size key:size key:size"
	  LOG(INFO) << "Received kGetStragInfo response from Node "
	    		<< recved.sender << " with Body " << recved.body;

	  barrier_group[ps::kWorkerGroup] ++;
	  if (recved.body.size()){
		  int server_id = recved.sender; // not server rank, but server node id
		  //TODO: decide the strag info
		  /*
		  StragInfo strag_info = StringToStragInfo(recved.body);
		  speeds_[server_id] = param_info.speed;
		  steps_[server_id] = param_info.step;
		  CHECK_EQ(param_info.ps_kvs.size(),1);
		  ps_kvs_[server_id] = std::move(param_info.ps_kvs[server_id]);
		  */
	  }
	  //check if received responses from all servers
	  if (barrier_group[ps::kWorkerGroup] == ps::NumWorkers()){
		  barrier_group[ps::kWorkerGroup] = 0;
		  CHECK_EQ(speeds_.size(), steps_.size());
		  CHECK_EQ(speeds_.size(), ps_kvs_.size());
		  LOG(INFO) << "Get kStragParaInfo responses from all Worker.";
		  //TODO: Decide if it needs to redistribute key
		  ReAssign = true;
		  if(ReAssign==true){
			//HandleKeyRedistribute(app);
			  // calculate new parameter assignment across servers
			  //ReAssignPSKVs();
			  // estimate when to start scaling
			  //EstScalingStep();
			  // send new parameter assignment to servers and workers
			  //ReSendParaAssign(app);
		  }
		  
	  }
  }


  /**
   * input: speeds_, steps_,
   * output: start_scaling_step
   * func: estimate when (in which iteration) to start scaling.
   */
  void EstScalingStep(){
	  double maxspeed = 0;
	  for(const auto& server_speed : speeds_){
		  if (maxspeed < server_speed.second){
			  maxspeed = server_speed.second;
		  }
	  }

	  int maxstep = 0;
	  for(const auto& server_step : steps_){
		  if (maxstep < server_step.second){
			  maxstep = server_step.second;
		  }
	  }

	  auto toc = std::chrono::system_clock::now();
	  std::chrono::duration<double> diff = toc - tic;
	  // x3 in case of failure
	  start_scaling_step = int(maxstep + 3 * maxspeed * diff.count()) + 3;
  }

  /*
   * Reassign parameters among servers
   */
  void AssignPSKVs(){
	  LOG(INFO) << "Running parameter assignment algorithm.";
	  // input: std::unordered_map<int, std::vector<std::pair<int, int>>> ps_kvs_;
	  // output: ps_kvs_
	  // heuristic: descending best fit
	  // same decoded key on different servers if NDArray size larger than bigarray_bound_
	  // should I uses encoded key?
	  size_t tot_size = 0;
	  std::vector<std::pair<int,int>> size_id_pairs;
	  for (const auto& server_kvs : ps_kvs_){
		  int server_id = server_kvs.first;
		  size_t ps_size = 0;
		  bool continue_signal = false;
		  for (const auto& key_size : server_kvs.second){
			  if(server_kvs.second.size()<=1){
				continue_signal = true;
				continue;
			  }
			  ps_size += key_size.second;
		  }
		  if(continue_signal){
			continue_signal = true;
			continue;
		  }
		  size_id_pairs.push_back(std::make_pair(ps_size, server_id));
		  tot_size += ps_size;
	  }

	  // calculate average parameter size
	  CHECK(scaling_server_id) << "No INC_SERVER or DEC_SERVER signal set!";
	  if (scaling_server_id > 0) { // scaling out
//		  size_t avg_size = tot_size / (ps_kvs_.size()+1) + 1;
		  size_t avg_size = tot_size / (size_id_pairs.size()+1) + 1;
		  // <size of parameters, server>, sorted from large to small
		  std::priority_queue<std::pair<int,int>> ps_sizes_pq;
		  for (const auto& size_id_pair : size_id_pairs){
			  ps_sizes_pq.push(size_id_pair);
		  }
		  size_t scaling_server_size = 0;
		  // move some parameters to the new server
		  while(!ps_sizes_pq.empty()) {
			  auto pair = ps_sizes_pq.top();
			  ps_sizes_pq.pop();
			  size_t ps_size = pair.first;
			  int server_id = pair.second;
			  size_t overflow = ps_size - avg_size; // should move such parameters to the new server
			  auto& kvs = ps_kvs_[server_id];
			  int bf_index = GetBestFit(kvs, overflow);

			  // move the best fit key-value to new server
			  ps_kvs_[scaling_server_id].push_back(kvs.at(bf_index));
			  scaling_server_size += kvs.at(bf_index).second;
			  // delete from the original server
			  ps_size -= kvs.at(bf_index).second;
			  kvs.erase(kvs.begin()+bf_index);
			  ps_sizes_pq.push(std::make_pair(ps_size, server_id));

			  if (scaling_server_size >= avg_size) break;
		  }
	  } else if (scaling_server_id < 0) {
		  size_t avg_size = tot_size / (ps_kvs_.size()-1) + 1;
		  // <size of parameters, server>, sorted from small to large
		  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int,int>>, Compare> ps_sizes_pq;
		  for (const auto& size_id_pair : size_id_pairs){
			  ps_sizes_pq.push(size_id_pair);
		  }
		  // move all parameters on the deleted server to others
		  auto& kvs = ps_kvs_[-scaling_server_id];
		  while(!ps_sizes_pq.empty()) {
			  auto pair = ps_sizes_pq.top();
			  ps_sizes_pq.pop();
			  size_t ps_size = pair.first;
			  int server_id = pair.second;
			  size_t underflow = ps_size - avg_size; // should move such parameters to this server
			  int bf_index = GetBestFit(kvs, underflow);

			  // move the best fit key-value to this server
			  ps_kvs_[server_id].push_back(kvs.at(bf_index));
			  ps_size += kvs.at(bf_index).second;
			  // delete from the to be deleted server
			  kvs.erase(kvs.begin()+bf_index);
			  ps_sizes_pq.push(std::make_pair(ps_size, server_id));

			  if (!kvs.size()) break;
		  }
		  ps_kvs_.erase(-scaling_server_id);
	  }

	  LOG(INFO) << "Finish parameter assignment algorithm";
  }

  void RandAssignPSKVs(){
  	std::unordered_map<int, std::vector<std::pair<int, int>>> ps_kvs_rand;
  	int num_servers = ps::NumServers();
  	for(auto& server_kvs : ps_kvs_){
  		for(auto& key : server_kvs.second){
  			int server = rand() % num_servers;
  			int server_id = 8 + server * 2;
  			ps_kvs_rand[server_id].push_back(key);
  		}
  	}
  	ps_kvs_.clear();
  	ps_kvs_ = ps_kvs_rand;
	for(auto& server_kvs:ps_kvs_){
		std::cerr<<"Key on Server "<<server_kvs.first<<std::endl;
		for(auto&key:server_kvs.second){
			std::cerr<<"key:"<<key.first<<", size:"<<key.second<<";  ";
		}
	}
	std::cerr<<"\n";
  }

  void RoundRobinAssignPSKVs(){
	std::unordered_map<int, std::vector<std::pair<int, int>>> ps_kvs_round;
  	int num_servers = ps::NumServers();
  	int count = 0;
  	for(auto& server_kvs : ps_kvs_){
  		for(auto& key : server_kvs.second){
  			int server = count % num_servers;
  			int server_id = 8 + server * 2;
  			ps_kvs_round[server_id].push_back(key);
  			count ++;
  		}
  	}
  	ps_kvs_.clear();
  	ps_kvs_ = ps_kvs_round;	
  }

  void ReAssignPSKVs(){
  	//TODO: Load balancing algorithm
          LOG(INFO) << "Running parameter assignment algorithm.";
          // input: std::unordered_map<int, std::vector<std::pair<int, int>>> ps_kvs_;
          // output: ps_kvs_
          // heuristic: descending best fit
          // same decoded key on different servers if NDArray size larger than bigarray_bound_
          // should I uses encoded key?
          size_t tot_size = 0;
          std::vector<std::pair<int,int>> size_id_pairs;
          for (const auto& server_kvs : ps_kvs_){
                  int server_id = server_kvs.first;
                  size_t ps_size = 0;
                  bool continue_signal = false;
                  for (const auto& key_size : server_kvs.second){
                          if(server_kvs.second.size()<=1){
                                continue_signal = true;
                                continue;
                          }
                          ps_size += key_size.second;
                  }
                  if(continue_signal){
                        continue_signal = true;
                        continue;
                  }
                  size_id_pairs.push_back(std::make_pair(ps_size, server_id));
                  tot_size += ps_size;
          }
          size_t avg_size = tot_size / (size_id_pairs.size()+1) + 1;
          std::priority_queue<std::pair<int,int>> ps_sizes_pq;
          for (const auto& size_id_pair : size_id_pairs){
                  ps_sizes_pq.push(size_id_pair);
          }
          size_t scaling_server_size = 0;
	  for(auto& server_kvs:ps_kvs_){
		int server_id = server_kvs.first;
		int bf_index;
		auto& kvs = ps_kvs_[server_id];
	 	if(server_id==8){
			for(int i = 0; i < kvs.size();i++){
			  std::cerr<<"check key "<<kvs.at(i).first<< " on server 8\n";
			  if(kvs.at(i).first>10){
			    bf_index = i;
			    ps_kvs_[10].push_back(kvs.at(bf_index));
			    kvs.erase(kvs.begin()+bf_index);
			  }
			}
		}	
		if(server_id==10){
                        for(int i = 0; i < kvs.size();i++){
                          std::cerr<<"check key "<<kvs.at(i).first<< " on server 10\n";
                          if(kvs.at(i).first<10){
                            bf_index = i;
                            ps_kvs_[8].push_back(kvs.at(bf_index));
                            kvs.erase(kvs.begin()+bf_index);
                          }
                        }
	
			//bf_index = 2;
                        //ps_kvs_[8].push_back(kvs.at(bf_index));
                        //kvs.erase(kvs.begin()+bf_index);			
		}
		
	  }
	/*
          while(!ps_sizes_pq.empty()) {
                  auto pair = ps_sizes_pq.top();
                  ps_sizes_pq.pop();
                  size_t ps_size = pair.first;
                  int server_id = pair.second;
                  auto& kvs = ps_kvs_[server_id];
                  int bf_index = GetBestFit(kvs, overflow);
                  ps_kvs_[scaling_server_id].push_back(kvs.at(bf_index));
                  scaling_server_size += kvs.at(bf_index).second;
                  ps_size -= kvs.at(bf_index).second;
                  kvs.erase(kvs.begin()+bf_index);
                  ps_sizes_pq.push(std::make_pair(ps_size, server_id));
          }*/
  }

  /*
   * find the best fit element from a <key, value_size> vector
   * return the index
   */
  int GetBestFit(const std::vector<std::pair<int, int>>& kvs, size_t fit_size){
	  CHECK(kvs.size());

	  // <gap, index> sorted from small to large
	  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int,int>>, Compare> pq;
	  for (size_t i=0; i<kvs.size(); i++){
		  auto pair = kvs.at(i); //<key, value_size>
		  size_t gap = std::abs(fit_size - pair.second);
		  pq.push(std::make_pair(gap, i));
	  }
	  return pq.top().second;
  }

  /*
   * Sent out new parameter assignment
   */
  void SendParaAssign(ps::SimpleApp* app){
	  // broadcast to all servers/workers
	  CHECK_GE(start_scaling_step,0);
	  ParamInfo param_info;
	  param_info.step = start_scaling_step;
	  // instead of speed, we place the scaling server id here.
	  param_info.speed = scaling_server_id;
	  param_info.ps_kvs = ps_kvs_;
	  std::string body = ParamInfoToString(param_info);
	  last_scaling = body;
	  LOG(INFO) << "New parameter assignment: " << body;
	  app->Request(static_cast<int>(CommandType::kSetParams), body, ps::kServerGroup+ps::kWorkerGroup);
	  LOG(INFO) << "Sent out new parameters assignment to all servers/workers";
  }

  void ReSendParaAssign(ps::SimpleApp* app){
  	//TODO
  	//app->Request(static_cast<int>(CommandType::kSetParams),body,ps::kServerGroup+ps::kWorkerGroup);
          // broadcast to all servers/workers
          //CHECK_GE(start_scaling_step,0);
          ParamInfo param_info;
          param_info.step = start_scaling_step;
          // instead of speed, we place the scaling server id here.
	  scaling_server_id = -2;
          param_info.speed = -2;
          param_info.ps_kvs = ps_kvs_;
          std::string body = ParamInfoToString(param_info);
          last_scaling = body;
          LOG(INFO) << "New parameter assignment: " << body;
          app->Request(static_cast<int>(CommandType::kSetParams), body, ps::kServerGroup+ps::kWorkerGroup);
          LOG(INFO) << "Sent out new parameters assignment to all servers/workers";

  }

  /*
   * Notify worker after server scaling
   */
  void NotifyWorker(const ps::SimpleData& recved, ps::SimpleApp* app){
	  CHECK(!(recved.sender%2)) << "Only servers send kSetParams response!";
	  barrier_group[ps::kServerGroup] ++;
	  if (barrier_group[ps::kServerGroup] == ps::NumServers()){ // not include the added server
	  	      struct timeval tv;
		      gettimeofday(&tv,NULL);
		      time_t interval = 1000000*tv.tv_sec+tv.tv_usec;
		      std::ofstream myfile;
		      myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
		      myfile<<"stage 3, end: "<<interval<<"\n";
		      myfile.close();

		  barrier_group[ps::kServerGroup] = 0;
		  LOG(INFO) << "All servers finished parameter move";
		  app->Request(static_cast<int>(CommandType::kStartTraining), "", ps::kWorkerGroup);
		  LOG(INFO) << "Sent kStartTraining to all workers";
	  }
  }

  // update metas after scaling
  void EndScaling(const ps::SimpleData& recved, ps::SimpleApp* app){
	  CHECK(recved.sender%2) << "Only workers send kStartTraining response!";
	  if(recved.body.size()){
		LOG(INFO)<<" Received kStartTraining response from Worker "<<recved.sender<<", key map body is: "<<recved.body;
		key_map = recved.body;
	  }
	  barrier_group[ps::kWorkerGroup] ++;
	  std::cerr<<"In function EndScaling, barrier_group["<<ps::kWorkerGroup
	  << "] = " << barrier_group[ps::kWorkerGroup] << " num_worker = " <<
	  ps::NumWorkers() << std::endl;
	  if (barrier_group[ps::kWorkerGroup] == ps::NumWorkers()){ // not include the added server
		  barrier_group[ps::kWorkerGroup] = 0;
		  LOG(INFO) << "THE WHOLE SCALING OF SERVER " << std::abs(scaling_server_id) << " IS OVER.";
		  // update node_ids_ in Postoffice
		if(scaling_server_id!=-2){
		  std::cerr<<"Call UPdateMetas for scheduler!"<<std::endl;
		  UpdateMetas(scaling_server_id);
		}
		  scaling_server_id = 0; //reset
		  start_scaling_step = -1;
	  }
  }

  // start scaling at this iteration
  int start_scaling_step = -1;
  std::string last_scaling;
  std::string key_map;
  int end_scaling_server_ = -1;
  std::unordered_map<int, double> speeds_;
  std::unordered_map<int, int> steps_;
  // save parameter info: server_id, <key,value_size> vector
  std::unordered_map<int, std::vector<std::pair<int, int>>> ps_kvs_;
  bool ReAssign = false;
  //
  std::unordered_map<int, int> barrier_group;

  ps::KVServer<float>* ps_scheduler_;

  // whether to LOG verbose information
  bool log_verbose_;
  // the node id of the scaling out/in server, < 0 means scaling in, > 0 means scaling out
  int scaling_server_id = 0;
  // tic: used to estimate the step on servers
  std::chrono::time_point<std::chrono::system_clock> tic;

  // sort from small to large
  struct Compare {
      constexpr bool operator()(std::pair<int, int> const & a,
                                std::pair<int, int> const & b) const noexcept
      { return a.first > b.first || (a.first == b.first && a.second > b.second); }
  };

};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SCHEDULER_H_
