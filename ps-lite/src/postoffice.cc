/**
 *  Copyright (c) 2015 by Contributors
 */
#include <unistd.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include "ps/internal/postoffice.h"
#include "ps/internal/message.h"
#include "ps/base.h"
#include <fstream>
#include <sys/time.h>
namespace ps {
Postoffice::Postoffice() {
  van_ = Van::Create("zmq");
  env_ref_ = Environment::_GetSharedRef();
  const char* val = NULL;
  val = CHECK_NOTNULL(Environment::Get()->find("DMLC_NUM_WORKER"));
  num_workers_ = atoi(val);
  val =  CHECK_NOTNULL(Environment::Get()->find("DMLC_NUM_SERVER"));
  num_servers_ = atoi(val);
  val = CHECK_NOTNULL(Environment::Get()->find("DMLC_ROLE"));
  std::string role(val);
  is_worker_ = role == "worker";
  is_server_ = role == "server";
  is_scheduler_ = role == "scheduler";
  verbose_ = GetEnv("PS_VERBOSE", 0);
  LOG(INFO) << "Postoffice is created.";
}

void Postoffice::Start(const char* argv0, const bool do_barrier) {
  // init glog
  if (argv0) {
    dmlc::InitLogging(argv0);
  } else {
    dmlc::InitLogging("ps-lite\0");
  }

  check_scaling_cmd_thread_ = std::unique_ptr<std::thread>(
	      new std::thread(&Postoffice::CheckScalingCMD, this));

  // wait until getting scaling_cmd by the thread
  while (scaling_cmd.empty()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }

  // init node info.
  if (scaling_cmd == "NONE") {
	  for (int i = 0; i < num_workers_; ++i) {
		int id = WorkerRankToID(i);
		for (int g : {id, kWorkerGroup, kWorkerGroup + kServerGroup,
				kWorkerGroup + kScheduler,
				kWorkerGroup + kServerGroup + kScheduler}) {
		  node_ids_[g].push_back(id);
		}
	  }
	  for (int i = 0; i < num_servers_; ++i) {
		int id = ServerRankToID(i);
		for (int g : {id, kServerGroup, kWorkerGroup + kServerGroup,
				kServerGroup + kScheduler,
				kWorkerGroup + kServerGroup + kScheduler}) {
		  node_ids_[g].push_back(id);
		}
	  }
  } else {
	  num_workers_ = 0;
	  num_servers_ = 0;
  }

  for (int g : {kScheduler, kScheduler + kServerGroup + kWorkerGroup,
          kScheduler + kWorkerGroup, kScheduler + kServerGroup}) {
    node_ids_[g].push_back(kScheduler);
  }

  // start van
  LOG(INFO) << "Before starting van.";
  van_->Start();
  LOG(INFO) << "After starting van.";
  // record start time
  start_time_ = time(NULL);
  // not allow scaling in when starting nodes
  CHECK_NE(scaling_cmd, "DEC_WORKER");
  CHECK_NE(scaling_cmd, "DEC_SERVER");
  // do a barrier here if no scaling out
  if (scaling_cmd == "NONE") {
	  if (do_barrier) Barrier(kWorkerGroup + kServerGroup + kScheduler);
  }
  LOG(INFO) << "Node " << van_->my_node().id << " finish basic setup.";
}

// need to use lock here or set flag
void Postoffice::UpdateNodeIDs(const int id, const bool addID){
  // only update what needed instead of clear all, which can cause errors
  // since num_server or num_worker is not corresponding to node id when scaling
  CHECK_NE(id, Node::kEmpty);
  const auto it = node_ids_.find(id);
  LOG(INFO) << "UPDATENODEID "<<addID<<" id "<<id;
  if (it != node_ids_.cend() && addID ) {
	  LOG(WARNING) << "Can not add, Node " << id << " is already in node_ids";
	  return;
  } else if (it == node_ids_.cend() && !addID ) {
	  LOG(WARNING) << "Can not delete, Node " << id << " is not in node_ids";
	  return;
  }

  int role;
  if (id == 1){
	  role = Node::SCHEDULER;
  } else if (id>=8 && id%2){
	  role = Node::WORKER;
  } else if (id>=8 && !(id%2)){
	  role = Node::SERVER;
  } else {
	  LOG(WARNING) << "Invalid Node ID.";
	  return;
  }

  if (role == Node::WORKER) {
	  if (addID){
		// add a worker id
		for (int g : {id, kWorkerGroup, kWorkerGroup + kServerGroup,
		            kWorkerGroup + kScheduler, kWorkerGroup + kServerGroup + kScheduler}) {
			node_ids_[g].push_back(id);
		}
		num_workers_++;
	  } else {
		// delete a worker id
		node_ids_.erase(id);
		for (int g : {kWorkerGroup, kWorkerGroup + kServerGroup,
			    kWorkerGroup + kScheduler, kWorkerGroup + kServerGroup + kScheduler}) {
			node_ids_[g].erase(std::remove(node_ids_[g].begin(), node_ids_[g].end(), id),
				node_ids_[g].end());
		}
		num_workers_--;
	  }
  } else if (role == Node::SERVER) {
	  if (addID) {
		 // add a server id
		  for (int g : {id, kServerGroup, kWorkerGroup + kServerGroup,
		          kServerGroup + kScheduler, kWorkerGroup + kServerGroup + kScheduler}) {
			  node_ids_[g].push_back(id);
		  }
		  num_servers_++;
	  } else {
		  // delete a server id
		  node_ids_.erase(id);
		  for (int g : {kServerGroup, kWorkerGroup + kServerGroup,
		  		  kServerGroup + kScheduler, kWorkerGroup + kServerGroup + kScheduler}) {
			  node_ids_[g].erase(std::remove(node_ids_[g].begin(), node_ids_[g].end(), id),
			  		node_ids_[g].end());
		  }
		  num_servers_--;
	  }
  }
  else {
	  CHECK_EQ(role, Node::SCHEDULER);
	  CHECK(addID); // must add scheduler
	  CHECK_EQ(id, kScheduler);
	  for (int g : {kScheduler, kScheduler + kServerGroup + kWorkerGroup,
	          kScheduler + kWorkerGroup, kScheduler + kServerGroup}) {
	    node_ids_[g].push_back(id);
	  }
  }
}

// this thread can also be in Van class. see which one is better later.
void Postoffice::CheckScalingCMD(){
	//const char* workdir = Environment::Get()->find("WORK_DIR");
	const char* workdir = "/home/net/test/";
	if (workdir == NULL){
		LOG(ERROR) << "Environment variable WORK_DIR is not set.";
//return;
	}
	int count = 0;
  while(!is_exit){
	//@yhpeng add scaling_cmd as environment variable for temporary tests
	  int node_id = -1;
	  		if (van_->IsReady()){
	  			node_id = van_->my_node().id;
	  		}
	std::string fn = std::string(workdir)+"SCALING.txt"+std::to_string(node_id);
	std::ifstream file(fn);
/*	if(!file.good()){
		LOG(ERROR)<<"SCALING FILE, "<<fn<<", NOT DETECTED!";
		return;
	}*/
	//std::string scaling_cmd;
	while(std::getline(file, scaling_cmd)){
		break;
	}
	if(scaling_cmd.empty()){
		scaling_cmd = "NONE";
	} else {
		CHECK(scaling_cmd.size());
	}

	if (scaling_cmd == "INC_SERVER" || scaling_cmd == "INC_WORKER"){
		is_added_node_ = true;
	}

	if(count%60==0){
		int node_id = -1;
		if (van_->IsReady()){
			node_id = van_->my_node().id;
		}
		LOG(INFO) << "Node " << node_id << " get scaling command when starting: " << scaling_cmd;
		count ++;
	}
    if (scaling_cmd == "DEC_WORKER"){
      //should expose interface in ps.h to high level mxnet kvstore
	  //send DEC_WORKER message to the scheduler
	//auto tic = std::chrono::system_clock::now();
	//std::cerr<<"**********************Delete Worker, Start Time:"<<tic<<std::endl;
	struct timeval tv;
	gettimeofday(&tv,NULL);
	time_t interval = 1000000*tv.tv_sec+tv.tv_usec;
	std::ofstream myfile;
	myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
	myfile<<"delete worker, start: "<<interval<<" ,";
	myfile.close();
      is_deled_node_ = true;
      Message req;
	  req.meta.recver = kScheduler;
	  req.meta.request = true;
	  req.meta.control.cmd = Control::DEC_WORKER;
	  req.meta.control.node.push_back(van_->my_node());
	  req.meta.timestamp = van_->GetTimestamp();
	  CHECK_GT(van_->Send(req), 0);
	  LOG(INFO) << "Node " << van_->my_node().id << " sent DEC_WORKER message.";
	  break;  //exit
    } else if (scaling_cmd == "DEC_SERVER"){
      is_deled_node_ = true;
      //should expose interface in ps.h to high level mxnet kvstore
	  //send DEC_SERVER message to the scheduler
      Message req;
	  req.meta.recver = kScheduler;
	  req.meta.request = true;
	  req.meta.control.cmd = Control::DEC_SERVER;
	  req.meta.control.node.push_back(van_->my_node());
	  req.meta.timestamp = van_->GetTimestamp();
	  CHECK_GT(van_->Send(req), 0);
	  LOG(INFO) << "Node " << van_->my_node().id << " sent DEC_SERVER message.";
	  break;  //exit
    }
    std::this_thread::sleep_for(std::chrono::seconds(1));
    count ++;
    
    if(true && count % 60 == 0&&is_scheduler_){
	  Message req;
    	  std::cerr<< " Request Get Strag Info!";
          req.meta.recver = kScheduler;
          req.meta.request = true;
          req.meta.control.cmd = Control::GETSTRAG;
          //req.meta.control.node.push_back(van_->my_node());
          req.meta.timestamp = van_->GetTimestamp();
	  CHECK_GT(van_->Send(req),0);
	  LOG(INFO) << "Request Get Strag Info!";
    }
  }
}

void Postoffice::Finalize(const bool do_barrier) {
  LOG(INFO) << "Node " << van_->my_node().id << ": Postoffice::Finalize() is called.";
  if (do_barrier) Barrier(kWorkerGroup + kServerGroup + kScheduler);
  LOG(INFO) << "Node " << van_->my_node().id << ": End barrier.";
  //check_scaling_cmd_thread_->join();
  van_->Stop();
  is_exit = true;
  if (exit_callback_) exit_callback_();
}


void Postoffice::AddCustomer(Customer* customer) {
  std::lock_guard<std::mutex> lk(mu_);
  int id = CHECK_NOTNULL(customer)->id();
  CHECK_EQ(customers_.count(id), (size_t)0) << "id " << id << " already exists";
  customers_[id] = customer;
}


void Postoffice::RemoveCustomer(Customer* customer) {
  std::lock_guard<std::mutex> lk(mu_);
  int id = CHECK_NOTNULL(customer)->id();
  customers_.erase(id);
}


Customer* Postoffice::GetCustomer(int id, int timeout) const {
  Customer* obj = nullptr;
  for (int i = 0; i < timeout*1000+1; ++i) {
    {
      std::lock_guard<std::mutex> lk(mu_);
      const auto it = customers_.find(id);
      if (it != customers_.end()) {
        obj = it->second;
        break;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  return obj;
}

void Postoffice::Barrier(int node_group) {
  if (GetNodeIDs(node_group).size() <= 1) return;
  auto role = van_->my_node().role;

  //not understand the logic well
  if (role == Node::SCHEDULER) {
    CHECK(node_group & kScheduler);
  } else if (role == Node::WORKER) {
    CHECK(node_group & kWorkerGroup);
  } else if (role == Node::SERVER) {
    CHECK(node_group & kServerGroup);
  }

  std::unique_lock<std::mutex> ulk(barrier_mu_);
  barrier_done_ = false;
  Message req;
  req.meta.recver = kScheduler;
  req.meta.request = true;
  req.meta.control.cmd = Control::BARRIER;
  req.meta.control.barrier_group = node_group;
  req.meta.timestamp = van_->GetTimestamp();
  CHECK_GT(van_->Send(req), 0);
  //std::cerr<<"Node "<<van_->my_node().id<<" sending barrier command"<<std::endl;
  barrier_cond_.wait(ulk, [this] {
      return barrier_done_;
    });
}

const std::vector<Range>& Postoffice::GetServerKeyRanges() {
  //@yhpeng: adapt to scaling servers
  if (server_key_ranges_.size() != (size_t)num_servers_) {
	server_key_ranges_.clear();
    for (int i = 0; i < num_servers_; ++i) {
      server_key_ranges_.push_back(Range(
          kMaxKey / num_servers_ * i,
          kMaxKey / num_servers_ * (i+1)));
    }
  }
  return server_key_ranges_;
}

const std::vector<Range>& Postoffice::GetServerKeyRanges_Scaling(int i){
  //@yrchen: fix the bugs when DEC_SERVER
  server_key_ranges_scaling_.clear();
  int pre_num_servers_ = num_servers_ + i;
//  std::cerr << "pre_num_servers:" << pre_num_servers_;
  for(int i = 0; i < pre_num_servers_; ++i){
//	std::cerr<< "current loop[" << i<<"], begin: "<<kMaxKey/pre_num_servers_ *i<<std::endl;
	server_key_ranges_scaling_.push_back(Range(
		kMaxKey / pre_num_servers_ * i,
		kMaxKey / pre_num_servers_ * (i+1)));
  }	
  return server_key_ranges_scaling_;
}

void Postoffice::Manage(const Message& recv) {
  CHECK(!recv.meta.control.empty());
  const auto& ctrl = recv.meta.control;
  if (ctrl.cmd == Control::BARRIER && !recv.meta.request) {
    barrier_mu_.lock();
    barrier_done_ = true;
    barrier_mu_.unlock();
    barrier_cond_.notify_all();
  }
}

std::vector<int> Postoffice::GetDeadNodes(int t) {
  std::vector<int> dead_nodes;
  if (!van_->IsReady() || t == 0) return dead_nodes;

  time_t curr_time = time(NULL);
  const auto& nodes = is_scheduler_
    ? GetNodeIDs(kWorkerGroup + kServerGroup)
    : GetNodeIDs(kScheduler);
  {
    std::lock_guard<std::mutex> lk(heartbeat_mu_);
    for (int r : nodes) {
      auto it = heartbeats_.find(r);
      if ((it == heartbeats_.end() || it->second + t < curr_time)
            && start_time_ + t < curr_time) {
        dead_nodes.push_back(r);
      }
    }
  }
  return dead_nodes;
}
}  // namespace ps
