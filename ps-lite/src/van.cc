/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/van.h"
#include <thread>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <fstream>
#include "ps/base.h"
#include "ps/sarray.h"
#include "ps/internal/postoffice.h"
#include "ps/internal/customer.h"
#include "./network_utils.h"
#include "./meta.pb.h"
#include "./zmq_van.h"
#include "./resender.h"
#include <sys/time.h>
#include <fstream>
namespace ps {

// interval in second between to heartbeast signals. 0 means no heartbeat.
// don't send heartbeast in default. because if the scheduler received a
// heartbeart signal from a node before connected to that node, then it could be
// problem.
static const int kDefaultHeartbeatInterval = 0;

Van* Van::Create(const std::string& type) {
  if (type == "zmq") {
    return new ZMQVan();
  } else {
    LOG(FATAL) << "unsupported van type: " << type;
    return nullptr;
  }
}

void Van::ProcessTerminateCommand() {
  PS_VLOG(1) << my_node().ShortDebugString() << " is stopped";
  ready_ = false;
}

void Van::ProcessAddNodeCommandAtScheduler(Message* msg, Meta& nodes, Meta& recovery_nodes) {
  recovery_nodes.control.cmd = Control::ADD_NODE;
  time_t t = time(NULL);
  size_t num_nodes = Postoffice::Get()->num_servers() + Postoffice::Get()->num_workers();
  if (nodes.control.node.size() == num_nodes) {
    // sort the nodes according their ip and port,
    std::sort(nodes.control.node.begin(), nodes.control.node.end(),
              [](const Node& a, const Node& b) {
                  return (a.hostname.compare(b.hostname) | (a.port < b.port)) > 0;
              });
    // assign node rank
    for (auto& node : nodes.control.node) {
      CHECK_EQ(node.id, Node::kEmpty);
      int id = node.role == Node::SERVER ?
               Postoffice::ServerRankToID(hist_num_servers_) :
               Postoffice::WorkerRankToID(hist_num_workers_);
      PS_VLOG(1) << "assign rank=" << id << " to node " << node.DebugString();
      node.id = id;
      Connect(node);
      std::cerr<<"Scheduler connects to node"<<node.id<<std::endl;
      if (node.role == Node::SERVER) num_servers_++;
      if (node.role == Node::WORKER) num_workers_++;
      if (node.role == Node::SERVER) hist_num_servers_++;
      if (node.role == Node::WORKER) hist_num_workers_++;
      Postoffice::Get()->UpdateHeartbeat(node.id, t);
    }
    //@yrchen:for debug
    global_nodes.push_back(my_node_);

    nodes.control.node.push_back(my_node_);
    nodes.control.cmd = Control::ADD_NODE;
    Message back;
    back.meta = nodes;
    for (int r : Postoffice::Get()->GetNodeIDs(
            kWorkerGroup + kServerGroup)) {
      back.meta.recver = r;
      back.meta.timestamp = timestamp_++;
      Send(back);
    }
    PS_VLOG(1) << "the scheduler is connected to "
               << num_workers_ << " workers and " << num_servers_ << " servers";
    ready_ = true;
  } else if (!recovery_nodes.control.node.empty()) {
    auto dead_nodes = Postoffice::Get()->GetDeadNodes(heartbeat_timeout);
    std::unordered_set<int> dead_set(dead_nodes.begin(), dead_nodes.end());
    // send back the recovery node
    CHECK_EQ(recovery_nodes.control.node.size(), 1);
    Connect(recovery_nodes.control.node[0]);
    Postoffice::Get()->UpdateHeartbeat(recovery_nodes.control.node[0].id, t);
    Message back;
    for (int r : Postoffice::Get()->GetNodeIDs(
            kWorkerGroup + kServerGroup)) {
      if (r != recovery_nodes.control.node[0].id
          && dead_set.find(r) != dead_set.end()) {
        // do not try to send anything to dead node
        continue;
      }
      // only send recovery_node to nodes already exist
      // but send all nodes to the recovery_node
      back.meta = (r == recovery_nodes.control.node[0].id) ? nodes : recovery_nodes;
      back.meta.recver = r;
      back.meta.timestamp = timestamp_++;
      Send(back);
    }
  }
}

void Van::UpdateLocalID(Message& msg, std::unordered_set<int>& deadnodes_set,
                        Meta& nodes, Meta& recovery_nodes) {
  auto& ctrl = msg.meta.control;
  size_t num_nodes = Postoffice::Get()->num_servers() + Postoffice::Get()->num_workers();
  // assign an id
  if (msg.meta.sender == Meta::kEmpty) {
    CHECK(is_scheduler_);
    CHECK_EQ(ctrl.node.size(), 1);
    if (nodes.control.node.size() < num_nodes) {
      nodes.control.node.push_back(ctrl.node[0]);
    } else {
      // some node dies and restarts
      CHECK(ready_);
      for (size_t i = 0; i < nodes.control.node.size() - 1; ++i) {
        const auto& node = nodes.control.node[i];
        if (deadnodes_set.find(node.id) != deadnodes_set.end() && node.role == ctrl.node[0].role) {
          auto& recovery_node = ctrl.node[0];
          // assign previous node id
          recovery_node.id = node.id;
          recovery_node.is_recovery = true;
          PS_VLOG(1) << "replace dead node " << node.DebugString()
                     << " by node " << recovery_node.DebugString();
          nodes.control.node[i] = recovery_node;
          recovery_nodes.control.node.push_back(recovery_node);
          break;
        }
      }
    }
  }

  // update my id
  for (size_t i = 0; i < ctrl.node.size(); ++i) {
    const auto& node = ctrl.node[i];
    if (my_node_.hostname == node.hostname && my_node_.port == node.port) {
      my_node_ = node;
      std::string rank = std::to_string(Postoffice::IDtoRank(node.id));
	if(node.id%2!=0&&node.id!=1){
	      std::ofstream myfile;
	      myfile.open("node_id.txt");
	      myfile << node.id<<std::endl;
	      myfile.close();
	}
#ifdef _MSC_VER
      _putenv_s("DMLC_RANK", rank.c_str());
#else
      setenv("DMLC_RANK", rank.c_str(), true);
#endif
    }
  }
}

void Van::ProcessHearbeat(Message &msg) {
  auto& ctrl = msg.meta.control;
  time_t t = time(NULL);
  for (auto &node : ctrl.node) {
    Postoffice::Get()->UpdateHeartbeat(node.id, t);
    if (is_scheduler_) {
      Message heartbeat_ack;
      heartbeat_ack.meta.recver = node.id;
      heartbeat_ack.meta.control.cmd = Control::HEARTBEAT;
      heartbeat_ack.meta.control.node.push_back(my_node_);
      heartbeat_ack.meta.timestamp = timestamp_++;
      // send back heartbeat
      Send(heartbeat_ack);
    }
  }
}

void Van::ProcessBarrierCommand(Message& msg) {
  auto& ctrl = msg.meta.control;
  if (msg.meta.request) {
    if (barrier_count_.empty()) {
      barrier_count_.resize(8, 0);
    }
    int group = ctrl.barrier_group;

    // @yhpeng could get invalid barrier message sent from shutdown workers/servers due to scaling in.
    // to filter them out, save them to barrier_group_ids_
    barrier_group_ids_[group].push_back(msg.meta.sender);
    std::vector<int> valid_group_nodes = Postoffice::Get()->GetNodeIDs(group);
    for (std::size_t k=0; k<barrier_group_ids_[group].size(); k++){
    	int node_id = barrier_group_ids_[group][k];
    	if(std::find(valid_group_nodes.begin(),valid_group_nodes.end(), node_id) == valid_group_nodes.end()){
    	// remove invalid barrier node id
    	barrier_group_ids_[group].erase(barrier_group_ids_[group].begin()+k);
    	}
    }
    barrier_count_[group] = barrier_group_ids_[group].size();
    PS_VLOG(1) << "Barrier count for " << group << " : " << barrier_count_[group];
    PS_VLOG(1) << "Barrier group size for " << group << " : " << Postoffice::Get()->GetNodeIDs(group).size();

    if (barrier_count_[group] ==
        static_cast<int>(Postoffice::Get()->GetNodeIDs(group).size())) {
      barrier_group_ids_[group].clear();
      barrier_count_[group] = 0;
      Message res;
      res.meta.request = false;
      res.meta.control.cmd = Control::BARRIER;
      for (int r : Postoffice::Get()->GetNodeIDs(group)) {
        res.meta.recver = r;
        res.meta.timestamp = timestamp_++;
        CHECK_GT(Send(res), 0);
      }
    }
  } else {
    Postoffice::Get()->Manage(msg);
  }
}

void Van::ProcessDataMsg(Message &msg) {
  // data msg
  CHECK_NE(msg.meta.sender, Meta::kEmpty);
  CHECK_NE(msg.meta.recver, Meta::kEmpty);
  CHECK_NE(msg.meta.customer_id, Meta::kEmpty);
  if(my_node_.id%2==0){
	struct timeval tv;
	gettimeofday(&tv,NULL);
	msg.meta.receive_time = 1000000*tv.tv_sec+tv.tv_usec;
  }
  int id = msg.meta.customer_id;
  auto* obj = Postoffice::Get()->GetCustomer(id, 5);
  CHECK(obj) << "Node " << my_node_.id << " timeout (5 sec) to wait App " << id << " ready";
  obj->Accept(msg);
}

void Van::ProcessAddNodeCommand(Message& msg, Meta& nodes, Meta& recovery_nodes) {
  auto dead_nodes = Postoffice::Get()->GetDeadNodes(heartbeat_timeout);
  std::unordered_set<int> dead_set(dead_nodes.begin(), dead_nodes.end());
  auto& ctrl = msg.meta.control;

  UpdateLocalID(msg, dead_set, nodes, recovery_nodes);

  if (is_scheduler_) {
    ProcessAddNodeCommandAtScheduler(&msg, nodes, recovery_nodes);
  } else {
    for (const auto& node : ctrl.node) {
      Connect(node);
      if (!node.is_recovery && node.role == Node::SERVER) ++num_servers_;
      if (!node.is_recovery && node.role == Node::WORKER) ++num_workers_;
    }
    std::cerr<<"This is node "<<my_node_.id<<" in ProcessAddNodeCommand!"<<std::endl;
    PS_VLOG(1) << my_node_.ShortDebugString() << " is connected to others";
    ready_ = true;
  }
}

void Van::ProcessIncWorkerCommand(Message& msg, Meta& nodes) {
  auto& ctrl = msg.meta.control;
  if (is_scheduler_) {
	auto node = ctrl.node[0];
	// assign id to the new node, connect to it and update num_worker, heartbeat
	CHECK_EQ(node.id, Node::kEmpty);
	int id = node.role == Node::SERVER ?
	    Postoffice::ServerRankToID(hist_num_servers_) :
	    Postoffice::WorkerRankToID(hist_num_workers_);
	PS_VLOG(1) << "assign rank=" << id << " to node " << node.DebugString();
	node.id = id;

	// push node to nodes once assigned id
	nodes.control.node.push_back(node);
	// @yrchen: for debug
	global_nodes.push_back(node);

	Connect(node);
	// PS_VLOG(1) << "node.role: " << node.role << (node.role==Node::WORKER);
	CHECK_EQ(node.role, Node::WORKER);
	num_workers_++;
    hist_num_workers_++;
	time_t t = time(NULL);
	Postoffice::Get()->UpdateHeartbeat(node.id, t);

	// send new node to existing nodes and send existing nodes to new node
	Message back;
	back.meta.control.cmd = Control::INC_WORKER;
	back.meta.control.node.push_back(node);
	//have not update node_ids[], so is existing nodes
	for (int r : Postoffice::Get()->GetNodeIDs(kWorkerGroup + kServerGroup)) {
	      back.meta.recver = r;
	      back.meta.timestamp = timestamp_++;
	      Send(back);
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(1)); // make sure other connections are setup

	back.meta = nodes;
	back.meta.control.cmd = Control::INC_WORKER;
	back.meta.recver = node.id;
	back.meta.timestamp = timestamp_++;
	Send(back);

	PS_VLOG(1) << "sent the new node " << node.id << " to existing workers and servers";

	// update Postoffice::num_workers_ and Postoffice::node_ids
	// so that the new workers can exit normally when scheduler finalizes kWorkerGroup+kServerGroup+kScheduler
	Postoffice::Get()->UpdateNodeIDs(node.id, true);

  } else {
	// workers or servers get INC_WORKER, must update num_workers_ and node_ids for them
	// due to "merged->request.size() == (size_t) ps::NumWorkers()" for sync training in kvstore_dist_server.h
	  for (const auto& node : ctrl.node) {
	  // the new node needs to get id
	    if (my_node_.hostname == node.hostname && my_node_.port == node.port) {
	      PS_VLOG(1) << "Get new node id " << node.id;
	      my_node_ = node;
	      std::string rank = std::to_string(Postoffice::IDtoRank(node.id));
	      #ifdef _MSC_VER
	        _putenv_s("DMLC_RANK", rank.c_str());
	      #else
	        setenv("DMLC_RANK", rank.c_str(), true);
	      #endif
	    }
	  }
	// must get id before connecting, otherwise error
    for (const auto& node : ctrl.node) {
      // connect to nodes
      Connect(node);
      // the new node may receive server nodes
      if (node.role == Node::SERVER) num_servers_++;
      if (node.role == Node::WORKER) num_workers_++;
      PS_VLOG(1) << my_node_.ShortDebugString() << " is connected to new node " << node.id;
      Postoffice::Get()->UpdateNodeIDs(node.id, true);
    }
    ready_ = true;
  }
}

void Van::ProcessDecWorkerCommand(Message& msg, Meta& nodes) {
	auto& ctrl = msg.meta.control;
	if (is_scheduler_) {
		auto node = ctrl.node[0];
		//
		CHECK_NE(node.id, Node::kEmpty);

		// delete the node from nodes, decrease num_workers_ and delete from heartbeats
		int index = -1;
		for (size_t i = 0; i < nodes.control.node.size(); ++i) {
			const auto& node_ = nodes.control.node[i];
			if (node.id == node_.id && node.hostname == node_.hostname && node.port == node_.port){
				index = i;
				break;
			}
		}
		CHECK_GT(index, -1) << "Can not find to be deleted node " << node.id << " in nodes.";
		nodes.control.node.erase(nodes.control.node.begin()+index);
		CHECK_EQ(node.role, Node::WORKER);
		--num_workers_;
		Postoffice::Get()->DelHeartbeat(node.id);

		// send new node to existing nodes and send existing nodes to new node
		Message back;
		back.meta.control.cmd = Control::DEC_WORKER;
		back.meta.control.node.push_back(node);
		//have not update node_ids[], so node exists in kWorkerGroup
		for (int r : Postoffice::Get()->GetNodeIDs(kWorkerGroup + kServerGroup)) {
		      back.meta.recver = r;
		      back.meta.timestamp = timestamp_++;
		      Send(back);
		}
		PS_VLOG(1) << "Sent the deleted node " << node.id << " to existing workers and servers";

		// update Postoffice::num_workers_ and Postoffice::node_ids
		Postoffice::Get()->UpdateNodeIDs(node.id, false);

	  } else {
		// workers or servers get DEC_WORKER, no need to update Postoffice::num_workers_ and node_ids for them
		  for (const auto& node : ctrl.node) {
			PS_VLOG(1) << my_node_.ShortDebugString() << " gets deleted node " << node.id;
			CHECK_EQ(node.role, Node::WORKER);
		    if (my_node_.hostname == node.hostname && my_node_.port == node.port) {
		        // exit
		    	PS_VLOG(1) << my_node().ShortDebugString() << " is stopped";
		    	terminated_=true;
		    	// Write Finish
		    	//const char* workdir = Environment::Get()->find("WORK_DIR");
			const char* workdir = "/home/net/test/";
		    	if (workdir == NULL){
		    		LOG(ERROR) << "Environment variable WORK_DIR is not set.";
		    	} else {
					std::string fn = std::string(workdir)+"SCALING.txt";
					std::ofstream file;
					file.open(fn);
					file << "FINISH\n";
					file.close();
//					auto toc = std::chrono::system_clock::now();
//					std::cerr<<"***********************Delete Worker End time:"<<toc<<std::endl;
				       struct timeval tv;
 				       gettimeofday(&tv,NULL);
       					time_t interval = 1000000*tv.tv_sec+tv.tv_usec;

					std::ofstream myfile;
				        myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
					myfile<<"delete worker, END: "<<interval<<" \n";
					myfile.close();
		    	}
		    	exit(0);
		    }
		    else{
		    	--num_workers_;
		    	// no need to delete socket since Connect() will reconnect the socket next time.
		    	Postoffice::Get()->UpdateNodeIDs(node.id, false);
		    }
		  }
	  }
}

void Van::ProcessIncServerCommand(Message& msg, Meta& nodes) {
  auto& ctrl = msg.meta.control;
  if (is_scheduler_) {
	auto node = ctrl.node[0];
	// assign id to the new node, connect to it and update num_worker, heartbeat
	CHECK_EQ(node.id, Node::kEmpty);
	int id = node.role == Node::SERVER ?
	    Postoffice::ServerRankToID(hist_num_servers_) :
	    Postoffice::WorkerRankToID(hist_num_workers_);
	PS_VLOG(1) << "assign rank=" << id << " to node " << node.DebugString();
	LOG(INFO)<<"assign rank = "<<id<<" to node "<< node.DebugString()
	<<", nodes.debugstr():"<<nodes.DebugString();
	node.id = id;
	
	// push node to nodes once assigned id
	nodes.control.node.push_back(node);

	Connect(node);
	// PS_VLOG(1) << "node.role: " << node.role << (node.role==Node::WORKER);
	CHECK_EQ(node.role, Node::SERVER);
	num_servers_++;
	hist_num_servers_++;
	time_t t = time(NULL);
	Postoffice::Get()->UpdateHeartbeat(node.id, t);
	LOG(INFO)<<"node.debugstring:"<<node.DebugString()<<"\n global_node:";
	for(const Node& n:global_nodes)std::cerr<<" "<<n.DebugString();
	std::cerr<<"\n";
	// send new node to existing nodes and send existing nodes to new node
	Message back;
	back.meta.control.cmd = Control::INC_SERVER;
	back.meta.control.node.push_back(node);
	//have not update node_ids[], so is existing nodes
	for (int r : Postoffice::Get()->GetNodeIDs(kWorkerGroup + kServerGroup)) {
	      back.meta.recver = r;
	      back.meta.timestamp = timestamp_++;
	      Send(back);
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(1)); // make sure other connections are setup

	back.meta = nodes;
	back.meta.control.cmd = Control::INC_SERVER;
	back.meta.recver = node.id;
	back.meta.timestamp = timestamp_++;
	Send(back);

	PS_VLOG(1) << "sent the new node " << node.id << " to existing workers and servers";
	LOG(INFO)<< "Sent the new node "<< node.id<<" to existing workers and servers:"<<back.DebugString();
	// update Postoffice::num_workers_ and Postoffice::node_ids
	// so that the new workers can exit normally when scheduler finalizes kWorkerGroup+kServerGroup+kScheduler
	// Postoffice::Get()->UpdateNodeIDs(node.id, true);

	// wait for all connections done
	// send a message to trigger scaling from upper layer
      struct timeval tv;
      gettimeofday(&tv,NULL);
      time_t interval = 1000000*tv.tv_sec+tv.tv_usec;
      std::ofstream myfile;
      myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
      myfile<<"stage 1-registration, end: "<<interval<<"\n";
      myfile.close();

	SendCommandToUpperLayer(kIncServerSignal, std::to_string(node.id));

  } else {
	// workers or servers get INC_SERVER, need to update num_workers_ and node_ids later from high level API
	  bool is_self = false;
	  for (const auto& node : ctrl.node) {
	  // the new node needs to get id
	    if (my_node_.hostname == node.hostname && my_node_.port == node.port) {
	      PS_VLOG(1) << "Get new node id " << node.id;
	      my_node_ = node;
	      std::string rank = std::to_string(Postoffice::IDtoRank(node.id));
	      #ifdef _MSC_VER
	        _putenv_s("DMLC_RANK", rank.c_str());
	      #else
	        setenv("DMLC_RANK", rank.c_str(), true);
	      #endif
	      is_self = true;
	    }
	  }
	// must get id before connecting
    for (const auto& node : ctrl.node) {
      // connect to nodes
      Connect(node);
      // the new node may receive server nodes
      if (node.role == Node::SERVER) num_servers_++;
      if (node.role == Node::WORKER) num_workers_++;
      PS_VLOG(1) << my_node_.ShortDebugString() << " is connected to new node " << node.id;
      if (is_self)  Postoffice::Get()->UpdateNodeIDs(node.id, true);
    }
    ready_ = true; // server is ready, but others are not very aware of the changes.
  }
}
/*
void Van::ProcessGetStragCommand(Message& msg, Meta& nodes){
	auto& ctrl = msg.meta.control;
	if (is_scheduler_){
		
	}
}*/


void Van::ProcessDecServerCommand(Message& msg, Meta& nodes) {
	auto& ctrl = msg.meta.control;
	if (is_scheduler_) {
		auto node = ctrl.node[0];
		//
		CHECK_NE(node.id, Node::kEmpty);

		// delete the node from nodes, decrease num_servers_ and delete from heartbeats
		int index = -1;
		for (size_t i = 0; i < nodes.control.node.size(); ++i) {
			const auto& node_ = nodes.control.node[i];
			if (node.id == node_.id && node.hostname == node_.hostname && node.port == node_.port){
				index = i;
				break;
			}
		}
		CHECK_GT(index, -1) << "Can not find to be deleted node " << node.id << " in nodes.";
		nodes.control.node.erase(nodes.control.node.begin()+index);
		CHECK_EQ(node.role, Node::SERVER);
		--num_servers_;
		Postoffice::Get()->DelHeartbeat(node.id);

		// send new node to existing nodes and send existing nodes to new node
		Message back;
		back.meta.control.cmd = Control::DEC_SERVER;
		back.meta.control.node.push_back(node);
		//have not update node_ids[], so node exists in kWorkerGroup
		for (int r : Postoffice::Get()->GetNodeIDs(kWorkerGroup + kServerGroup)) {
		      back.meta.recver = r;
		      back.meta.timestamp = timestamp_++;
		      Send(back);
		}
		PS_VLOG(1) << "Sent the deleted node " << node.id << " to existing workers and servers";

		// update Postoffice::num_workers_ and Postoffice::node_ids
		// Postoffice::Get()->UpdateNodeIDs(node.id, false);
		SendCommandToUpperLayer(kDecServerSignal, std::to_string(-node.id));

	  } else {
		// workers or servers get DEC_SERVER, need to update Postoffice::num_workers_ and node_ids later
		  for (const auto& node : ctrl.node) {
			PS_VLOG(1) << my_node_.ShortDebugString() << " gets deleted node " << node.id;
			CHECK_EQ(node.role, Node::SERVER);
		    if (my_node_.hostname == node.hostname && my_node_.port == node.port) {
		        // exit
		    }
		    else{
		    	--num_servers_;
		   // no need to delete socket since Connect() will reconnect the socket next time if same id.
		    }
		  }
	  }
}

void Van::SendCommandToUpperLayer(int req_head, const std::string& req_body){
	// setup message
	Message msg;
	msg.meta.head = req_head;
	if (req_body.size()) msg.meta.body = req_body;
	msg.meta.request = true;
	msg.meta.simple_app = true;
	msg.meta.timestamp = -1;
	int self_id = my_node_.id;
	CHECK_EQ(self_id, kScheduler) << "Only scheduler itself sends message to its upper layer";
	msg.meta.customer_id = 0; // default
	msg.meta.recver = self_id;
	Send(msg);
	PS_VLOG(1) << "Send Command " << req_head << " to upper layer, i.e., SimpleApp";
}

void Van::Start() {
  // get scheduler info
  scheduler_.hostname = std::string(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_URI")));
  scheduler_.port     = atoi(CHECK_NOTNULL(Environment::Get()->find("DMLC_PS_ROOT_PORT")));
  scheduler_.role     = Node::SCHEDULER;
  scheduler_.id       = kScheduler;
  is_scheduler_       = Postoffice::Get()->is_scheduler();

  // get my node info
  if (is_scheduler_) {
    my_node_ = scheduler_;
  } else {
    auto role = is_scheduler_ ? Node::SCHEDULER :
                (Postoffice::Get()->is_worker() ? Node::WORKER : Node::SERVER);
    const char* nhost = Environment::Get()->find("DMLC_NODE_HOST");
    std::string ip;
    if (nhost) ip = std::string(nhost);
    if (ip.empty()) {
      const char*  itf = Environment::Get()->find("DMLC_INTERFACE");
      std::string interface;
      if (itf) interface = std::string(itf);
      if (interface.size()) {
        GetIP(interface, &ip);
      } else {
        GetAvailableInterfaceAndIP(&interface, &ip);
      }
      CHECK(!interface.empty()) << "failed to get the interface";
    }
    int port = GetAvailablePort();
    const char* pstr = Environment::Get()->find("PORT");
    if (pstr) port = atoi(pstr);
    CHECK(!ip.empty()) << "failed to get ip";
    CHECK(port) << "failed to get a port";
    my_node_.hostname = ip;
    my_node_.role     = role;
    my_node_.port     = port;
    // cannot determine my id now, the scheduler will assign it later
    // set it explicitly to make re-register within a same process possible
    my_node_.id = Node::kEmpty;
  }

  // bind.
  my_node_.port = Bind(my_node_, is_scheduler_ ? 0 : 40);
  PS_VLOG(1) << "Bind to " << my_node_.DebugString();
  CHECK_NE(my_node_.port, -1) << "bind failed";

  // connect to the scheduler
  Connect(scheduler_);

  // for debug use
  if (Environment::Get()->find("PS_DROP_MSG")) {
    drop_rate_ = atoi(Environment::Get()->find("PS_DROP_MSG"));
  }
  // start receiver
  receiver_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Receiving, this));
  bool inc_worker=false;
      auto tic = std::chrono::system_clock::now();
  if (!is_scheduler_) {
    // let the scheduler know myself
    Message msg;
    msg.meta.recver = kScheduler;
    // @yhpeng this node is a new worker
    if (Postoffice::Get()->GetScalingCMD() == "INC_WORKER") {
      inc_worker = true;
      tic = std::chrono::system_clock::now();
      msg.meta.control.cmd = Control::INC_WORKER;
      PS_VLOG(1) << "Sent out INC_WORKER message to scheduler.";
    } else if (Postoffice::Get()->GetScalingCMD() == "INC_SERVER") {
      msg.meta.control.cmd = Control::INC_SERVER;
      struct timeval tv;
      gettimeofday(&tv,NULL);
      time_t interval = 1000000*tv.tv_sec+tv.tv_usec;
      std::ofstream myfile;
      myfile.open("/home/net/test/overhead.txt",std::ofstream::out|std::ofstream::app);
      myfile<<"stage 1, start: "<<interval<<", ";
      myfile.close();
	
      PS_VLOG(1) << "Sent out INC_SERVER message to scheduler.";
    }
    else {
      msg.meta.control.cmd = Control::ADD_NODE;
    }

    msg.meta.control.node.push_back(my_node_);
    msg.meta.timestamp = timestamp_++;
    Send(msg);
  }
  // wait until ready
  while (!ready_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  std::cerr<<"Node is ready_!"<<std::endl;
  if(inc_worker){
	auto toc = std::chrono::system_clock::now();
	std::chrono::duration<double> diff = toc-tic;
	std::ofstream myfile;
	myfile.open("/home/net/test/overhead.txt",std::ofstream::out | std::ofstream::app);
	myfile << "inc worker time for worker "<<diff.count()<<"\n";
	myfile.close();
	inc_worker = false;    
  } 
  // resender
  if (Environment::Get()->find("PS_RESEND") && atoi(Environment::Get()->find("PS_RESEND")) != 0) {
    int timeout = 1000;
    if (Environment::Get()->find("PS_RESEND_TIMEOUT")) {
      timeout = atoi(Environment::Get()->find("PS_RESEND_TIMEOUT"));
    }
    resender_ = new Resender(timeout, 10, this);
  }

  if (!is_scheduler_) {
    // start heartbeat thread
    heartbeat_thread_ = std::unique_ptr<std::thread>(
      new std::thread(&Van::Heartbeat, this));
  }
}

void Van::Stop() {
  // stop threads
  // send TERMINATE message to myself
  Message exit;
  exit.meta.control.cmd = Control::TERMINATE;
  exit.meta.recver = my_node_.id;
  SendMsg(exit);
  receiver_thread_->join();
  if (!is_scheduler_) heartbeat_thread_->join();
  if (resender_) delete resender_;
}

int Van::Send(const Message& msg) {
  if(my_node_.id%2!=0){
        //std::cerr<<"This is worker "<<my_node_.id<<", Sending MSG, Request time:"<<msg.meta.response_time;
//	std::cerr<<msg.DebugString();
  }else{
	if(msg.meta.receive_time==0||true){
		//LOG(INFO)<<"server prepare to response: start"<<msg.meta.start_time
		//<<", rcv:"<<msg.meta.receive_time<<", res:"<<msg.meta.response_time;
	}
  }

  int send_bytes = SendMsg(msg);
  if(my_node_.id%2!=0){
	//std::cerr<<"This is worker "<<my_node_.id<<", Sending MSG, Request time:"<<msg.meta.response_time
	//<<" return send_bytes:"<<send_bytes<<std::endl;
	
  }
  CHECK_NE(send_bytes, -1) << "Node: " << my_node_.id << " The message content: " << msg.DebugString();
  send_bytes_ += send_bytes;
  if (resender_) resender_->AddOutgoing(msg);
  if (Postoffice::Get()->verbose() >= 2) {
//    PS_VLOG(2) << "Sending " << msg.DebugString();
  }
  return send_bytes;
}

void Van::Receiving() {

  Meta nodes;
  Meta recovery_nodes;  // store recovery nodes
  recovery_nodes.control.cmd = Control::ADD_NODE;

  while (true) {
    Message msg;
    int recv_bytes = RecvMsg(&msg);
    // For debug, drop received message
    if (ready_ && drop_rate_ > 0) {
      unsigned seed = time(NULL) + my_node_.id;
      if (rand_r(&seed) % 100 < drop_rate_) {
        LOG(WARNING) << "Drop message " << msg.DebugString();
        continue;
      }
    }

    CHECK_NE(recv_bytes, -1);
    recv_bytes_ += recv_bytes;
    if (Postoffice::Get()->verbose() >= 2||msg.meta.control.cmd==Control::INC_SERVER) {
      PS_VLOG(2) << msg.DebugString();
    }
    // duplicated message
    if (resender_ && resender_->AddIncomming(msg)) continue;
    if (my_node_.id%2==0){
//	std::cerr<<"This is Server "<<my_node_.id<<" , start time:"<<msg.meta.start_time
//	<<", receive_time:";
	struct timeval tv;
	gettimeofday(&tv,NULL);
	time_t time_interval = 1000000*tv.tv_sec+tv.tv_usec - msg.meta.start_time;
//	if(time_interval<0)LOG(INFO)<<"Server receive time < 0 :"<<time_interval;
//	LOG(INFO)<<"DEBUGE start_time on server:"<<msg.meta.start_time<<" receiving:"<<time_interval;
//	std::cerr<<"0.5 RTT costs:"<<time_interval<<std::endl;	
	if(msg.meta.response_time==0)//std::cerr<<msg.meta.DebugString();
//	std::cerr<<time_interval<<" time on server:"<<1000000*tv.tv_sec+tv.tv_usec<<std::endl;
//	msg.meta.receive_time = time_interval;
	msg.meta.receive_time = 1000000*tv.tv_sec+tv.tv_usec;
    }else if (my_node_.id!=1){
//	std::cerr<<"This is Worker "<<my_node_.id<<" , request time:"<<msg.meta.response_time<<std::endl;
	if(msg.meta.start_time!=0){
	//    std::cerr<<"Receive Response from server "<<msg.meta.sender;
//	    if(msg.meta.sender==12){std::cerr<<"Receiving message from Server 12"<<std::endl;}
	    if(msg.meta.receive_time==0){
//		LOG(INFO)<<"WORKER RECEIVE TIME ==0";
          //    LOG(INFO)<<" start:"<<msg.meta.start_time<<", rcv:"<<msg.meta.receive_time<<", res:"
           //   <<msg.meta.response_time<<", end:"<<msg.meta.end_time<<std::endl;

	    }
	    struct timeval tv;
	    gettimeofday(&tv,NULL);
	    time_t time_interval = 1000000*tv.tv_sec + tv.tv_usec - msg.meta.start_time;
	    if(time_interval<0)LOG(INFO)<<"KeyTimeRecorder error! end_time:"<<time_interval;
//	    msg.meta.end_time = time_interval;
	    msg.meta.end_time = 1000000*tv.tv_sec + tv.tv_usec;
//	    LOG(INFO)<<" start:"<<msg.meta.start_time<<", rcv:"<<msg.meta.receive_time<<", res:"
//		<<msg.meta.response_time<<", end:"<<msg.meta.end_time<<std::endl;
	   if(time_interval>0 && time_interval<20000000){
	    auto it = detect_map.find(msg.meta.sender);
	   
	    if (it==detect_map.end()){
		struct response_detect detect;
		detect.sum_time += time_interval;
		detect.number += 1;
		detect_map.insert({msg.meta.sender,detect});
		std::cerr<<"*******************************************************\n New Node Inserted!\n";
	    } else{
		it->second.sum_time += time_interval;
		it->second.number += 1;
		if(it->second.number%100 == 0){
		    it->second.detect_ave.push_back(it->second.sum_time/it->second.number);
		    it->second.ave_num += 1;
		    it->second.sum_time = 0;
		    it->second.number = 0;
		  //  std::cerr<<"Response time for server "<<it->first<<" is: "<<it->second.detect_ave.back()<<", "<<it->second.ave_num<<std::endl;        
		    std::ofstream myfile;
		    myfile.open("Detect.txt");
		    myfile << "Response time for server "<<it->first<<" is: "<<it->second.detect_ave.back()<<", "<<it->second.ave_num<<"\n";
		    myfile.close();
		}
	    }/*
 	    if(msg.meta.sender==8){
		detect1.sum_time += time_interval;
		detect1.number += 1;
	        //std::cerr<< " , response time is:"<<time_interval<<", average time so far:"<<detect1.sum_time/detect1.number<<" us."<<std::endl;
		if(detect1.number%100 == 0){
		    detect_ave1.push_back(detect1.sum_time/detect1.number);
		    detect1.sum_time = 0;
		    detect1.number = 0;
		    num_detect1 += 1;
		    std::cerr<<"Response time for server 1 is:"<<detect_ave1.back()<<" ts:"<<num_detect1<<std::endl;
		}
	    } else if(msg.meta.sender==10){
		detect2.sum_time += time_interval;
		detect2.number += 1;
	        //std::cerr<< " , response time is:"<<time_interval<<", average time so far:"<<detect2.sum_time/detect2.number<<" us."<<std::endl;
                if(detect2.number%100 == 0){
                    detect_ave2.push_back(detect2.sum_time/detect2.number);
                    detect2.sum_time = 0;
                    detect2.number = 0;
		    num_detect2 +=1;
		    std::cerr<<"Response time for server 2 is:"<<detect_ave2.back()<<" ts:"<<num_detect2<<std::endl;
                }
	    } else{
		detect3.sum_time += time_interval;
		detect3.number += 1;
	        //std::cerr<< " , response time is:"<<time_interval<<", average time so far:"<<detect3.sum_time/detect3.number<<" us."<<std::endl;
                if(detect3.number%100 == 0){
                    detect_ave3.push_back(detect3.sum_time/detect3.number);
                    detect3.sum_time = 0;
                    detect3.number = 0;
		    num_detect3 += 1;
		    std::cerr<<"Response time for server 3 is:"<<detect_ave3.back()<<" ts:"<<num_detect2<<std::endl;
                }
	    }*/   
	   }
	    //std::cerr<< " , response time is:"<<time_interval<<", average time so far:"<<detect.sum_time/detect.number<<" us."<<std::endl;
	}	
    }
    if (!msg.meta.control.empty()) {
      // control msg
      auto& ctrl = msg.meta.control;
      LOG(INFO)<<"NODES DEBUG STRING IS:"<<nodes.DebugString();
      if (ctrl.cmd == Control::TERMINATE) {
        ProcessTerminateCommand();
        break;
      } else if (ctrl.cmd == Control::ADD_NODE) {
        ProcessAddNodeCommand(msg, nodes, recovery_nodes);
	LOG(INFO)<<"After add node processing,node.debugstr:"<<nodes.DebugString();
      } else if (ctrl.cmd == Control::BARRIER) {
        ProcessBarrierCommand(msg);
      } else if (ctrl.cmd == Control::HEARTBEAT) {
        ProcessHearbeat(msg);
      } else if (ctrl.cmd == Control::INC_WORKER) {
    	ProcessIncWorkerCommand(msg, nodes);
      } else if (ctrl.cmd == Control::DEC_WORKER) {
    	ProcessDecWorkerCommand(msg, nodes);
      } else if (ctrl.cmd == Control::INC_SERVER) {
    	ProcessIncServerCommand(msg, nodes);
	LOG(INFO)<<"After inc server processing,node.debugstr:"<<nodes.DebugString();
      } else if (ctrl.cmd == Control::DEC_SERVER) {
    	ProcessDecServerCommand(msg, nodes);
      } else if (ctrl.cmd == Control::GETSTRAG) {
	//ProcessGetStragCommand(msg, nodes);
	 LOG(INFO)<<"ENTER INTO send command to upper layer!";
	 SendCommandToUpperLayer(kGetStragSignal,"1");
      }
    } else {
      ProcessDataMsg(msg);
    }
  }
}

/*
void Van::Receiving() {
  const char* heartbeat_timeout_val = Environment::Get()->find("PS_HEARTBEAT_TIMEOUT");
  const int heartbeat_timeout
      = heartbeat_timeout_val ? atoi(heartbeat_timeout_val) : kDefaultHeartbeatInterval;
  Meta nodes;  // for scheduler usage
  while (true) {
    Message msg;
    int recv_bytes = RecvMsg(&msg);

    // For debug, drop received message
    if (ready_ && drop_rate_ > 0) {
      unsigned seed = time(NULL) + my_node_.id;
      if (rand_r(&seed) % 100 < drop_rate_) {
        LOG(WARNING) << "Drop message " << msg.DebugString();
        continue;
      }
    }

    CHECK_NE(recv_bytes, -1);
    recv_bytes_ += recv_bytes;
    if (Postoffice::Get()->verbose() >= 2) {
      PS_VLOG(2) << msg.DebugString();
    }
    // duplicated message
    if (resender_ && resender_->AddIncomming(msg)) continue;

    if (!msg.meta.control.empty()) {
      // do some management
      auto& ctrl = msg.meta.control;
      if (ctrl.cmd == Control::TERMINATE) {
        PS_VLOG(1) << my_node_.ShortDebugString() << " is stopped";
        ready_ = false;
        break;
      } else if (ctrl.cmd == Control::ADD_NODE) {
        size_t num_nodes = Postoffice::Get()->num_servers() +
                           Postoffice::Get()->num_workers();
        auto dead_nodes = Postoffice::Get()->GetDeadNodes(heartbeat_timeout);
        std::unordered_set<int> dead_set(dead_nodes.begin(), dead_nodes.end());
        Meta recovery_nodes;  // store recovery nodes
        recovery_nodes.control.cmd = Control::ADD_NODE;
        // assign an id
        if (msg.meta.sender == Meta::kEmpty) {
          CHECK(is_scheduler_);  //scheduler runs this piece of code
          CHECK_EQ(ctrl.node.size(), 1);
          if (nodes.control.node.size() < num_nodes) {
            nodes.control.node.push_back(ctrl.node[0]);
          } else {
            // some node dies and restarts
            CHECK(ready_);
            for (size_t i = 0; i < nodes.control.node.size() - 1; ++i) { // the last node is scheduler...
              const auto& node = nodes.control.node[i];
              if (dead_set.find(node.id) != dead_set.end() && node.role == ctrl.node[0].role) {
                auto& recovery_node = ctrl.node[0];
                // assign previous node id
                recovery_node.id = node.id;
                recovery_node.is_recovery = true;
                PS_VLOG(1) << "replace dead node " << node.DebugString()
                           << " by node " << recovery_node.DebugString();
                nodes.control.node[i] = recovery_node;
                recovery_nodes.control.node.push_back(recovery_node);
                break;
              }
            }
          }
        }

        // update my id
        for (size_t i = 0; i < ctrl.node.size(); ++i) {
          const auto& node = ctrl.node[i];
          if (my_node_.hostname == node.hostname &&
              my_node_.port == node.port) {
            my_node_ = node;
            std::string rank = std::to_string(Postoffice::IDtoRank(node.id));
#ifdef _MSC_VER
            _putenv_s("DMLC_RANK", rank.c_str());
#else
            setenv("DMLC_RANK", rank.c_str(), true);
#endif
          }
        }

        if (is_scheduler_) {
          time_t t = time(NULL);
          if (nodes.control.node.size() == num_nodes) { //under recovery this also establishes?
            // sort the nodes according their ip and port,
            std::sort(nodes.control.node.begin(), nodes.control.node.end(),
                      [](const Node& a, const Node& b) {
                        return (a.hostname.compare(b.hostname) | (a.port < b.port)) > 0;
                      });
            // assign node rank
            for (auto& node : nodes.control.node) {
              CHECK_EQ(node.id, Node::kEmpty);
              int id = node.role == Node::SERVER ?
                       Postoffice::ServerRankToID(num_servers_) :
                       Postoffice::WorkerRankToID(num_workers_);
              PS_VLOG(1) << "assign rank=" << id << " to node " << node.DebugString();
              node.id = id;
              Connect(node);
              if (node.role == Node::SERVER) ++num_servers_;
              if (node.role == Node::WORKER) ++num_workers_;
              Postoffice::Get()->UpdateHeartbeat(node.id, t);
            }
            nodes.control.node.push_back(my_node_);
            nodes.control.cmd = Control::ADD_NODE;
            Message back; back.meta = nodes;
            for (int r : Postoffice::Get()->GetNodeIDs(
                     kWorkerGroup + kServerGroup)) {
              back.meta.recver = r;
              back.meta.timestamp = timestamp_++;
              Send(back);
            }
            PS_VLOG(1) << "the scheduler is connected to "
                    << num_workers_ << " workers and " << num_servers_ << " servers";
            ready_ = true;
          } else if (recovery_nodes.control.node.size() > 0) {
            // send back the recovery node
            CHECK_EQ(recovery_nodes.control.node.size(), 1);
            Connect(recovery_nodes.control.node[0]);
            Postoffice::Get()->UpdateHeartbeat(recovery_nodes.control.node[0].id, t);
            Message back;
            for (int r : Postoffice::Get()->GetNodeIDs(
                     kWorkerGroup + kServerGroup)) {
              if (r != recovery_nodes.control.node[0].id
                    && dead_set.find(r) != dead_set.end()) {
                // do not try to send anything to dead node
                continue;
              }
              // only send recovery_node to nodes already exist
              // but send all nodes to the recovery_node
              back.meta = (r == recovery_nodes.control.node[0].id) ? nodes : recovery_nodes;
              back.meta.recver = r;
              back.meta.timestamp = timestamp_++;
              Send(back);
            }
          }
        } else {  // if not scheduler
          for (const auto& node : ctrl.node) {
            Connect(node);
            if (!node.is_recovery && node.role == Node::SERVER) ++num_servers_;
            if (!node.is_recovery && node.role == Node::WORKER) ++num_workers_;
          }
          PS_VLOG(1) << my_node_.ShortDebugString() << " is connected to others";
          ready_ = true;
        }
      } else if (ctrl.cmd == Control::BARRIER) {
        if (msg.meta.request) {
          if (barrier_count_.empty()) {
            barrier_count_.resize(8, 0); // only valid for group 1,2,3,4,5,6,7, not individual node
          }
          int group = ctrl.barrier_group;
          ++barrier_count_[group];
          PS_VLOG(1) << "Barrier count for " << group << " : " << barrier_count_[group];
          if (barrier_count_[group] ==
              static_cast<int>(Postoffice::Get()->GetNodeIDs(group).size())) {
            barrier_count_[group] = 0;
            Message res;
            res.meta.request = false;
            res.meta.control.cmd = Control::BARRIER;
            for (int r : Postoffice::Get()->GetNodeIDs(group)) {
              res.meta.recver = r;
              res.meta.timestamp = timestamp_++;
              CHECK_GT(Send(res), 0);
            }
          }
        } else {
          Postoffice::Get()->Manage(msg);
        }
      } else if (ctrl.cmd == Control::HEARTBEAT) {
        time_t t = time(NULL);
        for (auto &node : ctrl.node) {
          Postoffice::Get()->UpdateHeartbeat(node.id, t);
          if (is_scheduler_) {
            Message heartbeat_ack;
            heartbeat_ack.meta.recver = node.id;
            heartbeat_ack.meta.control.cmd = Control::HEARTBEAT;
            heartbeat_ack.meta.control.node.push_back(my_node_);
            heartbeat_ack.meta.timestamp = timestamp_++;
            // send back heartbeat
            Send(heartbeat_ack);
          }
        }
      }
    } else {
      CHECK_NE(msg.meta.sender, Meta::kEmpty);
      CHECK_NE(msg.meta.recver, Meta::kEmpty);
      CHECK_NE(msg.meta.customer_id, Meta::kEmpty);
      int id = msg.meta.customer_id;
      auto* obj = Postoffice::Get()->GetCustomer(id, 5);
      CHECK(obj) << "timeout (5 sec) to wait App " << id << " ready";
      obj->Accept(msg);
    }
  }
}
*/

void Van::PackMeta(const Meta& meta, char** meta_buf, int* buf_size) {
  // convert into protobuf
  PBMeta pb;
  pb.set_head(meta.head);
  if (meta.customer_id != Meta::kEmpty) pb.set_customer_id(meta.customer_id);
  if (meta.timestamp != Meta::kEmpty) pb.set_timestamp(meta.timestamp);
  if (meta.body.size()) pb.set_body(meta.body);
  pb.set_push(meta.push);
  pb.set_request(meta.request);
  pb.set_simple_app(meta.simple_app);
  pb.set_response_time(meta.response_time);
  pb.set_start_time(meta.start_time);
  pb.set_receive_time(meta.receive_time);
  pb.set_end_time(meta.end_time);
  for (auto d : meta.data_type) pb.add_data_type(d);
  if (!meta.control.empty()) {
    auto ctrl = pb.mutable_control();
    ctrl->set_cmd(meta.control.cmd);
    if (meta.control.cmd == Control::BARRIER) {
      ctrl->set_barrier_group(meta.control.barrier_group);
    } else if (meta.control.cmd == Control::ACK) {
      ctrl->set_msg_sig(meta.control.msg_sig);
    }
    for (const auto& n : meta.control.node) {
      auto p = ctrl->add_node();
      p->set_id(n.id);
      p->set_role(n.role);
      p->set_port(n.port);
      p->set_hostname(n.hostname);
      p->set_is_recovery(n.is_recovery);
    }
  }

  // to string
  *buf_size = pb.ByteSize();
  *meta_buf = new char[*buf_size+1];
  CHECK(pb.SerializeToArray(*meta_buf, *buf_size))
      << "failed to serialize protbuf";
}

void Van::UnpackMeta(const char* meta_buf, int buf_size, Meta* meta) {
  // to protobuf
  PBMeta pb;
  CHECK(pb.ParseFromArray(meta_buf, buf_size))
      << "failed to parse string into protobuf";

  // to meta
  meta->head = pb.head();
  meta->customer_id = pb.has_customer_id() ? pb.customer_id() : Meta::kEmpty;
  meta->timestamp = pb.has_timestamp() ? pb.timestamp() : Meta::kEmpty;
  meta->request = pb.request();
  meta->push = pb.push();
  meta->simple_app = pb.simple_app();
  meta->response_time = pb.response_time();
  meta->start_time = pb.start_time();
  meta->receive_time = pb.receive_time();
  meta->end_time = pb.end_time();
  meta->body = pb.body();
  meta->data_type.resize(pb.data_type_size());
  for (int i = 0; i < pb.data_type_size(); ++i) {
    meta->data_type[i] = static_cast<DataType>(pb.data_type(i));
  }
  if (pb.has_control()) {
    const auto& ctrl = pb.control();
    meta->control.cmd = static_cast<Control::Command>(ctrl.cmd());
    meta->control.barrier_group = ctrl.barrier_group();
    meta->control.msg_sig = ctrl.msg_sig();
    for (int i = 0; i < ctrl.node_size(); ++i) {
      const auto& p = ctrl.node(i);
      Node n;
      n.role = static_cast<Node::Role>(p.role());
      n.port = p.port();
      n.hostname = p.hostname();
      n.id = p.has_id() ? p.id() : Node::kEmpty;
      n.is_recovery = p.is_recovery();
      meta->control.node.push_back(n);
    }
  } else {
    meta->control.cmd = Control::EMPTY;
  }
}

void Van::Heartbeat() {
  const char* val = Environment::Get()->find("PS_HEARTBEAT_INTERVAL");
  const int interval = val ? atoi(val) : kDefaultHeartbeatInterval;
  while (interval > 0 && ready_) {
    std::this_thread::sleep_for(std::chrono::seconds(interval));
    Message msg;
    msg.meta.recver = kScheduler;
    msg.meta.control.cmd = Control::HEARTBEAT;
    msg.meta.control.node.push_back(my_node_);
    msg.meta.timestamp = timestamp_++;
    Send(msg);
  }
}
}  // namespace ps
