/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_VAN_H_
#define PS_INTERNAL_VAN_H_
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>
#include <thread>
#include <memory>
#include <atomic>
#include <ctime>
#include <unordered_set>
#include "ps/base.h"
#include "ps/internal/message.h"
namespace ps {
class Resender;
/**
 * \brief Van sends messages to remote nodes
 *
 * If environment variable PS_RESEND is set to be 1, then van will resend a
 * message if it no ACK messsage is received within PS_RESEND_TIMEOUT millisecond
 */
class Van {
 public:
  /**
   * \brief create Van
   * \param type zmq, socket, ...
   */
  static Van* Create(const std::string& type);
  /** \brief constructer, do nothing. use \ref Start for real start */
  Van() { }
  /**\brief deconstructer, do nothing. use \ref Stop for real stop */
  virtual ~Van() { }
  /**
   * \brief start van
   *
   * must call it before calling Send
   *
   * it initalizes all connections to other nodes.  start the receiving
   * threads, which keeps receiving messages. if it is a system
   * control message, give it to postoffice::manager, otherwise, give it to the
   * accoding app.
   */
  virtual void Start();
  /**
   * \brief send a message, It is thread-safe
   * \return the number of bytes sent. -1 if failed
   */
  int Send(const Message& msg);
  /**
   * \brief return my node
   */
  const Node& my_node() const {
    CHECK(ready_) << "call Start() first";
    return my_node_;
  }
  /**
   * \brief stop van
   * stop receiving threads
   */
  virtual void Stop();
  /**
   * \brief get next available timestamp. thread safe
   */
  int GetTimestamp() { return timestamp_++; }
  /**
   * \brief whether it is ready for sending. thread safe
   */
  bool IsReady() { return ready_; }
  bool IsTerminated() { return terminated_; }

 protected:
  /**
   * \brief connect to a node
   */
  virtual void Connect(const Node& node) = 0;
  /**
   * \brief bind to my node
   * do multiple retries on binding the port. since it's possible that
   * different nodes on the same machine picked the same port
   * \return return the port binded, -1 if failed.
   */
  virtual int Bind(const Node& node, int max_retry) = 0;
  /**
   * \brief block until received a message
   * \return the number of bytes received. -1 if failed or timeout
   */
  virtual int RecvMsg(Message* msg) = 0;
  /**
   * \brief send a mesage
   * \return the number of bytes sent
   */
  virtual int SendMsg(const Message& msg) = 0;
  /**
   * \brief pack meta into a string
   */
  void PackMeta(const Meta& meta, char** meta_buf, int* buf_size);
  /**
   * \brief unpack meta from a string
   */
  void UnpackMeta(const char* meta_buf, int buf_size, Meta* meta);

  Node scheduler_;
  Node my_node_;
  bool is_scheduler_;

 private:
  /** thread function for receving */
  void Receiving();
  /** thread function for heartbeat */
  void Heartbeat();
  /** whether it is ready for sending */
  std::atomic<bool> ready_{false};
  std::atomic<size_t> send_bytes_{0};
  size_t recv_bytes_ = 0;
  // current number of servers/workers
  int num_servers_ = 0;
  int num_workers_ = 0;
  int num_detect1 = 0;
  int num_detect2 = 0;
  int num_detect3 = 0;
  struct response_detect{
	int sum_time = 0;
	int number = 0;
	std::vector<int> detect_ave;
	int ave_num = 0;
  };
  struct response_detect detect1;
  struct response_detect detect2;
  struct response_detect detect3;
  std::vector<int> detect_ave1,detect_ave2,detect_ave3;
  std::unordered_map<int,struct response_detect> detect_map;
  //std::unordered_map<int,struct response_detect> server_response={{8,NULL},{10,NULL},{12,NULL}};
  // total number of servers/workers from the beginning during scaling process
  // only used for scheduler to allocate strictly increasing node id
  int hist_num_servers_ = 0;
  int hist_num_workers_ = 0;
  /** the thread for receiving messages */
  std::unique_ptr<std::thread> receiver_thread_;
  /** the thread for sending heartbeat */
  std::unique_ptr<std::thread> heartbeat_thread_;
  std::vector<int> barrier_count_;
  /** msg resender */
  Resender* resender_ = nullptr;
  int drop_rate_ = 0;
  std::atomic<int> timestamp_{0};
  DISALLOW_COPY_AND_ASSIGN(Van);

  /** refactor van::receiving() and devide it into serveral functions*/
  void ProcessAddNodeCommandAtScheduler(Message* msg, Meta& nodes, Meta& recovery_nodes);

  void ProcessTerminateCommand();

  void ProcessAddNodeCommand(Message& msg, Meta& nodes, Meta& recovery_nodes);

  void ProcessBarrierCommand(Message &msg);

  void ProcessHearbeat(Message &msg);

  void ProcessDataMsg(Message &msg);

  void UpdateLocalID(Message &msg, std::unordered_set<int> &deadnodes_set, Meta& nodes,
                         Meta &recovery_nodes);

  const char *heartbeat_timeout_val = Environment::Get()->find("PS_HEARTBEAT_TIMEOUT");
  int heartbeat_timeout = heartbeat_timeout_val ? atoi(heartbeat_timeout_val) : 0;

  /** @yhpeng the scaling_cmd is from k8s, can be NONE, INC_WORKER, DEC_WORKER, INC_SERVER, DEC_SERVER*/
  void ProcessIncWorkerCommand(Message& msg, Meta& nodes);
  void ProcessDecWorkerCommand(Message& msg, Meta& nodes);
  void ProcessIncServerCommand(Message& msg, Meta& nodes);
  void ProcessDecServerCommand(Message& msg, Meta& nodes);
  /** @yhpeng: send signal to SimpleApp to trigger scaling*/
  void SendCommandToUpperLayer(int req_head, const std::string& req_body);
  std::atomic<bool> terminated_{false};
  /** @yhpeng: save node ids for each barrier group in order to filter invalid ids later*/
  std::unordered_map<int, std::vector<int>> barrier_group_ids_;
  
  //@yrchen:global node list in scheduler.
  std::vector<Node> global_nodes;
};
}  // namespace ps
#endif  // PS_INTERNAL_VAN_H_
