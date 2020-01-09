/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_POSTOFFICE_H_
#define PS_INTERNAL_POSTOFFICE_H_
#include <mutex>
#include <algorithm>
#include <vector>
#include "ps/range.h"
#include "ps/internal/env.h"
#include "ps/internal/customer.h"
#include "ps/internal/message.h"
#include "ps/internal/van.h"
namespace ps {
/**
 * \brief the center of the system
 */
class Postoffice {
 public:
  /**
   * \brief return the singleton object
   */
  static Postoffice* Get() {
    static Postoffice e; return &e;
  }
  /** \brief get the van */
  Van* van() { return van_; }
  /**
   * \brief start the system
   *
   * This function will block until every nodes are started.
   * \param argv0 the program name, used for logging.
   * \param do_barrier whether to block until every nodes are started.
   */
  void Start(const char* argv0, const bool do_barrier);
  /**
   * \brief terminate the system
   *
   * All nodes should call this function before existing. 
   * \param do_barrier whether to do block until every node is finalized, default true.
   */
  void Finalize(const bool do_barrier = true);
  /**
   * \brief add an customer to the system. threadsafe
   */
  void AddCustomer(Customer* customer);
  /**
   * \brief remove a customer by given it's id. threasafe
   */
  void RemoveCustomer(Customer* customer);
  /**
   * \brief get the customer by id, threadsafe
   * \param id the customer id
   * \param timeout timeout in sec
   * \return return nullptr if doesn't exist and timeout
   */
  Customer* GetCustomer(int id, int timeout = 0) const;
  /**
   * \brief get the id of a node (group), threadsafe
   *
   * if it is a  node group, return the list of node ids in this
   * group. otherwise, return {node_id}
   */
  const std::vector<int>& GetNodeIDs(int node_id) const {
    const auto it = node_ids_.find(node_id);
    CHECK(it != node_ids_.cend()) << "node " << node_id << " doesn't exist";
    return it->second;
  }
  /**
   * \brief return the key ranges of all server nodes
   */
  const std::vector<Range>& GetServerKeyRanges();
  const std::vector<Range>& GetServerKeyRanges_Scaling(int i = 0);
  /**
   * \brief the template of a callback
   */
  using Callback = std::function<void()>;
  /**
   * \brief Register a callback to the system which is called after Finalize()
   *
   * The following codes are equal
   * \code {cpp}
   * RegisterExitCallback(cb);
   * Finalize();
   * \endcode
   *
   * \code {cpp}
   * Finalize();
   * cb();
   * \endcode
   * \param cb the callback function
   */
  void RegisterExitCallback(const Callback& cb) {
    exit_callback_ = cb;
  }
  /**
   * \brief convert from a worker rank into a node id
   * \param rank the worker rank
   */
  static inline int WorkerRankToID(int rank) {
    return rank * 2 + 9;
  }
  /**
   * \brief convert from a server rank into a node id
   * \param rank the server rank
   */
  static inline int ServerRankToID(int rank) {
    return rank * 2 + 8;
  }
  /**
   * \brief convert from a node id into a server or worker rank
   * \param id the node id
   */
  static inline int IDtoRank(int id) {
#ifdef _MSC_VER
#undef max
#endif
    return std::max((id - 8) / 2, 0);
  }
  /** \brief Returns the number of worker nodes */
  int num_workers() const { return num_workers_; }
  /** \brief Returns the number of server nodes */
  int num_servers() const { return num_servers_; }
  /** \brief Returns the rank of this node in its group
   *
   * Each worker will have a unique rank within [0, NumWorkers()). So are
   * servers. This function is available only after \ref Start has been called.
   */
  int my_rank() const { return IDtoRank(van_->my_node().id); }
  int my_id() const { return van_->my_node().id; }
  /** \brief Returns true if this node is a worker node */
  int is_worker() const { return is_worker_; }
  /** \brief Returns true if this node is a server node. */
  int is_server() const { return is_server_; }
  /** \brief Returns true if this node is a scheduler node. */
  int is_scheduler() const { return is_scheduler_; }
  /** whether this node is added later*/
  int is_added_node() const {return is_added_node_;}
  /** whether this node is deleted now*/
  int is_deled_node() const {return is_deled_node_;}
  /** \brief Returns the verbose level. */
  int verbose() const { return verbose_; }
  /** \brief Return whether this node is a recovery node */
  bool is_recovery() const { return van_->my_node().is_recovery; }
  /**
   * \brief barrier
   * \param node_id the barrier group id
   */
  void Barrier(int node_id);
  /**
   * \brief process a control message, called by van
   * \param the received message
   */
  void Manage(const Message& recv);
  /**
   * \brief update the heartbeat record map
   * \param node_id the \ref Node id
   * \param t the last received heartbeat time
   */
  void UpdateHeartbeat(int node_id, time_t t) {
    std::lock_guard<std::mutex> lk(heartbeat_mu_);
    heartbeats_[node_id] = t;
  }
  /**
   * \brief delete heartbeat record from the map
   */
  void DelHeartbeat(int node_id) {
	std::lock_guard<std::mutex> lk(heartbeat_mu_);
	heartbeats_.erase(node_id);
  }
  /**
   * \brief get node ids that haven't reported heartbeats for over t seconds
   * \param t timeout in sec
   */
  std::vector<int> GetDeadNodes(int t = 60);

  /** @yhpeng update num_server and num_worker here*/
  void UpdateNumWorkers(int num_workers){
	num_workers_ = num_workers;
  }
  void UpdateNumServers(int num_servers){
	num_servers_ = num_servers;
  }

  void UpdateNodeIDs(const int id, const bool addID);
  void ClearNodeIDs(){
	node_ids_.clear();
  }

  std::string GetScalingCMD(){
	return scaling_cmd;
  }
  // when scaling, change node_ids first before call this
  int map_server_id_to_rank(int server_id){
	CHECK(!(server_id%2));
	if((size_t)num_servers_ != server_id_to_rank_.size())
	{
		std::vector<int> servers;
		for (const auto& server : node_ids_[kServerGroup]){
			servers.push_back(server);
		}
		struct Comp {
		  bool operator() (int i,int j) { return (i<j);}
		} comp;
		std::sort(servers.begin(), servers.end(), comp);
		for(size_t i=0; i<servers.size(); i++){
			server_id_to_rank_[servers[i]] = i;
		}
	}
	return server_id_to_rank_[server_id];
  }

  int map_server_rank_to_id(int server_rank){
	if((size_t)num_servers_ != server_rank_to_id_.size())
	{
		std::vector<int> servers;
		for (const auto& server : node_ids_[kServerGroup]){
			servers.push_back(server);
		}
		struct Comp {
		  bool operator() (int i,int j) { return (i<j);}
		} comp;
		std::sort(servers.begin(), servers.end(), comp);
		for(size_t i=0; i<servers.size(); i++){
			server_rank_to_id_[i] = servers[i];
		}
	}
	return server_rank_to_id_[server_rank];
  }

  int map_worker_id_to_rank(int worker_id){
	CHECK(worker_id%2);
	if((size_t)num_workers_ != worker_id_to_rank_.size())
	{
		std::vector<int> workers;
		for (const auto& worker : node_ids_[kWorkerGroup]){
			workers.push_back(worker);
		}
		struct Comp {
		  bool operator() (int i,int j) { return (i<j);}
		} comp;
		std::sort(workers.begin(), workers.end(), comp);
		for(size_t i=0; i<workers.size(); i++){
			worker_id_to_rank_[workers[i]] = i;
		}
	}
	return worker_id_to_rank_[worker_id];
  }

 private:
  Postoffice();
  ~Postoffice() { delete van_; }
  Van* van_;
  mutable std::mutex mu_;
  std::unordered_map<int, Customer*> customers_;
  std::unordered_map<int, std::vector<int>> node_ids_;
  std::unordered_map<int, int> server_id_to_rank_;
  std::unordered_map<int, int> server_rank_to_id_;
  std::unordered_map<int, int> worker_id_to_rank_;
  std::vector<Range> server_key_ranges_;
  std::vector<Range> server_key_ranges_scaling_;
  bool is_worker_, is_server_, is_scheduler_;
  bool is_added_node_; // @yhpeng: whether this node is a inc node
  bool is_deled_node_; // @yhpeng: whether this node is a del node
  int num_servers_, num_workers_;
  bool barrier_done_;
  bool is_exit = false;  // @yhpeng: set exit flag to let thread exit
  int verbose_;
  std::mutex barrier_mu_;
  std::condition_variable barrier_cond_;
  std::mutex heartbeat_mu_;
  std::unordered_map<int, time_t> heartbeats_;
  Callback exit_callback_;
  /** \brief Holding a shared_ptr to prevent it from being destructed too early */
  std::shared_ptr<Environment> env_ref_;
  time_t start_time_;
  DISALLOW_COPY_AND_ASSIGN(Postoffice);

  /** @yhpeng the scaling_cmd is from k8s, can be NONE, INC_WORKER, DEC_WORKER, INC_SERVER, DEC_SERVER*/
  std::string scaling_cmd;
  /** the thread for checking scaling_cmd */
  std::unique_ptr<std::thread> check_scaling_cmd_thread_;
  /** thread function for checking scaling_cmd */
  void CheckScalingCMD();

};

/** \brief verbose log */
#define PS_VLOG(x) LOG_IF(INFO, x <= Postoffice::Get()->verbose())
}  // namespace ps
#endif  // PS_INTERNAL_POSTOFFICE_H_
