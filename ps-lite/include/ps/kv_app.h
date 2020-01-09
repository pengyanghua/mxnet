/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_KV_APP_H_
#define PS_KV_APP_H_
#include <unistd.h>
#include <algorithm>
#include <utility>
#include <vector>
#include "ps/base.h"
#include "ps/simple_app.h"
#include <sys/time.h>
#include <time.h>
#include <fstream>
namespace ps {
/*
struct KeyTime{
    time_t p_time;
    time_t h_time;
    int num = 0;
};
std::unordered_map<int,struct KeyTime> KTRecorder;*/

/**
 * \brief the structure for a list of key-value pairs
 *
 * The keys must be unique and sorted in an increasing order.  The length of a
 * value can be more than one. If \a lens is empty, then the length
 * of a value is determined by `k=vals.size()/keys.size()`.  The \a i-th KV pair
 * is then
 *
 * \verbatim {keys[i], (vals[i*k], ..., vals[(i+1)*k-1])} \endverbatim
 *
 * If \a lens is given, then `lens[i]` is the length of the \a i-th
 * value. Let
 *
 * \verbatim n = lens[0] + .. + lens[i-1]  \endverbatim
 *
* then the \a i-th KV pair is presented as
 *
 * \verbatim {keys[i], (vals[n], ..., vals[lens[i]+n-1])} \endverbatim
 */
template <typename Val>
struct KVPairs {
  // /** \brief empty constructor */
  // KVPairs() {}
  /** \brief the list of keys */
  SArray<Key> keys;
  /** \brief the according values */
  SArray<Val> vals;
  /** \brief the according value lengths (could be empty) */
  SArray<int> lens;
  /** \brief the version of the vals */
  SArray<int> vers;
};

/**
 * \brief A worker node that can \ref Push (\ref Pull) key-value pairs to (from) server
 * nodes
 *
 * \tparam Val the type of value, which should be primitive types such as
 * int32_t and float
 */
template<typename Val>
class KVWorker : public SimpleApp {
 public:
  /** avoid too many this-> */
  using SimpleApp::obj_;
  /**
   * \brief callback function for \ref Push and \ref Pull
   *
   * It is called by the data receiving thread of this instance when the push or
   * pull is actually finished. Namely the kv pairs have already written into
   * servers' data structure or the kv pairs have already pulled back.
   */
  using Callback = std::function<void()>;

  /**
   * \brief constructor
   *
   * \param app_id the app id, should match with \ref KVServer's id
   */
  explicit KVWorker(int app_id) : SimpleApp() {
    using namespace std::placeholders;
    slicer_ = std::bind(&KVWorker<Val>::DefaultSlicer, this, _1, _2, _3);
    obj_ = new Customer(app_id, std::bind(&KVWorker<Val>::Process, this, _1));
  }

  /** \brief deconstructor */
  virtual ~KVWorker() { delete obj_; obj_ = nullptr; }

  /**
   * \brief Pushes a list of key-value pairs to all server nodes.
   *
   * This function pushes a KV list specified by \a keys and \a vals to all
   * server nodes.
   *
   * Sample usage: the following codes push two KV pairs `{1, (1.1, 1.2)}` and `{3,
   * (3.1,3.2)}` to server nodes, where the value is a length-2 float vector
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals = {1.1, 1.2, 3.1, 3.2};
   *   w.Push(keys, vals);
   * \endcode
   *
   * If \a lens is given, then the value can be various length. See
   * \ref KVPairs for more information.
   *
   * The KV list is partitioned and sent based on the key range each server
   * maintaining. This function returns without waiting the data are sent
   * actually. Instead, use either \ref Wait or the callback to know when
   * finished. This function is thread-safe.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the according values
   * @param lens optional, lens[i] stores the value length of the \a
   * i-th KV pair
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the push is finished.
   * @return the timestamp of this request
   */
  int Push(const std::vector<Key>& keys,
           const std::vector<Val>& vals,
           const std::vector<int>& lens = {},
           int cmd = 0,
           const Callback& cb = nullptr) {
    return ZPush(
        SArray<Key>(keys), SArray<Val>(vals), SArray<int>(lens), cmd, cb);
  }

  /**
   * \brief Pulls the values associated with the keys from the server nodes
   *
   * This function pulls the values of the keys specified in \a keys from the
   * server nodes. The format is same to \ref KVPairs
   *
   * Sample usage: the following codes pull the values of keys \a 1 and \a 3
   * from the server nodes.
   * \code
   *   KVWorker<float> w;
   *   std::vector<Key> keys = {1, 3};
   *   std::vector<float> vals;
   *   ps.Pull(keys, &vals);
   * \endcode
   *
   * It's a non-blocking call. The actual pulling is finished,
   * namely \a vals (and \a lens) is filled with pulled values, only
   * if \ref Wait returns or the callback is called.
   *
   * @param keys a list of keys, must be unique and sorted in increasing order
   * @param vals the buffer for the pulled values. It can be 0 size.
   * @param lens optional buffer for the value length. If set, it can be 0 size.
   * @param cmd an optional command sent to the servers
   * @param cb the callback which is called when the pull is finished.
   * @return the timestamp of this request
   */
  int Pull(const std::vector<Key>& keys,
           std::vector<Val>* vals,
           std::vector<int>* lens = nullptr,
           int cmd = 0,
           const Callback& cb = nullptr) {
    return Pull_(SArray<Key>(keys), vals, lens, cmd, cb);
  }

  /**
   * \brief Waits until a push or pull has been finished
   *
   * Sample usage:
   * \code
   *   int ts = w.Pull(keys, &vals);
   *   Wait(ts);
   *   // now vals is ready for use
   * \endcode
   *
   * \param timestamp the timestamp returned by the push or pull
   */
  void Wait(int timestamp) { obj_->WaitRequest(timestamp); } // used once in kvstore_dist.h

  /**
   * \brief zero-copy Push
   *
   * This function is similar to \ref Push except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPush(const SArray<Key>& keys,
            const SArray<Val>& vals,
            const SArray<int>& lens = {},
            int cmd = 0,
            const Callback& cb = nullptr) {
    int ts = obj_->NewRequest(kServerGroup);
    AddCallback(ts, cb);
    KVPairs<Val> kvs;
    kvs.keys = keys;
    kvs.vals = vals;
    kvs.lens = lens;
    Send(ts, true, cmd, kvs);
    return ts;
  }

  /**
   * \brief zero-copy Pull
   *
   * This function is similar to \ref Pull except that all data
   * will not be copied into system for better performance. It is the caller's
   * responsibility to keep the content to be not changed before actually
   * finished.
   */
  int ZPull(const SArray<Key>& keys,
            SArray<Val>* vals,
            SArray<int>* lens = nullptr,
			int* step = 0, // here why int& step not work?
            int cmd = 0,
            const Callback& cb = nullptr) {
    return Pull_(keys, vals, lens, step, cmd, cb);
  }
  using SlicedKVs = std::vector<std::pair<bool, KVPairs<Val>>>;
  /**
   * \brief a slicer partitions a key-value list according to the key ranges
   * \param send the kv list for partitioning
   * \param ranges the key ranges, ranges[i] is the key range of server i
   * \param sliced the sliced lists. slices[i] should only contains keys in
   * ranges[i] and the according values
   */
  using Slicer = std::function<void(
      const KVPairs<Val>& send, const std::vector<Range>& ranges,
      SlicedKVs* sliced)>;

  /**
   * \brief set a user-defined slicer
   */
  void set_slicer(const Slicer& slicer) {
    CHECK(slicer); slicer_ = slicer;
  }

 private:
  /**
   * \brief internal pull, C/D can be either SArray or std::vector
   */
  template <typename C, typename D>
  int Pull_(const SArray<Key>& keys, C* vals, D* lens, int* step,
            int cmd, const Callback& cb);
  /**
   * \brief add a callback for a request. threadsafe.
   * @param cb callback
   * @param timestamp the timestamp of the request
   */
  void AddCallback(int timestamp, const Callback& cb) {
    if (!cb) return;
    std::lock_guard<std::mutex> lk(mu_);
    callbacks_[timestamp] = cb;
  }

  /**
   * \brief run and delete the callback
   * \param timestamp the timestamp of the callback
   */
  void RunCallback(int timestamp);
  /**
   * \brief send the kv list to all servers
   * @param timestamp the timestamp of the request
   * @param push whether or not it is a push request
   * @param cmd command
   */
  void Send(int timestamp, bool push, int cmd, const KVPairs<Val>& kvs);
  /** \brief internal receive handle */
  void Process(const Message& msg);
  /** \brief default kv slicer */
  void DefaultSlicer(const KVPairs<Val>& send,
                     const std::vector<Range>& ranges,
                     SlicedKVs* sliced);

  /** \brief data buffer for received kvs for each timestamp */
  std::unordered_map<int, std::vector<KVPairs<Val>>> recv_kvs_;
  /** \brief callbacks for each timestamp */
  std::unordered_map<int, Callback> callbacks_;
  /** \brief lock */
  std::mutex mu_;
  /** \brief kv list slicer */
  Slicer slicer_;
  struct KeyTime{
    time_t p_time;
    time_t h_time;
    int num = 0;
  };
  std::unordered_map<int,struct KeyTime> KTRecorder;
};

/** \brief meta information about a kv request */
struct KVMeta {
  /** \brief the int cmd */
  int cmd;
  /** \brief whether it is a request*/
  bool request;
  /** \brief whether or not this is a push request */
  bool push;
  /** \brief sender's node id */
  int sender;
  /** \brief the associated timestamp */
  int timestamp;
  time_t response_time;
  time_t receive_time;
  time_t start_time;
  time_t end_time;
};

/**
 * \brief A server node for maintaining key-value pairs
 */
template <typename Val>
class KVServer : public SimpleApp {
 public:
  /**
   * \brief constructor
   * \param app_id the app id, should match with \ref KVWorker's id
   */
  explicit KVServer(int app_id) : SimpleApp() {
    using namespace std::placeholders;
    obj_ = new Customer(app_id, std::bind(&KVServer<Val>::Process, this, _1));
  }

  /** \brief deconstructor */
  virtual ~KVServer() { delete obj_; obj_ = nullptr; }

  /**
   * \brief the handle to process a push/pull request from a worker
   * \param req_meta meta-info of this request
   * \param req_data kv pairs of this request
   * \param server this pointer
   */
  using ReqHandle = std::function<void(const KVMeta& req_meta,
                                       const KVPairs<Val>& req_data,
                                       KVServer* server)>;
  void set_request_handle(const ReqHandle& request_handle) {
    CHECK(request_handle) << "invalid request handle";
    request_handle_ = request_handle;
  }

  /**
   * \brief response to the push/pull request
   * \param req the meta-info of the request
   * \param res the kv pairs that will send back to the worker
   */
  void Response(const KVMeta& req, const KVPairs<Val>& res = KVPairs<Val>());
  void Send(int req_head, const KVPairs<Val>& res, int recv_id);

 private:
  /** \brief internal receive handle */
  void Process(const Message& msg);
  /** \brief request handle */
  ReqHandle request_handle_;
};


/**
 * \brief an example handle adding pushed kv into store
 */
template <typename Val>
struct KVServerDefaultHandle {
  void operator()(
      const KVMeta& req_meta, const KVPairs<Val>& req_data, KVServer<Val>* server) {
    size_t n = req_data.keys.size();
    KVPairs<Val> res;
    if (req_meta.push) {
      CHECK_EQ(n, req_data.vals.size());
    } else {
      res.keys = req_data.keys; res.vals.resize(n);
    }
    for (size_t i = 0; i < n; ++i) {
      Key key = req_data.keys[i];
      if (req_meta.push) {
        store[key] += req_data.vals[i];
      } else {
        res.vals[i] = store[key];
      }
    }
    struct timeval tv;
    gettimeofday(&tv,NULL);
    
    //std::cerr<<"In KVServerDefaultHandle, request time:"<<req_meta.response_time<<" response time:"<<(tv.tv_usec-req_meta.response_time)<<std::endl;
    server->Response(req_meta, res);
  }
  std::unordered_map<Key, Val> store;
};


///////////////////////////////////////////////////////////////////////////////

template <typename Val>
void KVServer<Val>::Process(const Message& msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg); return; // call Process in simpleapp
  }
  KVMeta meta;
  meta.cmd       = msg.meta.head;
  meta.push      = msg.meta.push;
  meta.sender    = msg.meta.sender;
  meta.timestamp = msg.meta.timestamp;
  meta.request   = msg.meta.request;
  meta.response_time = msg.meta.response_time;
  //struct timeval tv;
  //gettimeofday(&tv,NULL);
  meta.receive_time = msg.meta.receive_time;
  meta.start_time = msg.meta.start_time;
  KVPairs<Val> data;
  int n = msg.data.size();
  if (n) {
    CHECK_GE(n, 1);
    data.keys = msg.data[0];
    if (n >= 2) {
      data.vals = msg.data[1];
      if (n >= 3){
        data.lens = msg.data[2];
        CHECK_EQ(data.lens.size(), data.keys.size());
        if (n >= 4){
    	  data.vers = msg.data[3];
    	  CHECK_EQ(data.vers.size(), data.keys.size());
        }
      }
    }
  }
  CHECK(request_handle_); // handle both request and response message
  
  struct timeval tv;  
  gettimeofday(&tv,NULL);
  int time_interval = tv.tv_usec - msg.meta.start_time;
//  std::cerr<<"In KVServerProcess, request time:"<<msg.meta.response_time<<" response time:"<<time_interval<<std::endl;
  request_handle_(meta, data, this);
}

template <typename Val>
void KVServer<Val>::Response(const KVMeta& req, const KVPairs<Val>& res) {
  Message msg;
  msg.meta.customer_id = obj_->id();
  msg.meta.request     = false;
  msg.meta.push        = req.push;
  msg.meta.head        = req.cmd;
  msg.meta.timestamp   = req.timestamp;
  msg.meta.recver      = req.sender;
  msg.meta.response_time = req.response_time;
  msg.meta.start_time = req.start_time;
  msg.meta.receive_time = req.receive_time;
  if(msg.meta.receive_time==0&&msg.meta.push==0){
//	LOG(INFO)<<"ERROR RECEIVE_TIME IS 0! start:"<<msg.meta.start_time
//	<<" receive:"<<msg.meta.receive_time;
  }
  if (res.keys.size()) { // pull response
    msg.AddData(res.keys); // only key is available for parameter movement
    if (res.vals.size()){ // value available for pull response
      msg.AddData(res.vals);
    }
    if (res.lens.size()) {
      msg.AddData(res.lens);
    }
    if (res.vers.size()) {
      msg.AddData(res.vers);
    }
  }
  struct timeval tv;
  gettimeofday(&tv,NULL);
  time_t time_interval = 1000000*tv.tv_sec+tv.tv_usec - msg.meta.start_time;
  //std::cerr<<"In KVServerResponse, request time:"<<msg.meta.response_time<<" response time:"<<time_interval<<std::endl;

  //std::cerr<<"In KVServerResponse, request time:"<<msg.meta.response_time<<" response time:"<<(time(NULL)-msg.meta.response_time)<<std::endl;
  //if(time_interval<0)
//	LOG(INFO)<<"SERVER response time < 0 :"<<time_interval<<" start:"<<msg.meta.start_time;
//  msg.meta.response_time = time_interval-msg.meta.receive_time;
  msg.meta.response_time = 1000000*tv.tv_sec+tv.tv_usec - msg.meta.receive_time;

  time_t h_time = msg.meta.response_time - msg.meta.receive_time;
 
  if((h_time<0||h_time>1000000)&&msg.meta.push==0){
	//LOG(INFO)<<"KeyTimeRecorder error! h_time:"<<h_time<<", res:"
	//<<msg.meta.response_time<<", rcv:"<<msg.meta.receive_time;	
  }	
  Postoffice::Get()->van()->Send(msg);
}


template <typename Val>
void KVServer<Val>::Send(int req_head, const KVPairs<Val>& res, int recv_id) {
  Message msg;
  msg.meta.customer_id = obj_->id();
  msg.meta.request     = true;
  msg.meta.push        = true;
  msg.meta.head        = req_head;
  int ts = obj_->NewRequest(recv_id);
  msg.meta.timestamp   = ts;

  if (res.keys.size()) {
    msg.AddData(res.keys);
    msg.AddData(res.vals);
    msg.AddData(res.lens);
    msg.AddData(res.vers);
  }
  // send
  for (int r : Postoffice::Get()->GetNodeIDs(recv_id)) {
    msg.meta.recver = r;
    Postoffice::Get()->van()->Send(msg);
  }
}


template <typename Val>
void KVWorker<Val>::DefaultSlicer(
    const KVPairs<Val>& send, const std::vector<Range>& ranges,
    typename KVWorker<Val>::SlicedKVs* sliced) {
  sliced->resize(ranges.size());

  // find the positions in msg.key
  size_t n = ranges.size();
  std::vector<size_t> pos(n+1);
  const Key* begin = send.keys.begin();
  const Key* end = send.keys.end();
  for (size_t i = 0; i < n; ++i) {
    if (i == 0) {
      pos[0] = std::lower_bound(begin, end, ranges[0].begin()) - begin;
      begin += pos[0];
    } else {
      CHECK_EQ(ranges[i-1].end(), ranges[i].begin());
    }
    size_t len = std::lower_bound(begin, end, ranges[i].end()) - begin;
    begin += len;
    pos[i+1] = pos[i] + len;

    // don't send it to severs for empty kv
    sliced->at(i).first = (len != 0);
  }
  CHECK_EQ(pos[n], send.keys.size());
  if (send.keys.empty()) return;

  // the length of value
  size_t k = 0, val_begin = 0, val_end = 0;
  if (send.lens.empty()) {
    k = send.vals.size() / send.keys.size();
    CHECK_EQ(k * send.keys.size(), send.vals.size());
  } else {
    CHECK_EQ(send.keys.size(), send.lens.size());
  }

  // slice
  for (size_t i = 0; i < n; ++i) {
    if (pos[i+1] == pos[i]) {
      sliced->at(i).first = false;
      continue;
    }
    sliced->at(i).first = true;
    auto& kv = sliced->at(i).second;
    kv.keys = send.keys.segment(pos[i], pos[i+1]);
//	LOG(INFO)<<"IN defaultslicer, keys segment";
    if (send.lens.size()) {
//	LOG(INFO)<<"IN defaultslicer, lens segment";
      kv.lens = send.lens.segment(pos[i], pos[i+1]);
      for (int l : kv.lens) val_end += l;
//	LOG(INFO)<<"IN defaultslicer, lens segment";
      kv.vals = send.vals.segment(val_begin, val_end);
      val_begin = val_end;
    } else {
      kv.vals = send.vals.segment(pos[i]*k, pos[i+1]*k);
    }
  }
}

template <typename Val>
void KVWorker<Val>::Send(int timestamp, bool push, int cmd, const KVPairs<Val>& kvs) {
  // slice the message
  SlicedKVs sliced;
  slicer_(kvs, Postoffice::Get()->GetServerKeyRanges(), &sliced);  // yhpeng: the keys need to be sorted

  // need to add response first, since it will not always trigger the callback
  int skipped = 0;
  for (size_t i = 0; i < sliced.size(); ++i) {
    if (!sliced[i].first) ++skipped;
  }
  obj_->AddResponse(timestamp, skipped);
  if ((size_t)skipped == sliced.size()) {
    RunCallback(timestamp);
  }
  for (size_t i = 0; i < sliced.size(); ++i) {
    const auto& s = sliced[i];
    struct timeval tv;
    gettimeofday(&tv,NULL);
    if (!s.first) continue;
    Message msg;
    msg.meta.customer_id = obj_->id();
    msg.meta.request     = true;
    msg.meta.push        = push;
    msg.meta.head        = cmd;
    msg.meta.timestamp   = timestamp;
 //   std::cerr<<"Enter into KVWorker Send!"<<std::endl;
    msg.meta.start_time = 1000000*tv.tv_sec+tv.tv_usec;
    //msg.meta.response_time = 1000000*tv.tv_sec+tv.tv_usec;
    //LOG(INFO)<<"Worker call send, push:"<<push<<" to server "<<i<<" id:"<<Postoffice::Get()->map_server_rank_to_id(i);
    msg.meta.recver      = Postoffice::Get()->map_server_rank_to_id(i); // yhpeng: may trigger issues here if scaling
    if(msg.meta.recver==0){
	    LOG(INFO)<<"Worker call send, push:"<<push<<" to server "<<i<<" id:"<<Postoffice::Get()->map_server_rank_to_id(i)<< ",recver:"<<msg.meta.recver;
	msg.meta.recver = i*2+8;
    }
    //LOG(INFO)<<"Debug start_time:"<<msg.meta.start_time; 
    const auto& kvs = s.second;
    if (kvs.keys.size()) {
      msg.AddData(kvs.keys);
      msg.AddData(kvs.vals);
      if (kvs.lens.size()) {
        msg.AddData(kvs.lens);
      }
      if (kvs.vers.size()){
    	msg.AddData(kvs.vers);
      }
    }
//    std::cerr<<"In KVWorker Send, request time:"<<msg.meta.response_time<<std::endl;
    Postoffice::Get()->van()->Send(msg);
  }
}


template <typename Val>
void KVWorker<Val>::Process(const Message& msg) {
  if (msg.meta.simple_app) {
    SimpleApp::Process(msg); return;
  }

  // store the data for pulling
  int ts = msg.meta.timestamp;
  if (!msg.meta.push && msg.data.size()) {
    CHECK_GE(msg.data.size(), (size_t)2);
    KVPairs<Val> kvs;
    kvs.keys = msg.data[0];
    kvs.vals = msg.data[1];
    if (msg.data.size() > (size_t)2) {
      kvs.lens = msg.data[2];
      if (msg.data.size() > (size_t)3){
    	  kvs.vers = msg.data[3];
      }
    }
    int size = kvs.lens[0];
    struct timeval tv;
    gettimeofday(&tv,NULL);

    time_t p_time = 1000000*tv.tv_sec+tv.tv_usec - msg.meta.start_time - msg.meta.response_time;
//    time_t h_time = msg.meta.response_time - msg.meta.receive_time;
    time_t h_time = msg.meta.response_time;
//    time_t p_time = msg.meta.end_time - h_time - msg.meta.start_time;
    if(h_time<0||p_time<0){
//	std::cerr<<"key time error:\n";
//	std::cerr<<"start_time:"<<msg.meta.start_time<<", receive_time:"
//	<<msg.meta.receive_time<<", response_time:"<<msg.meta.response_time
//	<<", end_time:"<<msg.meta.end_time<<std::endl;
    }
    //std::unordered_map<int, struct KeyTime>::iterator it;
    auto it = KTRecorder.find(size);
    if (it != KTRecorder.end()) {
	it->second.p_time += p_time;
	it->second.h_time += h_time;
	it->second.num += 1;
	if(it->second.p_time <0 || it->second.h_time<0){
//		LOG(INFO)<<"KeyTimeRecord error, p_time:"<<it->second.p_time<<" h_time:"
//		<<it->second.h_time<<" num:"<<it->second.num;
	}
        //std::cout << it->second << std::endl;
	if (it->second.num % 100 == 0){
		
	        std::ofstream myfile;
      		char buf[80];
	        getcwd(buf,80);
	        std::string wd;
	        wd = std::string(buf) + std::string("/results/");
 	        std::string file_name = wd+std::string("keyTimeRecorder_")+std::to_string(Postoffice::Get()->my_id())+".csv";
	        myfile.open(file_name,std::ios_base::app);
            myfile<<"from,"<<msg.meta.sender<<", size,"<<size<<", avg propagation time,"<<it->second.p_time/it->second.num
                << ", avg handling time,"<<it->second.h_time/it->second.num<<std::endl;
	        myfile.close();
		it->second.p_time = 0;
		it->second.h_time = 0;
		it->second.num = 0;
	}
    } else{
	struct KeyTime kt;
	kt.p_time = p_time;
	kt.h_time = h_time;
	KTRecorder.insert(std::make_pair(size,kt));
    }   
//    std::cerr<<"Size,"<<kvs.lens<<" keys,"<<kvs.keys<<" start_time,"<<msg.meta.start_time
//	<<" receive_time,"<<msg.meta.receive_time<<" response_time,"<<msg.meta.response_time
//	<<" end_time,"<<msg.meta.end_time<<std::endl;
    mu_.lock();
    recv_kvs_[ts].push_back(kvs); // store the parameters of a key from many servers
    mu_.unlock();
    int64_t pull_receive = std::chrono::duration_cast<std::chrono::microseconds>(
    std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    Key key = kvs.keys[0];
    auto kr_scaling = ps::Postoffice::Get()->GetServerKeyRanges_Scaling(0);
    auto key_temp = key;
    for(int i = 0; i<Postoffice::Get()->num_servers();i++){
        auto id = 8 + i*2; 
        auto rank = Postoffice::Get()->map_server_rank_to_id(id);
        auto temp = kr_scaling[i];
        key_temp = key - temp.begin();
        if(key_temp>=0 &&key_temp<=100000){break;}
    }
    if(key_temp>=0 && key_temp<=100000){
        //return key_temp;
    }else{
        LOG(ERROR)<<"DECODE FAIL! "<<key_temp;
    } 
    

    std::ofstream myfile;
    /*
    myfile.open("engine_output.txt",std::ios_base::app);
    myfile<<"Receiving pull response, key:"<<kvs.keys<<",time:"<<pull_receive<<"\n";
    */
      char buf[80];
      getcwd(buf,80);
    //std::cerr<<"Buf of getcwd is:"<<buf<<std::endl;
      std::string wd;
      wd = std::string(buf) + std::string("/results/");
     // wd = wd + "engine_output1.txt";
      std::string file_name = wd+std::string("key_response_")+std::to_string(Postoffice::Get()->my_id())+".csv";
//    std::string file_name = "~/elastic-mxnet/example/image-classification/results/key_response_"+std::to_string(Postoffice::Get()->my_id())+".csv";
    myfile.open(file_name,std::ios_base::app);
    myfile<<"from server "<<msg.meta.sender<<",key "<< key_temp<<','<<pull_receive<<std::endl;
    myfile.close();    
    //std::cerr<<"Enter into KVWorker::Process, Worker "<<Postoffice::Get()->my_id()<<", keys: "<<kvs.keys<<", time:"<<pull_receive<<std::endl;
  }

  // finished, run callbacks
  // check Customer::Receiving() in customer.cc
  if (obj_->NumResponse(ts) == Postoffice::Get()->num_servers() - 1)  {
    RunCallback(ts);
  }
}

template <typename Val>
void KVWorker<Val>::RunCallback(int timestamp) {
  mu_.lock();
  auto it = callbacks_.find(timestamp);
  if (it != callbacks_.end()) {
    mu_.unlock();

    CHECK(it->second);
    it->second();

    mu_.lock();
    callbacks_.erase(it);
  }
  mu_.unlock();
}

template <typename Val>
template <typename C, typename D>
int KVWorker<Val>::Pull_(const SArray<Key>& keys, C* vals, D* lens, int* step, int cmd, const Callback& cb) {

  //keys must be sorted
  int ts = obj_->NewRequest(kServerGroup);
  // call after receiving all pull responses for a KVPairs (a key in upper layer)
  AddCallback(ts, [this, ts, keys, vals, lens, step, cb]() mutable {
      mu_.lock();
      auto& kvs = recv_kvs_[ts];
      mu_.unlock();

	
      // do check
      size_t total_key = 0, total_val = 0;
      for (const auto& s : kvs) { // s is a KVPairs
        Range range = FindRange(keys, s.keys.front(), s.keys.back()+1);
        CHECK_EQ(range.size(), s.keys.size())
            << "unmatched keys size from one server";
        if (lens) CHECK_EQ(s.lens.size(), s.keys.size());
        total_key += s.keys.size();
        total_val += s.vals.size();
      }
      CHECK_EQ(total_key, keys.size()) << "lost some servers?";
      
      // fill vals and lens
      std::sort(kvs.begin(), kvs.end(), [](
          const KVPairs<Val>& a, const KVPairs<Val>& b) {
                  return a.keys.front() < b.keys.front();
        }); // sort kvs based on key
      CHECK_NOTNULL(vals);
      if (vals->empty()) {
        vals->resize(total_val);
      } else {
        CHECK_EQ(vals->size(), total_val);
      }
      Val* p_vals = vals->data(); // vals is an SArray

      //lens is optional
      int *p_lens = nullptr;
      if (lens) {
        if (lens->empty()) {
          lens->resize(keys.size());
        } else {
	  if(lens->size()!=keys.size()){
		std::cerr<<"Incosistent key for pull. key:";
		for(auto key:keys){
			std::cerr<<key;
		}
		std::cerr<<"\n";
	  }
	  //std::cerr<<"CHECK_EQ: lens->size:"<<lens->size()<<" keys.size:"<<keys.size();
          CHECK_EQ(lens->size(), keys.size());
        }
        p_lens = lens->data();
      }
      // copy vals/lens
      for (const auto& s : kvs) {
        memcpy(p_vals, s.vals.data(), s.vals.size() * sizeof(Val)); // additional copy here
        p_vals += s.vals.size();
        if (p_lens) {
          memcpy(p_lens, s.lens.data(), s.lens.size() * sizeof(int));
          p_lens += s.lens.size();
        }
      }

      // should be same for all KVPairs
      *step = kvs[0].vers[0];
      //memcpy(step->data(), kvs[0].vers.data(), kvs[0].vers.size()*sizeof(int));

      mu_.lock();
      recv_kvs_.erase(ts);
      mu_.unlock();
      if (cb) cb();
    });

  KVPairs<Val> kvs; kvs.keys = keys;
  Send(ts, false, cmd, kvs);
  return ts;
}

}  // namespace ps
#endif  // PS_KV_APP_H_
