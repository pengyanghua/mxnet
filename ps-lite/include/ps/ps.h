/*!
 *  Copyright (c) 2015 by Contributors
 * @file   ps.h
 * \brief  The parameter server interface
 */
#ifndef PS_PS_H_
#define PS_PS_H_
/** \brief basic setups in ps */
#include "ps/base.h"
/** \brief communicating with a pair of (int, string). */
#include "ps/simple_app.h"
/** \brief communcating with a list of key-value paris. */
#include "ps/kv_app.h"
namespace ps {
/** \brief Returns the number of worker nodes */
inline int NumWorkers() { return Postoffice::Get()->num_workers(); }
/** \brief Returns the number of server nodes */
inline int NumServers() { return Postoffice::Get()->num_servers(); }
/** \brief Returns true if this node is a worker node */
inline bool IsWorker() { return Postoffice::Get()->is_worker(); }
/** \brief Returns true if this node is a server node. */
inline bool IsServer() { return Postoffice::Get()->is_server(); }
/** \brief Returns true if this node is a scheduler node. */
inline bool IsScheduler() { return Postoffice::Get()->is_scheduler(); }
/** \brief Returns the rank of this node in its group
 *
 * Each worker will have a unique rank within [0, NumWorkers()). So are
 * servers. This function is available only after \ref Start has been called.
 */
inline int MyRank() { return Postoffice::Get()->my_rank(); }
inline int MyID() { return Postoffice::Get()->my_id(); }
inline int MapServerIDToRank(int id) {return Postoffice::Get()->map_server_id_to_rank(id); }
inline int MapWorkerIDToRank(int id) {return Postoffice::Get()->map_worker_id_to_rank(id); }

/**
 * \brief start the system
 *
 * This function will block until every nodes are started.
 * \param argv0 the program name, used for logging
 */
inline void Start(const char* argv0 = nullptr) {
  Postoffice::Get()->Start(argv0, true);
}
/**
 * \brief start the system
 *
 * This function will NOT block.
 * \param argv0 the program name, used for logging
 */
inline void StartAsync(const char* argv0 = nullptr) {
  Postoffice::Get()->Start(argv0, false);
}
/**
 * \brief terminate the system
 *
 * All nodes should call this function before existing. 
 * \param do_barrier whether to block until every node is finalized, default true.
 */
inline void Finalize(const bool do_barrier = true) {
  Postoffice::Get()->Finalize(do_barrier);
}
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
inline void RegisterExitCallback(const std::function<void()>& cb) {
  Postoffice::Get()->RegisterExitCallback(cb);
}
/*
struct KeyTime{
    time_t p_time;
    time_t h_time;
    int num = 0;
};
std::unordered_map<int,struct KeyTime> KTRecorder;*/
}  // namespace ps
#endif  // PS_PS_H_
