#include "ps/ps.h"
#include <unistd.h>
#include <iostream>
using namespace ps;

void StartServer() {
  if (!IsServer()) return;
  auto server = new KVServer<float>(0);
  server->set_request_handle(KVServerDefaultHandle<float>());
  RegisterExitCallback([server](){ delete server; });
}

void RunWorker() {
  if (!IsWorker()) return;
  KVWorker<float> kv(0);

  // init
  int num = 10000;
  std::vector<Key> keys(num);
  std::vector<float> vals(num);

  int rank = MyRank();
  srand(rank + 7);
  for (int i = 0; i < num; ++i) {
    keys[i] = kMaxKey / num * i + rank;
    vals[i] = (rand() % 1000);
  }

  // push
  int repeat = 10;
  std::vector<int> ts;
  for (int i = 0; i < repeat; ++i) {
    std::cout << "My rank:" << MyRank() << " Time: " << i << std::endl;
    ts.push_back(kv.Push(keys, vals));
    sleep(1);
    // to avoid too frequency push, which leads huge memory usage
    if (i > 10) kv.Wait(ts[ts.size()-10]);
  }
  for (int t : ts) kv.Wait(t);

  // pull
  std::vector<float> rets;
  kv.Wait(kv.Pull(keys, &rets));

  float res = 0;
  //for (int i = 0; i < num; ++i) {
  //  res += fabs(rets[i] - vals[i] * repeat);
  //}
  CHECK_LT(res / repeat, 1e-5);
  LL << "error: " << res / repeat;
}

int main(int argc, char *argv[]) {
  // setup server nodes
  StartServer();
  // start system
  Start();
  // run worker nodes
  RunWorker();
  // stop system
  Finalize();
  return 0;
}
