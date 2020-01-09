#include <iostream>
#include <unordered_map>
#include <string>
#include <vector>
struct Detect{
	int overal_time = 0;
	int num = 0;
	std::vector<int> detect_ave;
	int ave_num = 0;
};


int main(int argc, char **argv) {
    struct Detect d1,d2;
    d2.overal_time += 123;
    d2.num +=1;
    d1.overal_time = d2.overal_time +199;
    d1.num +=1;
    std::unordered_map<int,struct Detect> map1;
    map1.insert({2,d2});
    map1.insert(std::make_pair(1,d1));
    std::cout << "contents:\n";
    for(auto& p:map1){
	p.second.detect_ave.push_back(10);
	std::cout<<" "<<p.first<<" => "<<p.second.overal_time<<", "<<p.second.num<<", "<<p.second.detect_ave.back()<<std::endl;
    }
    std::unordered_map<int, std::string> map;
    map.insert(std::make_pair(1, "Scala"));
    map.insert(std::make_pair(2, "Haskell"));
    map.insert(std::make_pair(3, "C++"));
    map.insert(std::make_pair(6, "Java"));
    map.insert(std::make_pair(14, "Erlang"));
    std::unordered_map<int, std::string>::iterator it;
    int num=14;
    if ((it = map.find(num)) != map.end()) {
        std::cout << it->second << std::endl;
    }
    return 0;
}
