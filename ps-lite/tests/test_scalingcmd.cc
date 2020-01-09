#include <unistd.h>
#include <iostream>
#include <fstream>
#include <algorithm>
//#include <glog/logging.h>


int main(){
	
	int node_id = 12;
        std::string fn = std::string("~/test/")+"SCALING.txt"+std::to_string(node_id);
        std::ifstream file(fn);
	if(file.good()){
		std::cerr<<" file exists!"<<std::endl;
	}else{
		std::cerr<<" NO FILE DETECTED!"<<std::endl;
		return 0;
	}
//	CHECK_NOTNULL(file.good()) << "SCALING file, "<<fn<<", not detected!";
	std::string scaling_cmd = "INC_SERVER";
        //std::string scaling_cmd;
        while(std::getline(file, scaling_cmd)){
                break;
        }
	if(scaling_cmd.size()){
		std::cerr<<" Node_id " << node_id << " get scaling command: "<< scaling_cmd<<std::endl;
	}else{
		std::cerr<< " No file detected!"<<scaling_cmd<<std::endl;
	}
	return 0;
}
