#include <fstream>
//#include <stdlib>
#include <string>
#include <vector>
#include <iostream>
using namespace std;
struct Opr_symbol{
	string Op;
	string name;
	vector<string> inputs;
	vector<int> keys;
};

vector<string> key_extract(string filename){
	ifstream myfile;
	myfile.open(filename);
	std::vector<string> keys;
	if(myfile.is_open()){
		string line;
		while(std::getline(myfile,line)){
			keys.push_back(line.substr(0,line.size()-1));
		}
	}
	return keys;
}
int main(){
	vector<Opr_symbol> opr_symbol;
	ifstream myfile;
	myfile.open("debug_str.txt");
	if(myfile.is_open()){
		string line;
		while(std::getline(myfile,line)){
			if(line!="--------------------"){
				continue;
			}else{
				string opr_name;
				getline(myfile,opr_name);
				char* pattern = ",";
				size_t pos = opr_name.find(pattern);
				string Op = opr_name.substr(3,pos-3);
				string name = opr_name.substr(pos+7,opr_name.size());
				getline(myfile,opr_name);
				std::vector<string> inputs;
				string line;
				while(true){
					std::getline(myfile,line);

					if(line[0]=='A' or line[0]=='V'){
						break;
					}else{
						inputs.push_back(line);
					}
				}
				struct Opr_symbol temp;
				temp.Op = Op;
				temp.name = name;
				temp.inputs = inputs;
				opr_symbol.push_back(temp);
			}
		}
	}
	std::cerr<<"Prepare to print\n";/*
	for(struct Opr_symbol &opr:opr_symbol){
	    std::cerr<<"opr:"<<opr.Op<<", name:"<<opr.name<<", input:\n"<<std::endl;
	    for(int i=0;i<opr.inputs.size();i++){		
		std::cerr<<opr.inputs[i]<<std::endl;
	    }
	}*/
	auto keys = key_extract("resnet-50.csv");	
	for(auto& key:keys){
	   // std::cerr<<key<<std::endl;
	}
//	std::cerr<<"********************key********************\n";
//	for(int i=0;i<keys.size();i++){
//	    std::cerr<<keys[i]<<std::endl;
//	}	
	int count = 0;
	string temp = "input:	arg[1]=bn_data_gamma(0) version=0";
	string pat = "bn_data_gamma";
	std::cerr<<"input:"<<temp<<std::endl;
	std::cerr<<"find:"<<temp.find(keys[0])<<", "<<temp[temp.find(keys[0])-1]<<" key_size:"<<keys[0].size()<<", last char:"<<keys[0][keys[0].size()-1]<<std::endl;
/*	for (int i=0;keys.size();i++){
		std::cerr<<"key:"<<keys[i]<<std::endl;
		auto pos = temp.find(keys[i]);
		std::cerr<<"finding result:"<<pos<<std::endl;
	}
*/	for(auto &opr:opr_symbol){
		
		for(auto &input:opr.inputs){
			count+=1;
			if(count<10){
			  //  std::cerr<<"input:"<<input<<std::endl;
			}
			for(int i=0;i<keys.size();i++){
			    if(count<10){
			//	std::cerr<<"***********key:********************:"<<keys[i]<<std::endl;
			    }
				auto pos = input.find(keys[i]);
				if(pos!=input.npos&&input[pos-1]=='='){
					opr.keys.push_back(i);
//					std::cerr<<" key:"<<i<<std::endl;
				}
			}
		}
	}
        for(struct Opr_symbol &opr:opr_symbol){
            std::cerr<<"opr:"<<opr.Op<<", name:"<<opr.name<<", input:\n"<<std::endl;
            for(int i=0;i<opr.inputs.size();i++){
                std::cerr<<opr.inputs[i]<<std::endl;
            }
	    std::cerr<<"keys:\n"<<std::endl;
	    for(int i=0;i<opr.keys.size();i++){
		std::cerr<<opr.keys[i]<<std::endl;
	    }
        }

	
}
