#ifndef ALLOCATION_H
	#define ALLOCATION_H
	
	using namespace std;
	#include <iostream> 
	#include <iomanip> 
	#include <fstream>
	#include <sstream>
	#include <vector>
	#include <string>
	#include <set>

	class Info;
	class Allocation;

/*	*************************************************************************************
	************************************* INFO ******************************************
	************************************************************************************* */

	class Info{
	public:
		bool opt;
		vector<double> timeCPU;
		int LB;
		int UB;
		float contUB;
		int nbCons;
		int nbVar;
		int nbNZ;
		int K;
	};

/*	*************************************************************************************
	********************************** ALLOCATION ***************************************
	************************************************************************************* */

	class Allocation{
	public:
		
		// Data read from the file
		string name;
		int nbNodes;
		vector<vector<int> > edges;
		vector<vector<int> > adj;
		vector<vector<int> > adm;
		vector<int> itx;
		vector<int> xti;
		
		// Given by the ILP model
		Info infos;

		void load(const string& path, const string& filein);
		void printProb();
		void printInfo(const string& pathAndFileout);
	};
	

#endif 