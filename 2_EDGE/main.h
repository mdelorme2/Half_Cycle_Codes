#ifndef MAIN_H 
	#define MAIN_H
	using namespace std;

	#include <iostream> 
	#include <math.h> 
	#include <cstdlib>
	#include <algorithm>
	#include "gurobi_c++.h"
	#include "time.h"
	#include "Allocation.h" 

	float EPSILON = 0.001;

	int edge(Allocation& allo, const int& K);

	class mycallback: public GRBCallback
	{
		public:
			int Ncuts;
			int nbNodes;
			int K;
			vector<vector<int> > edges;
			vector<bool> iea;
			vector<GRBVar> ieu;
	    
			mycallback(const vector<vector<int> >& xedges, const int& xnbNodes, const int& xK, const vector<bool>& xiea, const vector<GRBVar>& xieu);
			int getNcuts();

		protected:
			void callback();
	};

#endif 
