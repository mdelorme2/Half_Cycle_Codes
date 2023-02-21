#include "main.h"

/*	*************************************************************************************
	*************************************  MAIN *****************************************
	************************************************************************************* */

mycallback::mycallback(const vector<vector<int> >& xedges, const int& xnbNodes, const int& xK, const vector<bool>& xiea, const vector<GRBVar>& xieu){
    // initialize the Callback object
    this->Ncuts = 0;  
	this->nbNodes = xnbNodes;
	this->K = xK;
    this->edges = xedges;     
	this->iea = xiea;     
    this->ieu = xieu;
}

int mycallback::getNcuts() {
    return this->Ncuts;
}

void mycallback::callback() {
    if (where == GRB_CB_MIPSOL) { 
	
        // find the incumbent solution
        vector<int> next(nbNodes,-1);              			
		vector<GRBLinExpr> var(nbNodes);   
		
        for(int i = 0; i < edges.size(); i++){
			if(iea[i] && ceil(getSolution(ieu[i]) - EPSILON) == 1){
				next[edges[i][0]] = edges[i][1]; 
				var[edges[i][0]] = ieu[i];
			}
		}
		
        // find all cycles
		for(int i = 0; i < nbNodes; i++){
			if(next[i] >= 0){
				cout << i << " ";
				vector<GRBLinExpr> ele (K,0);
				int count = 1;
				ele[count] = var[i];
				int cur = next[i];
				next[i] = -1;
				while(cur != i){
					cout << cur << " ";
					count++;
					ele[count%K] = var[cur];
					if(count > K){
						GRBLinExpr sum = 0; 
						for(int j =0; j<K;j++) sum+=ele[j];
						addLazy(sum <= K - 1);  			
						Ncuts++;
						cout << "K ";
					}
					int temp = next[cur];					
					next[cur] = -1;
					cur = temp;
				}
				if(count <= K) cout << "OK";
				cout << endl;
			}
		}
		cout << "Added " << Ncuts << " cuts " << endl;
		cout << "-------------------------------------------------------" << endl;
	}
}

double initTimeModelCPU;

int main(int argc, char **argv){
	
	initTimeModelCPU = getCPUTime();
	
	// local variables
	Allocation allo ;
	string filein = argv[2];
	string path = argv[1];
	string pathAndFileout = argv[3];
	int K = atoi(argv[4]);
	
	// functions
	allo.load(path,filein);
	allo.printProb();
	allo.infos.timeCPU.push_back(0);
	allo.infos.K = K;
	
	edge(allo,K);
	allo.printInfo(pathAndFileout);
}


int edge(Allocation& allo, const int& K){

	// Local variables
	vector<vector<int> > sedges; for (int i = 0; i < allo.edges.size();i++) sedges.push_back({allo.itx[allo.edges[i][0]],allo.itx[allo.edges[i][1]]});
	vector<bool> iea(allo.edges.size(),false);
	
	// Create Floyd's matrix
	vector<vector<int> > flop (allo.nbNodes, vector<int> (allo.nbNodes, allo.nbNodes));
	for(int i = 0; i < allo.adj.size();i++){
		for(int j = 0; j < allo.adj[i].size();j++)
			flop[i][allo.adj[i][j]] = 1; 	
	}
	
	for(int i = 0; i < allo.nbNodes;i++){
		for(int j = 0; j < allo.nbNodes;j++){
			for(int k = 0; k < allo.nbNodes;k++)
				flop[j][k] = min(flop[j][k],flop[j][i] + flop[i][k]);
		}
	}
	
	allo.infos.timeCPU.push_back(getCPUTime() - initTimeModelCPU);
	
	GRBEnv env = GRBemptyenv;
	env.set(GRB_DoubleParam_MemLimit, 30);
	env.start();
	
	// Model
	try{
		// Local variables
		GRBModel model = GRBModel(env);
		GRBLinExpr objFun = 0;
		vector<GRBVar> isEdgeUsed (sedges.size());
		vector<GRBLinExpr> fi(allo.nbNodes);
		vector<GRBLinExpr> fo(allo.nbNodes);
		vector<GRBLinExpr> isNodeUsed(allo.nbNodes);

		// Initialization
		for (int i = 0; i < sedges.size(); i++){
			if(flop[sedges[i][1]][sedges[i][0]] <= K - 1){
				iea[i] = true;
				isEdgeUsed[i] = model.addVar(0, 1, 0, GRB_BINARY);
			}
		}

		for (int i = 0; i < allo.nbNodes; i++){
			isNodeUsed[i] = 0;
			fi[i] = 0;
			fo[i] = 0;
		}

		model.update();

		// Perform values
		for (int i = 0; i < sedges.size(); i++){
			if(iea[i]){
				isNodeUsed[sedges[i][1]] += isEdgeUsed[i];
				fi[sedges[i][1]] += isEdgeUsed[i];
				fo[sedges[i][0]] += isEdgeUsed[i];
				objFun += isEdgeUsed[i];
			}
		}

		// Flow conservation and use node once
		for (int i = 0; i < allo.nbNodes; i++){
			model.addConstr(isNodeUsed[i] <= 1);
			model.addConstr(fi[i] == fo[i]);
		}
				
		// Objective function
		model.setObjective(objFun, GRB_MAXIMIZE);
		
		// Setting of Gurobi
		model.getEnv().set(GRB_DoubleParam_TimeLimit,  3600);
		model.getEnv().set(GRB_IntParam_Method, 2);
		model.getEnv().set(GRB_IntParam_Threads, 1);
		model.getEnv().set(GRB_DoubleParam_MIPGap, 0);


		// create a callback
		mycallback cb = mycallback(sedges, allo.nbNodes, K, iea,isEdgeUsed);
		model.set(GRB_IntParam_LazyConstraints, 1); 
		model.setCallback(&cb);  
	
		model.optimize();
		
		// Filling Info
		allo.infos.timeCPU[0] = getCPUTime() - initTimeModelCPU;
		allo.infos.UB = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
		allo.infos.opt = false;

		// Get Info pre preprocessing
		allo.infos.nbVar =  model.get(GRB_IntAttr_NumVars);
		allo.infos.nbCons = model.get(GRB_IntAttr_NumConstrs);
		allo.infos.nbNZ = model.get(GRB_IntAttr_NumNZs);
		allo.infos.nbCuts = cb.getNcuts();
		
		// If no solution found
		if (model.get(GRB_IntAttr_SolCount) < 1){
			cout << "Failed to optimize ILP. " << endl;
			allo.infos.LB  = 0;
			return -1;
		}

		// If solution found
		allo.infos.LB = ceil(model.get(GRB_DoubleAttr_ObjVal) - EPSILON);	
		if(allo.infos.LB == allo.infos.UB) allo.infos.opt = true;

		GRBModel modelRelaxed = model.relax();
		modelRelaxed.optimize();
		allo.infos.contUB = modelRelaxed.get(GRB_DoubleAttr_ObjVal);

		// Filling Solution
		for (int i = 0; i < allo.edges.size(); i++){
			if(iea[i] && ceil(isEdgeUsed[i].get(GRB_DoubleAttr_X) - EPSILON) == 1){
				cout << allo.edges[i][0] << " " << allo.edges[i][1] << endl;
			}
		}
	}

	// Exceptions
	catch (GRBException e) {
		cout << "Error code = " << e.getErrorCode() << endl;
		cout << e.getMessage() << endl;
	}
	catch (...) {
		cout << "Exception during optimization" << endl;
	}


	// End
	return 0;
}
