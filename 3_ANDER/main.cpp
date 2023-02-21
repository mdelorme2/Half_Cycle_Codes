#include "main.h"

/*	*************************************************************************************
	*************************************  MAIN *****************************************
	************************************************************************************* */

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
	
	ander(allo,K);
	allo.printInfo(pathAndFileout);
}

int ander(Allocation& allo, const int& K){

	// Local variables
	vector<vector<vector<int> > > anedges(allo.nbNodes);
		
	/*for(int i =0; i<allo.nbNodes;i++){
		cout << ideg[i].first << " " << ideg[i].second << endl;
	}*/
	
	// Create Floyd's matrix
	/*vector<vector<int> > flop (allo.nbNodes, vector<int> (allo.nbNodes, allo.nbNodes));
	for(int i = 0; i < allo.adj.size();i++){
		for(int j = 0; j < allo.adj[i].size();j++)
			flop[i][allo.adj[i][j]] = 1; 	
	}
	
	for(int i = 0; i < allo.nbNodes;i++){
		for(int j = 0; j < allo.nbNodes;j++){
			for(int k = 0; k < allo.nbNodes;k++)
				flop[j][k] = min(flop[j][k],flop[j][i] + flop[i][k]);
		}
	}*/
	
	
	// Create ander cycles
	/*for(int i = 0; i < allo.nbNodes;i++){
		vector<vector<int> > ee (allo.nbNodes,vector<int> (allo.nbNodes, 0));
		vector<bool> tails (allo.nbNodes,false);
		tails[i] = true;
		for(int j = 1; j <= K; j++){
			vector<bool> heads (allo.nbNodes,false);
			for(int k = i; k < allo.nbNodes;k++){
				if(tails[k]){
					for(int l = 0; l < allo.adj[k].size();l++){
						if(allo.adj[k][l] == i || (allo.adj[k][l] > i && flop[allo.adj[k][l]][i] <= K - j)){
							heads[allo.adj[k][l]] = true;	
							if(ee[k][allo.adj[k][l]] == 0){
								anedges[i].push_back({k,allo.adj[k][l]});
								cout << "add " << i << " " << k << " " << allo.adj[k][l] << " at step " << j << endl;
								ee[k][allo.adj[k][l]] = 1;
							}
						}
					}
				}
			}
			tails = heads;
		}
	}		*/

	for(int i = 0; i < allo.nbNodes;i++){
		vector<vector<bool> > ee (allo.nbNodes,vector<bool>(allo.nbNodes,false));
		
		// Back
		vector<vector<int> > pe (K,vector<int> (allo.edges.size()));
		vector<vector<bool> > heads (K,vector<bool>(allo.nbNodes,false));
		heads[K-1][i] = true;
		for(int j = K-1; j >= 0; j--){
			for(int k = 0; k < allo.edges.size();k++){
				if(heads[j][allo.itx[allo.edges[k][1]]] && allo.itx[allo.edges[k][0]] >= i){
					for(int jp = j-1; jp>=0;jp--) heads[jp][allo.itx[allo.edges[k][0]]] = true;
					for(int jp = j; jp>=0;jp--) pe[jp][k] = true;
				}
			}
		}
		// Front 
		vector<vector<bool> > tails (K,vector<bool>(allo.nbNodes,false));
		tails[0][i] = true;
		for(int j = 0; j < K; j++){
			for(int k = 0; k < allo.edges.size();k++){
				if(pe[j][k] && tails[j][allo.itx[allo.edges[k][0]]]){
					if(j < K - 1) tails[j+1][allo.itx[allo.edges[k][1]]] = true;
					if(ee[allo.itx[allo.edges[k][0]]][allo.itx[allo.edges[k][1]]] == false){
						ee[allo.itx[allo.edges[k][0]]][allo.itx[allo.edges[k][1]]] = true;
				//		cout << "ADD " << allo.xti[i] << " " << allo.edges[k][0] << " " << allo.edges[k][1] << " pos " << j << endl;
						anedges[i].push_back({allo.itx[allo.edges[k][0]],allo.itx[allo.edges[k][1]]});
					}
				}
			}
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
		vector<vector<GRBVar> > isEdgeUsed (allo.nbNodes);
		vector<vector<GRBLinExpr> > fi(allo.nbNodes);
		vector<vector<GRBLinExpr> > fo(allo.nbNodes);
		vector<vector<bool> > fb(allo.nbNodes);
		vector<GRBLinExpr> isNodeUsed(allo.nbNodes);

		// Initialization
		for (int i = 0; i < allo.nbNodes; i++){
			isEdgeUsed[i].resize(anedges[i].size());
			for (int j = 0; j < anedges[i].size();j++){
				isEdgeUsed[i][j] = model.addVar(0, 1, 0, GRB_BINARY);
			}
			fi[i].resize(allo.nbNodes);
			fo[i].resize(allo.nbNodes);
			fb[i].resize(allo.nbNodes,false);
			isNodeUsed[i] = 0;
			for (int j = 0; j < allo.nbNodes; j++){
				fi[i][j]=0;
				fo[i][j]=0;
			}
		}

		model.update();

		// Perform values
		for (int i = 0; i < allo.nbNodes; i++){
			for (int j = 0; j < anedges[i].size(); j++){
				isNodeUsed[anedges[i][j][1]] += isEdgeUsed[i][j];
				fi[i][anedges[i][j][1]] += isEdgeUsed[i][j];
				fo[i][anedges[i][j][0]] += isEdgeUsed[i][j];
				fb[i][anedges[i][j][1]] = true;
				objFun += isEdgeUsed[i][j];
			}
		}

		// Flow conservation and use node once
		for (int i = 0; i < allo.nbNodes; i++){
			GRBLinExpr sum = 0;
			for (int j = 0; j < allo.nbNodes; j++){
				if(fb[i][j]){
					sum += fi[i][j];
					model.addConstr(fi[i][j] == fo[i][j]);
					model.addConstr(fi[i][j] <= fo[i][i]);
				}
			}
			model.addConstr(isNodeUsed[i] <= 1);
			model.addConstr(sum <= K);
		}
				
		// Objective function
		model.setObjective(objFun, GRB_MAXIMIZE);
				
		// Setting of Gurobi
		model.getEnv().set(GRB_DoubleParam_TimeLimit, 3600);
		model.getEnv().set(GRB_IntParam_Method, 2);
		model.getEnv().set(GRB_IntParam_Threads, 1);
		model.getEnv().set(GRB_DoubleParam_MIPGap, 0);
		model.optimize();
		
		// Filling Info
		allo.infos.timeCPU[0] = getCPUTime() - initTimeModelCPU;
		allo.infos.UB = ceil(model.get(GRB_DoubleAttr_ObjBound) - EPSILON);
		allo.infos.opt = false;

		// Get Info pre preprocessing
		allo.infos.nbVar =  model.get(GRB_IntAttr_NumVars);
		allo.infos.nbCons = model.get(GRB_IntAttr_NumConstrs);
		allo.infos.nbNZ = model.get(GRB_IntAttr_NumNZs);

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
		for (int i = 0; i < allo.nbNodes; i++){
			for (int j = 0; j < anedges[i].size(); j++){			
				if(ceil(isEdgeUsed[i][j].get(GRB_DoubleAttr_X) - EPSILON) == 1)
					cout << allo.xti[i] << " " << allo.xti[anedges[i][j][0]] << " " << allo.xti[anedges[i][j][1]] << endl;
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
